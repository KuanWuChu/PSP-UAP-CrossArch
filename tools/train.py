"""Fixed-budget PSP research trainer; full runs require a verified manifest bundle.

No target outputs are used for checkpoint selection. --dry-run never trains.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import _bootstrap
import argparse
import json
import time
import uuid
from pathlib import Path
import torch
from psp_research.config import load_config, validate
from psp_research.data import verify_bundle
from psp_research.provenance import git_identity, check_upstream, write_json, sha256, digest
from psp_research.registry import load_target
from psp_research.trainer import ControlledTrainer


def execute(a):
    c = load_config(a.config)
    if a.smoke_steps is not None:
        if not 1 <= a.smoke_steps <= 200:
            raise ValueError('smoke steps must be 1..200')
        c['max_steps'] = a.smoke_steps
        c = validate(c)
    resolved = dict(config=c, seed=a.seed, purpose='smoke' if a.smoke_steps else 'full',
                    execution_label='research_fixed_budget', device=a.device,
                    config_sha256=digest(c))
    if a.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    out = Path(a.output or ('outputs/train-' + uuid.uuid4().hex[:12]))
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(git_identity(), **resolved, run_id='train-' + uuid.uuid4().hex[:12], status='running', error_reason=None)
    write_json(out / 'config.resolved.json', resolved)
    write_json(out / 'provenance.json', provenance)
    start = time.perf_counter()
    try:
        check_upstream()
        if not a.manifest_dir:
            raise ValueError('BLOCKED: supply provenance-verified --manifest-dir; no real training without validated data pipeline')
        index = verify_bundle(a.manifest_dir)
        if index.get('dataset') != 'ILSVRC2012_validation' or index.get('counts') != {'selection': 1000, 'holdout': 49000, 'smoke_holdout': 1000}:
            raise ValueError('expected full ImageNet validation manifest bundle')
        provenance['manifest_hashes'] = index['manifest_hashes']
        provenance['data_source'] = index['source']
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, _, weights = load_target(c['surrogate'], a.weights_lock, a.weights_dir, a.device)
        provenance.update(weights_id=weights['weights_id'], weights_sha256=weights['weights_sha256'],
            environment=dict(torch=torch.__version__, cuda=torch.version.cuda, deterministic_algorithms=True))
        runner = ControlledTrainer(model, c, a.seed, a.device)
        if a.resume:
            runner.resume(a.resume)
            provenance['resumed_checkpoint_sha256'] = sha256(a.resume)
        if str(a.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        runner.run(a.stop_after if a.stop_after is not None else c['max_steps'], out)
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize()
            provenance['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated()
        provenance.update(exogenous_draws_sha256=runner.exogenous_draws_sha256, prior_draws_sha256=runner.prior_draws_sha256,
                          training_implementation_sha256=runner.implementation_sha256, step=runner.step, status='complete' if runner.step == c['max_steps'] else 'incomplete')
        if provenance['status'] == 'complete':
            provenance['uap_sha256'] = sha256(out / 'uap.npy')
    except Exception as e:
        provenance.update(status='failed', error_reason=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        provenance['elapsed_seconds'] = time.perf_counter() - start
        write_json(out / 'provenance.json', provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, choices=[126, 2025, 3407], required=True)
    p.add_argument('--manifest-dir')
    p.add_argument('--weights-lock', default='configs/targets/weights.lock.json')
    p.add_argument('--weights-dir', default='models_cache')
    p.add_argument('--output')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int, help='absolute step to stop early; incomplete runs export no final UAP')
    p.add_argument('--smoke-steps', type=int, help='separately labeled 1..200 step engineering smoke budget')
    p.add_argument('--dry-run', action='store_true')
    execute(p.parse_args())
