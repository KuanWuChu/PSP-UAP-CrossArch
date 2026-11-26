"""Re-evaluate the original returned UAP on all50000 validation images and five CNNs.

This deliberately includes the original stopping selection and is separate from
the disjoint49000 research evaluation. No target gradients or checkpoint tuning.
"""
import _bootstrap
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from psp_research.data import verify_bundle, read_manifest
from psp_research.evaluator import ManifestDataset, evaluate_batches
from psp_research.metrics import load_uap
from psp_research.registry import E0_NAMES, load_target
from psp_research.provenance import check_upstream, git_identity, sha256, write_json


def execute(a):
    check_upstream()
    index = verify_bundle(a.manifest_dir)
    rows = sorted(read_manifest(Path(a.manifest_dir)/'selection.csv') +
                  read_manifest(Path(a.manifest_dir)/'holdout.csv'), key=lambda r:r['image_id'])
    if len(rows) != 50000 or len({r['image_id'] for r in rows}) != 50000:
        raise ValueError('E0 requires all50000 unique original images')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    delta = load_uap(a.uap).to(a.device)
    report = dict(git_identity(), stage='e0_full_validation', status='running',
        execution_label=a.execution_label, uap_sha256=sha256(a.uap),
        manifest_hashes=index['manifest_hashes'], data_source=index['source'],
        stopping_selection_overlap=True, targets={},
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda),
        weights_lock_sha256=sha256(a.weights_lock))
    write_json(out/'provenance.json',report)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for name in E0_NAMES:
            model, geometry, weights = load_target(name,a.weights_lock,a.weights_dir,a.device)
            loader = DataLoader(ManifestDataset(a.data_root,rows,geometry),batch_size=a.batch_size,num_workers=a.workers,shuffle=False)
            start=time.perf_counter()
            with (out/(name+'-predictions.csv')).open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(['image_id','label','clean_prediction','adv_prediction'])
                result=evaluate_batches(model,loader,delta,weights['mean'],weights['std'],writer)
            if str(a.device).startswith('cuda'): torch.cuda.synchronize()
            if result['total_count'] != 50000: raise ValueError('E0 count mismatch')
            result.update(target=name,weights_id=weights['weights_id'],weights_sha256=weights['weights_sha256'],
                transform_sha256=weights['transform_sha256'],elapsed_seconds=time.perf_counter()-start)
            report['targets'][name]=result
            write_json(out/(name+'-metrics.json'),result)
            write_json(out/'provenance.json',report)
            print(name,json.dumps(result),flush=True)
            del model
        report['status']='complete'
    except Exception as e:
        report.update(status='failed',error_reason=type(e).__name__+': '+str(e)); raise
    finally:
        write_json(out/'provenance.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uap',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--manifest-dir',required=True)
    p.add_argument('--weights-lock',default='configs/targets/e0_weights.lock.json')
    p.add_argument('--weights-dir',default='models_cache')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--execution-label',required=True,choices=['upstream_modern_compatibility','upstream_legacy_core'])
    p.add_argument('--output',required=True)
    execute(p.parse_args())
