"""CPU synthetic fixture: strict checkpoint load/finite output and upstream FR parity.

This is an engineering check, not an ImageNet evaluation or E0/E1 result.
"""
import _bootstrap
import argparse
import gc
import torch
from psp_research.registry import TARGETS, load_target
from psp_research.provenance import write_json
from psp_research.metrics import normalize
from psp_research.evaluator import evaluate_batches

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-lock', default='configs/targets/weights.lock.json')
    p.add_argument('--weights-dir', default='models_cache')
    p.add_argument('--output', default='outputs/model-fixtures.json')
    a = p.parse_args()
    from functions import normalize as original_normalize
    from loss import get_fooling_rate
    report = {}
    torch.set_num_threads(2)
    x = torch.linspace(0, 1, 3*224*224).reshape(1, 3, 224, 224)
    for name in TARGETS:
        model, _, s = load_target(name, a.weights_lock, a.weights_dir, 'cpu')
        with torch.inference_mode():
            y = model(normalize(x, s['mean'], s['std']))
        assert y.shape == (1, 1000) and torch.isfinite(y).all()
        report[name] = dict(strict_load=True, finite_1x1000_cpu_fixture=True, weights_sha256=s['weights_sha256'])
        if name == 'resnet50':
            torch.testing.assert_close(normalize(x, s['mean'], s['std']), original_normalize(x), rtol=0, atol=0)
            for d in [torch.zeros_like(x), torch.full_like(x, 10/255)]:
                old = get_fooling_rate(model, d, [(x, torch.tensor([0]))], 'cpu')
                new = evaluate_batches(model, [(x, torch.tensor([0]), ['synthetic_fixture'])], d, s['mean'], s['std'])
                assert old == new['fr']
            report[name]['upstream_evaluator_fixture_parity'] = True
        print(name, 'PASS', flush=True)
        del model
        gc.collect()
    write_json(a.output, dict(purpose='CPU synthetic fixture, NOT ImageNet/E1 results', targets=report))
