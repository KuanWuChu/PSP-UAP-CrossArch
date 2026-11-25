"""Evaluate a real .npy UAP; raw RGB add -> clip -> checkpoint normalization."""
import _bootstrap
import argparse
from psp_research.registry import TARGETS
from psp_research.evaluator import evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uap', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--targets', nargs='+', choices=TARGETS, default=[t for t in TARGETS if t != 'resnet152'])
    p.add_argument('--weights-lock', default='configs/targets/weights.lock.json')
    p.add_argument('--weights-dir', default='models_cache')
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['smoke', 'selection', 'final'], required=True)
    p.add_argument('--epsilon', type=float, default=10/255)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    evaluate(p.parse_args())
