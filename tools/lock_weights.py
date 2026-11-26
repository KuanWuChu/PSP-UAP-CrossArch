"""Download explicit official checkpoints, validate prefix, freeze full SHA256."""
import _bootstrap
import argparse
from psp_research.registry import lock_weights, TARGETS, E0_TARGETS

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', nargs='+', choices=TARGETS | E0_TARGETS, default=list(TARGETS))
    p.add_argument('--weights-dir', default='models_cache')
    p.add_argument('--output', default='configs/targets/weights.lock.json')
    a = p.parse_args()
    lock_weights(a.targets, a.weights_dir, a.output)
