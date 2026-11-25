"""Independent replayable Python/NumPy/torch CPU/CUDA streams for upstream helpers."""
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch


def capture(use_cuda=False):
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=n[0], keys=n[1].tolist(), pos=n[2],
                has_gauss=n[3], cached_gaussian=n[4]), cpu=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if use_cuda else [])


def restore(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32), n['pos'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG topology changed; exact resume refused')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class RNGStreams:
    def __init__(self, seed, use_cuda=False):
        self.use_cuda = use_cuda
        self.states = {}
        outer = capture(use_cuda)
        try:
            for name, offset in [('init', 11), ('prior', 23), ('crop', 37), ('transform', 53)]:
                random.seed(seed + offset)
                np.random.seed(seed + offset)
                torch.random.default_generator.manual_seed(seed + offset)
                if use_cuda:
                    torch.cuda.manual_seed_all(seed + offset)
                self.states[name] = capture(use_cuda)
        finally:
            restore(outer)

    @contextmanager
    def use(self, name):
        outer = capture(self.use_cuda)
        restore(self.states[name])
        try:
            yield
        finally:
            self.states[name] = capture(self.use_cuda)
            restore(outer)

    def state_dict(self):
        return copy.deepcopy(self.states)

    def load_state_dict(self, states):
        if set(states) != set(self.states):
            raise ValueError('RNG stream names mismatch')
        self.states = copy.deepcopy(states)
