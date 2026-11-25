from dataclasses import dataclass
import torch


@dataclass
class Counts:
    total_count: int = 0
    flip_count: int = 0
    clean_correct_count: int = 0
    adv_correct_count: int = 0
    clean_correct_became_wrong_count: int = 0

    def update(self, labels, clean, adv):
        if labels.ndim != 1 or labels.shape != clean.shape or labels.shape != adv.shape:
            raise ValueError('predictions and labels must be equal length vectors')
        self.total_count += labels.numel()
        self.flip_count += int((clean != adv).sum())
        cc = clean == labels
        self.clean_correct_count += int(cc.sum())
        self.adv_correct_count += int((adv == labels).sum())
        self.clean_correct_became_wrong_count += int((cc & (adv != labels)).sum())

    def result(self):
        if not self.total_count:
            raise ValueError('empty evaluation')
        return dict(vars(self), fr=self.flip_count / self.total_count,
                    clean_acc=self.clean_correct_count / self.total_count,
                    adv_acc=self.adv_correct_count / self.total_count,
                    cc_asr=(self.clean_correct_became_wrong_count / self.clean_correct_count
                            if self.clean_correct_count else None),
                    cc_asr_reason=None if self.clean_correct_count else 'no_clean_correct_samples')


def normalize(x, mean, std):
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(std)[None, :, None, None]


def perturb(x, delta):
    if x.ndim != 4 or x.shape[1] != 3 or delta.shape != (1, 3, *x.shape[-2:]):
        raise ValueError('expected RGB NCHW and one same-resolution UAP')
    if not torch.isfinite(x).all() or not torch.isfinite(delta).all():
        raise ValueError('non-finite input')
    if x.min() < 0 or x.max() > 1:
        raise ValueError('images must be raw RGB in [0,1]')
    return (x + delta).clamp(0, 1)


def load_uap(path, epsilon=10 / 255):
    import numpy as np
    a = np.load(path, allow_pickle=False)
    if a.dtype != np.float32 or a.shape != (1, 3, 224, 224):
        raise ValueError('UAP must be float32 [1,3,224,224] .npy in raw RGB units')
    if not np.isfinite(a).all() or np.abs(a).max() > epsilon + 1e-7:
        raise ValueError('UAP non-finite or exceeds epsilon; refusing silent repair')
    return torch.from_numpy(a.copy())
