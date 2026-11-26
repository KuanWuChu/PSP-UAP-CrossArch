"""Fixed-budget research loop. Activation objective and helpers imported unchanged.

Differences from upstream's stopping/checkpoint flow are documented in TRAINER_RUNBOOK_ZH.md.
"""
import copy
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torchvision import transforms
import loss as upstream_loss
import semantic_aug as aug
import strategy
from .config import validate
from .provenance import digest, sha256, ROOT, UPSTREAM
from .rng import RNGStreams, capture, restore

TRACE_FIELDS = ['step', 'loss', 'prior_batch', 'samples', 'prior_sha256', 'crop_boxes_sha256',
                'crop_area_min', 'crop_area_max', 'crop_area_mean', 'delta_linf', 'saturation', 'rescaled']


def finite(tensor, name):
    if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
        raise FloatingPointError('non-finite ' + name)


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, x in model.state_dict().items():
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def numerical_policy():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)


def build_sources(prior, delta, source_mode, num_copies, area, ratio):
    """Call inside crop RNG context. Sources detached; only outer delta has gradient."""
    if source_mode not in {'noise_crops', 'psp_crops'}:
        raise ValueError('unknown source mode')
    base = prior.detach() if source_mode == 'noise_crops' else prior.detach() + delta.detach()
    base = base.repeat(num_copies, 1, 1, 1)
    crops, boxes = [], []
    for image in base:
        cropped, h, w, top, left = aug.random_crop(image, scale=area, ratio=ratio)
        crops.append(transforms.Resize(list(delta.shape[-2:]), antialias=True)(cropped))
        boxes.append([h, w, top, left])
    source = torch.stack(crops).detach()
    return source + delta, source.clone(), boxes


def paired_transforms(semantic_delta, source, args, prior, device):
    # Stack instead of in-place writes to keep autograd versions valid in modern torch.
    images, sources = [], []
    for image, teacher in zip(semantic_delta, source):
        p = torch.rand(1).item()
        if p < 1/3:
            image, teacher = aug.rotate_fill_prior(image, teacher, args, prior, device=device)
        elif p < 2/3:
            image, teacher = aug.scaling_transform(image, teacher, args, device=device)
        else:
            image, teacher = aug.shuffle_only(image, teacher, num_block=args.shuffle_block)
        images.append(image)
        sources.append(teacher)
    return torch.stack(images), torch.stack(sources).detach()


class ControlledTrainer:
    def __init__(self, model, config, seed, device='cpu'):
        self.config = validate(config)
        if seed not in self.config['seeds']:
            raise ValueError('seed must be predeclared')
        self.seed, self.device = seed, torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.model_identity = model_hash(self.model)
        self.numerical_policy = numerical_policy()
        self.implementation_sha256 = digest({p: sha256(ROOT / p) for p in ['psp_research/trainer.py', 'psp_research/rng.py', 'psp_research/config.py', 'loss.py', 'functions.py', 'semantic_aug.py', 'strategy.py']})
        self.rng = RNGStreams(seed, self.device.type == 'cuda')
        with self.rng.use('init'):
            # Same uniform distribution and sign convention as the original.
            self.delta = (self.config['epsilon'] * (1 - 2 * torch.rand(1, 3, 224, 224, device=self.device))).requires_grad_()
        self.optimizer = torch.optim.Adam([self.delta], lr=self.config['lr'])
        self.step = 0
        self.saturation = 0.0
        self.exogenous_draws_sha256 = digest({'stream': 'prior+crop'})
        self.prior_draws_sha256 = digest({'stream': 'prior'})
        c = self.config
        self.args = SimpleNamespace(surrogate_model=c['surrogate'], delta_size=224, prior=c['prior'],
            prior_batch=1, std=10, fre=1, gauss_t0=400, gauss_gamma=10, jigsaw_t0=600,
            jigsaw_gamma=1, jigsaw_end_iter=4200, p_active=c['p_active'], p_rate=c['p_rate'],
            num_copies=c['num_copies_per_prior'], temper=c['temper'], angle=c['angle'],
            scale_t_low=c['scale_t_low'], scale_t_high=c['scale_t_high'], shuffle_block=c['shuffle_block'])

    def advance(self):
        if self.step >= self.config['max_steps']:
            raise ValueError('fixed budget exhausted')
        self.optimizer.zero_grad(set_to_none=True)
        iteration = self.step + 1
        with self.rng.use('prior'):
            if self.args.prior == 'gauss':
                strategy.curriculum_strategy_gauss(iteration, self.args)
                prior = upstream_loss.get_gauss_prior(self.args)
            else:
                strategy.curriculum_strategy_jigsaw_resnet152(iteration, self.args)
                # Original always-true jigsaw branch deliberately retained in loss.py.
                prior = upstream_loss.get_jigsaw(self.delta, self.args, filter=True)
        finite(prior, 'random prior')
        prior_sha = tensor_hash(prior)
        prior = prior.to(self.device)
        with self.rng.use('crop'):
            images, sources, boxes = build_sources(prior, self.delta, self.config['source_mode'],
                self.args.num_copies, self.config['crop_area'], self.config['aspect_ratio'])
        if self.config['input_transform']:
            with self.rng.use('transform'):
                images, sources = paired_transforms(images, sources, self.args, prior, self.device)
        finite(images, 'semantic samples')
        finite(sources, 'detached source')
        if self.config['re_weight']:
            objective = upstream_loss.l2_layer_loss_weight(self.model, images, sources, self.args,
                                                         self.args.num_copies, self.device)
        else:
            objective = upstream_loss.l2_layer_loss(self.model, images, self.args, self.device)
        finite(objective, 'original activation loss')
        objective.backward()
        finite(self.delta.grad, 'delta gradient')
        self.optimizer.step()
        # Check before clamp: clamp must not hide overflow from an optimizer update.
        finite(self.delta, 'post-update delta')
        eps = self.config['epsilon']
        with torch.no_grad():
            self.delta.clamp_(-eps, eps)
        sat = upstream_loss.get_rate_of_saturation(self.delta.detach().cpu().numpy(), eps)
        rescaled = abs(sat - self.saturation) < 0.00001 and sat > 0.5
        self.saturation = float(sat)
        if rescaled:
            with torch.no_grad():
                self.delta.div_(2)
        self.step = iteration
        self.exogenous_draws_sha256 = digest(dict(previous=self.exogenous_draws_sha256, step=iteration, prior=prior_sha, crops=digest(boxes)))
        self.prior_draws_sha256 = digest(dict(previous=self.prior_draws_sha256, step=iteration, prior=prior_sha))
        areas = [h*w/(224*224) for h,w,_,_ in boxes]
        return dict(step=self.step, loss=float(objective.detach()), prior_batch=self.args.prior_batch,
                    samples=len(boxes), prior_sha256=prior_sha, crop_boxes_sha256=digest(boxes),
                    crop_area_min=min(areas), crop_area_max=max(areas), crop_area_mean=sum(areas)/len(areas),
                    delta_linf=float(self.delta.detach().abs().max()), saturation=self.saturation, rescaled=bool(rescaled))

    def snapshot(self):
        finite(self.delta, 'checkpoint delta')
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    finite(value, 'optimizer state')
        return dict(schema_version=1, upstream_commit=UPSTREAM, implementation_sha256=self.implementation_sha256, delta=self.delta.detach().clone().cpu(),
            optimizer=copy.deepcopy(self.optimizer.state_dict()), step=self.step,
            config=copy.deepcopy(self.config), seed=self.seed, model_sha256=self.model_identity,
            curriculum=copy.deepcopy(vars(self.args)), saturation=self.saturation,
            exogenous_draws_sha256=self.exogenous_draws_sha256, prior_draws_sha256=self.prior_draws_sha256,
            rng_streams=self.rng.state_dict(), global_rng=capture(self.rng.use_cuda),
            loss_sha256=sha256(ROOT / 'loss.py'), execution_label='research_fixed_budget', device_type=self.device.type,
            numerical_policy=self.numerical_policy)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError('checkpoint exists: ' + str(path))
        tmp = path.with_suffix('.tmp.pt')
        torch.save(self.snapshot(), tmp)
        tmp.replace(path)

    def resume(self, path):
        s = torch.load(path, map_location='cpu', weights_only=True)
        if (s['schema_version'] != 1 or s['upstream_commit'] != UPSTREAM or s['implementation_sha256'] != self.implementation_sha256 or s['config'] != self.config or s['seed'] != self.seed or
            s['loss_sha256'] != sha256(ROOT / 'loss.py') or s['model_sha256'] != self.model_identity or
            s['device_type'] != self.device.type or s.get('numerical_policy') != self.numerical_policy):
            raise ValueError('resume config/seed/loss/model/device/numerical policy mismatch')
        if not 0 <= s['step'] <= self.config['max_steps'] or s['delta'].shape != self.delta.shape:
            raise ValueError('invalid checkpoint step/shape')
        finite(s['delta'], 'resumed delta')
        if s['delta'].abs().max() > self.config['epsilon'] + 1e-7:
            raise ValueError('resumed delta exceeds epsilon')
        with torch.no_grad():
            self.delta.copy_(s['delta'].to(self.device))
        self.optimizer.load_state_dict(s['optimizer'])
        self.step, self.saturation = s['step'], s['saturation']
        self.exogenous_draws_sha256 = s['exogenous_draws_sha256']
        self.prior_draws_sha256 = s['prior_draws_sha256']
        self.args = SimpleNamespace(**s['curriculum'])
        self.rng.load_state_dict(s['rng_streams'])
        restore(s['global_rng'])

    def run(self, stop_step, directory):
        if not self.step < stop_step <= self.config['max_steps']:
            raise ValueError('stop step must advance within fixed budget')
        directory = Path(directory)
        with (directory / 'loss_trace.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            while self.step < stop_step:
                record = self.advance()
                writer.writerow(record)
                f.flush()
                if self.step % self.config['checkpoint_every'] == 0 or self.step == stop_step:
                    self.save(directory / 'checkpoints' / f'{self.step:06d}.pt')
        if model_hash(self.model) != self.model_identity:
            raise RuntimeError('frozen surrogate weights changed')
        if self.step == self.config['max_steps']:
            np.save(directory / 'uap.npy', self.delta.detach().cpu().numpy())
            (directory / 'uap.sha256').write_text(sha256(directory / 'uap.npy') + '  uap.npy\n')
