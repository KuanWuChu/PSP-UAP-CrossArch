import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
from psp_research.config import DEFAULTS
from psp_research.trainer import ControlledTrainer, build_sources, model_hash, paired_transforms
from psp_research.rng import RNGStreams
import loss as original_loss


def fixture_model():
    with torch.random.fork_rng():
        torch.manual_seed(912)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(), nn.Linear(256, 5))
        with torch.no_grad():
            model[-1].weight.mul_(20)
        return model


def fixture_config(**kw):
    return dict(DEFAULTS, max_steps=4, num_copies_per_prior=2, checkpoint_every=2, prior='gauss', lr=0.001, **kw)


def same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('mode', ['noise_crops', 'psp_crops'])
def test_detach_path_exact_delta_gradient(mode):
    delta = torch.full((1, 3, 224, 224), .02, requires_grad=True)
    prior = torch.rand(2, 3, 224, 224, requires_grad=True)
    with RNGStreams(126).use('crop'):
        images, source, boxes = build_sources(prior, delta, mode, 2, [.08, 1], [.75, 4/3])
    assert not source.requires_grad and len(boxes) == 4
    images.sum().backward()
    assert prior.grad is None
    torch.testing.assert_close(delta.grad, torch.full_like(delta, 4), rtol=0, atol=0)


def test_resume_bitwise_and_checkpoint_is_snapshot(tmp_path):
    cfg = fixture_config()
    full = ControlledTrainer(fixture_model(), cfg, 126)
    full_records = [full.advance() for _ in range(4)]
    segmented = ControlledTrainer(fixture_model(), cfg, 126)
    first = [segmented.advance() for _ in range(2)]
    snap = segmented.snapshot()
    checkpoint = tmp_path / 'step2.pt'
    segmented.save(checkpoint)
    frozen = snap['delta'].clone()
    segmented.advance()
    assert torch.equal(frozen, snap['delta']) and not torch.equal(frozen, segmented.delta)
    resumed = ControlledTrainer(fixture_model(), cfg, 126)
    resumed.resume(checkpoint)
    last = [resumed.advance() for _ in range(2)]
    assert full_records == first + last
    same(full.snapshot(), resumed.snapshot())
    assert all(p.grad is None and not p.requires_grad for p in resumed.model.parameters())
    assert model_hash(resumed.model) == resumed.model_identity


def test_paired_exogenous_draws_despite_transform_stream_consumption():
    a = ControlledTrainer(fixture_model(), fixture_config(), 126)
    b = ControlledTrainer(fixture_model(), fixture_config(variant='e2_psp_only', source_mode='psp_crops'), 126)
    for _ in range(3):
        with b.rng.use('transform'):
            torch.rand(100)
            np.random.rand(50)
            random.random()
        ra, rb = a.advance(), b.advance()
        assert ra['prior_sha256'] == rb['prior_sha256']
        assert ra['crop_boxes_sha256'] == rb['crop_boxes_sha256']
        assert ra['samples'] == rb['samples'] == 2
    assert not torch.equal(a.delta, b.delta)


def test_global_rng_isolation():
    streams = RNGStreams(126)
    py, npstate, t = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with streams.use('prior'):
        random.random(); np.random.normal(); torch.rand(10)
    assert py == random.getstate()
    assert np.array_equal(npstate[1], np.random.get_state()[1])
    assert torch.equal(t, torch.get_rng_state())


def test_nonfinite_original_loss_fails_without_repair(monkeypatch):
    runner = ControlledTrainer(fixture_model(), fixture_config(), 126)
    monkeypatch.setattr(original_loss, 'l2_layer_loss', lambda model, x, args, dev: x.sum() * float('nan'))
    with pytest.raises(FloatingPointError, match='original activation loss'):
        runner.advance()
    assert runner.step == 0


def test_jigsaw_curriculum_resume_boundary(tmp_path):
    c = dict(DEFAULTS, max_steps=2000, num_copies_per_prior=1, checkpoint_every=1, lr=.001)
    a = ControlledTrainer(fixture_model(), c, 126)
    a.step = 1599
    a.args.fre = 5
    a.save(tmp_path / 'boundary.pt')
    b = ControlledTrainer(fixture_model(), c, 126)
    b.resume(tmp_path / 'boundary.pt')
    assert a.advance() == b.advance()
    assert a.args.prior_batch == b.args.prior_batch == 2 and a.args.fre == 6
    same(a.snapshot(), b.snapshot())


def test_run_partial_then_resume_exports_only_final_uap(tmp_path):
    a = ControlledTrainer(fixture_model(), fixture_config(), 126)
    part, final = tmp_path / 'part', tmp_path / 'final'
    part.mkdir(); final.mkdir()
    a.run(2, part)
    assert not (part / 'uap.npy').exists()
    b = ControlledTrainer(fixture_model(), fixture_config(), 126)
    b.resume(part / 'checkpoints/000002.pt')
    b.run(4, final)
    d = np.load(final / 'uap.npy')
    assert d.dtype == np.float32 and np.abs(d).max() <= np.float32(DEFAULTS['epsilon'])
    with pytest.raises(ValueError, match='budget'):
        b.advance()
    wrong = ControlledTrainer(fixture_model(), fixture_config(), 2025)
    with pytest.raises(ValueError, match='mismatch'):
        wrong.resume(part / 'checkpoints/000002.pt')


def test_full_original_uses_frozen_weighted_loss():
    c = fixture_config(variant='full_original', source_mode='psp_crops', input_transform=True, re_weight=True)
    runner = ControlledTrainer(fixture_model(), c, 126)
    r = runner.advance()
    assert np.isfinite(r['loss']) and r['samples'] == 2
    assert all(p.grad is None for p in runner.model.parameters())


def test_resume_rejects_changed_numerical_policy(tmp_path):
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(True)
        a = ControlledTrainer(fixture_model(), fixture_config(), 126)
        a.advance()
        a.save(tmp_path / 'strict.pt')
        torch.use_deterministic_algorithms(True, warn_only=True)
        b = ControlledTrainer(fixture_model(), fixture_config(), 126)
        with pytest.raises(ValueError, match='numerical policy mismatch'):
            b.resume(tmp_path / 'strict.pt')
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_original_first_teacher_updates_batchnorm_but_controlled_model_stays_frozen():
    # Original prepare_for_model leaves training=True; weighted teacher runs before eval().
    with torch.random.fork_rng():
        torch.manual_seed(912)
        model = nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),nn.Flatten(),nn.Linear(256,5))
        with torch.no_grad():model[-1].weight.mul_(20)
        fresh = copy.deepcopy(model)
        cfg = fixture_config(variant='full_original',source_mode='psp_crops',input_transform=True,re_weight=True)
        runner = ControlledTrainer(fresh,cfg,126)
        source = torch.rand(2,3,224,224) + 1
        images = source + .02
        before = model[1].running_mean.clone()
        original_loss.l2_layer_loss_weight(model,images,source,runner.args,2,'cpu')
        assert not torch.equal(before,model[1].running_mean)
        assert not model.training and model[1].num_batches_tracked.item()==1
        original_loss.l2_layer_loss_weight(model,images,source,runner.args,2,'cpu')
        assert model[1].num_batches_tracked.item()==1
        runner.advance()
        assert runner.model[1].num_batches_tracked.item()==0
        assert model_hash(runner.model)==runner.model_identity
