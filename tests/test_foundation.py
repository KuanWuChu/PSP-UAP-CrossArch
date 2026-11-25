import copy
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from psp_research.config import DEFAULTS, validate
from psp_research.metrics import Counts, normalize, perturb, load_uap
from psp_research.data import split_rows, read_manifest, verify_bundle, FIELDS
from psp_research.provenance import sha256, write_json, ROOT
from psp_research.evaluator import evaluate_batches, ManifestDataset
from psp_research.registry import TARGETS, specification


def test_analytic_counts():
    c = Counts()
    c.update(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 0]), torch.tensor([1, 1, 2, 1]))
    r = c.result()
    assert r['total_count'] == 4 and r['flip_count'] == 3
    assert r['clean_correct_count'] == 2 and r['adv_correct_count'] == 2
    assert r['clean_correct_became_wrong_count'] == 1
    assert (r['fr'], r['clean_acc'], r['adv_acc'], r['cc_asr']) == (0.75, 0.5, 0.5, 0.5)
    c = Counts()
    c.update(torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
    assert c.result()['cc_asr'] is None
    with pytest.raises(ValueError):
        Counts().result()


def test_rgb_clamp_then_normalize():
    x = torch.tensor([0.99, 0.01, 0.5]).reshape(1, 3, 1, 1)
    d = torch.tensor([0.03, -0.03, 0.03]).reshape_as(x)
    y = normalize(perturb(x, d), [0.5, 0.25, 0.1], [0.2, 0.5, 1.0])
    torch.testing.assert_close(y.flatten(), torch.tensor([2.5, -0.5, 0.43]))
    assert (perturb(x, d) - x).abs().max() <= d.abs().max()
    with pytest.raises(ValueError):
        perturb(x * 3, d)


def test_uap_contract(tmp_path):
    p = tmp_path / 'uap.npy'
    d = np.zeros((1, 3, 224, 224), dtype=np.float32)
    np.save(p, d)
    assert load_uap(p).count_nonzero() == 0
    for invalid in [d.astype(np.float64), d[0], d + 0.04, d + np.nan, d + np.inf]:
        np.save(p, invalid)
        with pytest.raises(ValueError):
            load_uap(p)


def test_zero_delta_identity_no_grad():
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    images = torch.rand(5, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 0])
    before = copy.deepcopy(model.state_dict())
    result = evaluate_batches(model, [(images, labels, ['a', 'b', 'c', 'd', 'e'])], torch.zeros(1, 3, 8, 8), [0]*3, [1]*3)
    assert result['fr'] == 0 and result['clean_acc'] == result['adv_acc']
    assert all(p.grad is None for p in model.parameters())
    for k, v in before.items():
        assert torch.equal(v, model.state_dict()[k])
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_batches(model, [(images, labels, ['a']*5)], torch.zeros(1, 3, 8, 8), [0]*3, [1]*3)


def test_nonfinite_logits_rejected():
    class Bad(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 2), float('nan'))
    with pytest.raises(FloatingPointError):
        evaluate_batches(Bad(), [(torch.zeros(1, 3, 4, 4), torch.tensor([0]), ['a'])], torch.zeros(1, 3, 4, 4), [0]*3, [1]*3)


def test_splits_deterministic_and_disjoint(tmp_path):
    rows = [dict(image_id=f'n00000001/{i}.JPEG', class_wnid='n00000001', class_idx=0, file_sha256='0'*64) for i in range(30)]
    splits = split_rows(rows, 126, 3, 5)
    assert splits == split_rows(list(reversed(rows)), 126, 3, 5)
    assert splits != split_rows(rows, 2025, 3, 5)
    hashes = {}
    for s, rr in splits.items():
        p = tmp_path / (s + '.csv')
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rr)
        hashes[s] = sha256(p)
    index = dict(manifest_hashes=hashes, counts={s: len(v) for s, v in splits.items()})
    write_json(tmp_path / 'index.json', index)
    verify_bundle(tmp_path)
    # Correctly rehashed but overlapping manifest must still be rejected.
    p = tmp_path / 'selection.csv'
    p.write_text(p.read_text().replace(splits['selection'][0]['image_id'], splits['holdout'][0]['image_id']))
    index['manifest_hashes']['selection'] = sha256(p)
    write_json(tmp_path / 'index.json', index)
    with pytest.raises(ValueError, match='overlap'):
        verify_bundle(tmp_path)
    with pytest.raises(ValueError):
        split_rows(rows + rows[:1], 126, 3, 5)


def test_data_content_and_label(tmp_path):
    from PIL import Image
    p = tmp_path / 'n00000001' / 'x.JPEG'
    p.parent.mkdir()
    Image.new('RGB', (8, 8)).save(p)
    row = dict(image_id='n00000001/x.JPEG', class_idx=42, file_sha256=sha256(p))
    ds = ManifestDataset(tmp_path, [row], lambda im: im.size)
    assert ds[0][1:] == (42, row['image_id'])
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        ds[0]


@pytest.mark.parametrize('change', [dict(max_steps=0), dict(input_transform='false'), dict(re_weight=True),
    dict(prior='invented'), dict(epsilon=float('nan')), dict(epsilon=1.5), dict(crop_area=[0, 1]),
    dict(unknown=1), dict(seeds=[1]), dict(source_mode='typo'), dict(variant='full_local')])
def test_strict_config_rejections(change):
    with pytest.raises(ValueError):
        validate(dict(DEFAULTS, **change))


def test_registry_native_224_explicit():
    for name in TARGETS:
        s = specification(name)
        assert s['crop_size'] == [224] and 'DEFAULT' not in s['weights_id']
        assert s['url'].startswith('https://download.pytorch.org/models/')
        assert len(s['transform_sha256']) == 64


def test_original_source_hash():
    subprocess.run(['python3', 'scripts/check_upstream_integrity.py'], cwd=ROOT, check=True)
