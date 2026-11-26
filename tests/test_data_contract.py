import io
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat
from psp_research.data import devkit_labels, make_manifests


def test_official_devkit_shape_mapping_and_groundtruth(tmp_path, monkeypatch):
    # Synthetic devkit-format fixture, not ImageNet images or experiment data.
    meta = np.empty((1000,), dtype=object)
    for i in range(1000):
        meta[i] = np.array([i + 1, f'n{i+1:08d}', 'name', 'gloss', 0], dtype=object)
    b = io.BytesIO()
    savemat(b, {'synsets': meta})
    tarpath = tmp_path / 'synthetic.tar.gz'
    with tarfile.open(tarpath, 'w:gz') as tar:
        for name, content in [('meta.mat', b.getvalue()),
            ('ILSVRC2012_validation_ground_truth.txt', b'1\n' * 50000)]:
            info = tarfile.TarInfo('ILSVRC2012_devkit_t12/data/' + name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    labels = tmp_path / 'labels'
    # Override only in this synthetic fixture; the real CLI requires the official MD5.
    monkeypatch.setattr('psp_research.data.OFFICIAL_DEVKIT_MD5', hashlib.md5(tarpath.read_bytes()).hexdigest())
    devkit_labels(tarpath, labels)
    mapping = json.loads((labels / 'class_index.json').read_text())
    assert mapping['0'][0] == 'n00000001' and mapping['999'][0] == 'n00001000'
    assert (labels / 'validation_wnids.txt').read_text().splitlines() == ['n00000001'] * 50000
    root = tmp_path / 'images'
    root.mkdir()
    for pair in mapping.values():
        (root / pair[0]).mkdir()
    # Filename #1 belongs to class 1, regardless of folder sorting.
    wrong = root / 'n00000002' / 'ILSVRC2012_val_00000001.JPEG'
    wrong.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='ground-truth'):
        make_manifests(root, labels, tmp_path / 'out')
    wrong.unlink()
    with pytest.raises(ValueError, match='50000'):
        make_manifests(root, labels, tmp_path / 'out')
    (labels / 'class_index.json').write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        make_manifests(root, labels, tmp_path / 'out')


def test_devkit_checksum_rejected_before_parsing(tmp_path):
    bad = tmp_path / 'bad.tar.gz'
    bad.write_bytes(b'not an official devkit')
    with pytest.raises(ValueError, match='MD5 mismatch'):
        devkit_labels(bad, tmp_path / 'labels')
    assert not (tmp_path / 'labels').exists()
