"""Synthetic metadata only; no ImageNet images or original devkit in fixtures."""
import csv
import json
import pytest
from psp_research.data import FIELDS, split_rows
from psp_research.devkit_verification import compare_mapping, verify_devkit_mapping
from psp_research.provenance import sha256, write_json


@pytest.fixture
def bundles(tmp_path):
    labels, manifests = tmp_path / 'labels', tmp_path / 'manifests'
    labels.mkdir(); manifests.mkdir()
    wnids = [f'n{i+1:08d}' for i in range(1000)]
    truth = [wnids[i % 1000] for i in range(50000)]
    write_json(labels / 'class_index.json', {str(i): [w, w] for i, w in enumerate(wnids)})
    (labels / 'validation_wnids.txt').write_text('\n'.join(truth) + '\n')
    source = dict(class_index_sha256=sha256(labels / 'class_index.json'),
                  ground_truth_sha256=sha256(labels / 'validation_wnids.txt'),
                  label_source='synthetic_fixture')
    write_json(labels / 'source.json', source)
    rows = [dict(image_id=f'{w}/ILSVRC2012_val_{i+1:08d}.JPEG', class_wnid=w,
                 class_idx=i % 1000, file_sha256='0' * 64) for i, w in enumerate(truth)]
    splits = split_rows(rows, 126)
    for split, selected in splits.items():
        with (manifests / (split + '.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(selected)
    write_json(manifests / 'index.json', dict(dataset='ILSVRC2012_validation', seed=126,
        source=source, counts={s: len(r) for s, r in splits.items()},
        manifest_hashes={s: sha256(manifests / (s + '.csv')) for s in splits}))
    return wnids, truth, labels, manifests


def test_all_original_ids_and_indices_match(bundles):
    result = compare_mapping(*bundles)
    assert result['matched_validation_labels'] == result['matched_manifest_images'] == 50000
    assert result['classes'] == 1000 and result['selection_holdout_disjoint']


@pytest.mark.parametrize('case,reason', [('class_order', 'class index'),
                                      ('truth', 'validation labels'),
                                      ('manifest_label', 'manifest class label')])
def test_self_consistent_reference_hashes_cannot_hide_wrong_mapping(bundles, case, reason):
    wnids, truth, labels, manifests = bundles
    index = json.loads((manifests / 'index.json').read_text())
    source = json.loads((labels / 'source.json').read_text())
    if case == 'class_order':
        mapping = json.loads((labels / 'class_index.json').read_text())
        mapping['0'], mapping['1'] = mapping['1'], mapping['0']
        write_json(labels / 'class_index.json', mapping)
        source['class_index_sha256'] = sha256(labels / 'class_index.json')
    elif case == 'truth':
        wrong = list(truth); wrong[0] = wnids[1]
        (labels / 'validation_wnids.txt').write_text('\n'.join(wrong) + '\n')
        source['ground_truth_sha256'] = sha256(labels / 'validation_wnids.txt')
    else:
        path = manifests / 'selection.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        rows[0]['class_idx'] = str((int(rows[0]['class_idx']) + 1) % 1000)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
        index['manifest_hashes']['selection'] = sha256(path)
    write_json(labels / 'source.json', source)
    index['source'] = source
    write_json(manifests / 'index.json', index)
    with pytest.raises(ValueError, match=reason):
        compare_mapping(wnids, truth, labels, manifests)


def test_bad_archive_rejected_by_independent_official_parser(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    archive.write_bytes(b'not the original devkit')
    with pytest.raises(RuntimeError, match='corrupted'):
        verify_devkit_mapping(archive, tmp_path / 'absent-labels', tmp_path / 'absent-manifests', tmp_path / 'report.json')
    assert not (tmp_path / 'report.json').exists()
