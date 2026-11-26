"""Crosscheck existing labels/manifests with torchvision's official devkit parser."""
import hashlib
import json
import re
import tempfile
from pathlib import Path
import importlib.metadata as md
from torchvision.datasets.imagenet import parse_devkit_archive, load_meta_file
from .data import verify_bundle, read_manifest
from .provenance import sha256, write_json, git_identity


def compare_mapping(wnids, truth, labels_dir, manifests_dir):
    """Require the official 1000-class order and all 50000 original validation IDs."""
    wnids = sorted(wnids)
    if len(wnids) != 1000 or len(set(wnids)) != 1000 or len(truth) != 50000:
        raise ValueError('official parser did not return 1000 classes / 50000 labels')
    labels_dir = Path(labels_dir)
    source = json.loads((labels_dir / 'source.json').read_text())
    for field, name in [('class_index_sha256', 'class_index.json'),
                        ('ground_truth_sha256', 'validation_wnids.txt')]:
        if sha256(labels_dir / name) != source[field]:
            raise ValueError('reference label digest mismatch')
    mapping = json.loads((labels_dir / 'class_index.json').read_text())
    if set(mapping) != {str(i) for i in range(1000)} or [mapping[str(i)][0] for i in range(1000)] != wnids:
        raise ValueError('class index differs from official sorted WNIDs')
    reference_truth = (labels_dir / 'validation_wnids.txt').read_text().splitlines()
    if reference_truth != list(truth):
        raise ValueError('reference validation labels differ from official devkit')
    index = verify_bundle(manifests_dir)
    if index['dataset'] != 'ILSVRC2012_validation' or index['counts'] != dict(selection=1000, holdout=49000, smoke_holdout=1000):
        raise ValueError('expected full validation manifest bundle')
    if any(index['source'][field] != source[field] for field in ['class_index_sha256', 'ground_truth_sha256']):
        raise ValueError('manifest source disagrees with reference label bundle')
    numbers = set()
    by_wnid = {w: i for i, w in enumerate(wnids)}
    for split in ['selection', 'holdout']:
        for row in read_manifest(Path(manifests_dir) / (split + '.csv')):
            match = re.fullmatch(r'n\d{8}/ILSVRC2012_val_(\d{8})\.JPEG', row['image_id'])
            if not match:
                raise ValueError('manifest lacks original validation filename')
            number = int(match[1])
            if number not in range(1, 50001) or number in numbers:
                raise ValueError('duplicate or invalid original validation number')
            numbers.add(number)
            expected = truth[number - 1]
            if row['class_wnid'] != expected or int(row['class_idx']) != by_wnid[expected]:
                raise ValueError('manifest class label differs from official devkit')
    if numbers != set(range(1, 50001)):
        raise ValueError('manifest must cover every original validation ID')
    return dict(status='PASS', classes=1000, matched_validation_labels=50000,
        matched_manifest_images=50000, class_order='sorted_WNID_torchvision_ImageFolder',
        reference_label_source=source, manifest_hashes=index['manifest_hashes'],
        manifest_counts=index['counts'], selection_holdout_disjoint=True,
        smoke_is_holdout_subset=True)


def verify_devkit_mapping(devkit, labels_dir, manifests_dir, output):
    devkit = Path(devkit).resolve()
    # torchvision checks the published MD5 before parsing or extracting the archive.
    # Its independent parser is used instead of reusing our label-builder implementation.
    with tempfile.TemporaryDirectory(prefix='psp-devkit-check-') as temporary:
        parse_devkit_archive(temporary, file=str(devkit))
        wnid_to_names, truth = load_meta_file(temporary)
    result = compare_mapping(wnid_to_names, truth, labels_dir, manifests_dir)
    checksum = hashlib.md5(devkit.read_bytes()).hexdigest()
    result.update(official_devkit_verified=True, devkit_md5=checksum,
        devkit_sha256=sha256(devkit), devkit_bytes=devkit.stat().st_size,
        independent_parser='torchvision.datasets.imagenet.parse_devkit_archive',
        torchvision_version=md.version('torchvision'),
        checksum_reference='https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/imagenet.html')
    result.update(git_identity())
    write_json(output, result)
    return result
