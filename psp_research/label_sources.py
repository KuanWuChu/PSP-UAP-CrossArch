"""Audit official Kaggle labels against an independent pinned TFDS reference.

This route never claims that the original ImageNet devkit archive was obtained.
"""
import csv
import json
import re
from pathlib import Path
from .provenance import sha256, write_json

COMPETITION_URL = 'https://www.kaggle.com/competitions/imagenet-object-localization-challenge'


def kaggle_labels(csv_file, reference_labels, reference_truth, reference_commit, output):
    if not re.fullmatch(r'[a-f0-9]{40}', reference_commit):
        raise ValueError('require a pinned tensorflow/datasets commit SHA')
    classes = Path(reference_labels).read_text().splitlines()
    truth = Path(reference_truth).read_text().splitlines()
    class_set = set(classes)
    if (len(classes) != 1000 or classes != sorted(set(classes)) or
        any(not re.fullmatch(r'n\d{8}', c) for c in classes)):
        raise ValueError('TFDS reference must contain 1000 sorted unique WNIDs')
    if len(truth) != 50000 or set(truth) != set(classes):
        raise ValueError('TFDS reference requires 50000 labels and all 1000 classes')
    by_id = {}
    with Path(csv_file).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ImageId', 'PredictionString']:
            raise ValueError('unexpected official Kaggle label columns')
        for row in reader:
            m = re.fullmatch(r'ILSVRC2012_val_(\d{8})', row['ImageId'])
            if not m or not 1 <= int(m[1]) <= 50000 or row['ImageId'] in by_id:
                raise ValueError('invalid, out-of-range or duplicate validation ID')
            fields = row['PredictionString'].split()
            if not fields or len(fields) % 5:
                raise ValueError('malformed localization annotation')
            labels = set(fields[::5])
            if len(labels) != 1 or not labels <= class_set:
                raise ValueError('ambiguous or unknown classification label')
            label = next(iter(labels))
            if truth[int(m[1]) - 1] != label:
                raise ValueError('Kaggle label disagrees with pinned TFDS ground truth')
            by_id[row['ImageId']] = label
    if len(by_id) != 50000:
        raise ValueError('expected all 50000 unique validation IDs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'class_index.json', {str(i): [wnid, wnid] for i, wnid in enumerate(classes)})
    (output / 'validation_wnids.txt').write_text('\n'.join(truth) + '\n')
    prefix = f'https://raw.githubusercontent.com/tensorflow/datasets/{reference_commit}/tensorflow_datasets/datasets/imagenet2012/'
    source = dict(label_source='official_kaggle_crosschecked_tfds',
        official_devkit_verified=False, devkit_sha256=None,
        kaggle=dict(competition_url=COMPETITION_URL, file_name='LOC_val_solution.csv', file_sha256=sha256(csv_file)),
        independent_reference=dict(repository='tensorflow/datasets', commit=reference_commit,
            labels_url=prefix+'labels.txt', labels_sha256=sha256(reference_labels),
            validation_labels_url=prefix+'validation_labels.txt', validation_labels_sha256=sha256(reference_truth)),
        matched_validation_labels=50000, class_order='sorted_WNID_torchvision_ImageFolder',
        class_index_sha256=sha256(output / 'class_index.json'),
        ground_truth_sha256=sha256(output / 'validation_wnids.txt'))
    write_json(output / 'source.json', source)
    return source
