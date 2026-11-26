"""Official devkit label checks and immutable disjoint validation manifests."""
import csv
import hashlib
import io
import json
import random
import re
import tarfile
from pathlib import Path, PurePosixPath
from .provenance import sha256, write_json

FIELDS = ['image_id', 'class_wnid', 'class_idx', 'file_sha256', 'split']
SPLITS = ['selection', 'holdout', 'smoke_holdout']
OFFICIAL_DEVKIT_MD5 = 'fa75699e90414af021442c21a62c3abf'


def devkit_labels(devkit, output):
    from scipy.io import loadmat
    checksum = hashlib.md5()
    with open(devkit, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != OFFICIAL_DEVKIT_MD5:
        raise ValueError('official devkit MD5 mismatch; refusing mapping extraction')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(devkit) as tar:
        def member(suffix):
            found = [x for x in tar.getmembers() if x.isfile() and x.name.endswith(suffix)]
            if len(found) != 1:
                raise ValueError('devkit member absent or ambiguous: ' + suffix)
            return tar.extractfile(found[0]).read()
        meta = loadmat(io.BytesIO(member('/data/meta.mat')), squeeze_me=True)['synsets']
        pairs = [(int(s[0]), str(s[1])) for s in meta if int(s[4]) == 0]
        if len(pairs) != 1000 or len(set(w for _, w in pairs)) != 1000:
            raise ValueError('expected 1000 leaf synsets from official devkit')
        by_id = dict(pairs)
        wnids = sorted(by_id.values())
        write_json(output / 'class_index.json', {str(i): [w, w] for i, w in enumerate(wnids)})
        truth = member('/data/ILSVRC2012_validation_ground_truth.txt').decode().split()
        labels = [by_id[int(x)] for x in truth]
        if len(labels) != 50000:
            raise ValueError('expected 50000 validation labels')
        (output / 'validation_wnids.txt').write_text('\n'.join(labels) + '\n')
    write_json(output / 'source.json', {'devkit_sha256': sha256(devkit),
               'devkit_md5': checksum.hexdigest(), 'official_devkit_verified': True,
               'label_source': 'official_devkit',
               'class_index_sha256': sha256(output / 'class_index.json'),
               'ground_truth_sha256': sha256(output / 'validation_wnids.txt')})


def split_rows(rows, seed, selection_count=1000, smoke_count=1000):
    rows = sorted(rows, key=lambda r: r['image_id'])
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate image IDs')
    if selection_count < 1 or smoke_count < 1 or selection_count + smoke_count > len(rows):
        raise ValueError('invalid split sizes')
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    sel = set(order[:selection_count])
    smoke = set(order[selection_count:selection_count + smoke_count])
    def choose(indices, split):
        return [dict(r, split=split) for i, r in enumerate(rows) if i in indices]
    return dict(selection=choose(sel, 'selection'),
                holdout=choose(set(range(len(rows))) - sel, 'holdout'),
                smoke_holdout=choose(smoke, 'smoke_holdout'))


def make_manifests(root, labels_dir, out, seed=126):
    root, labels_dir, out = Path(root).resolve(), Path(labels_dir), Path(out)
    source = json.loads((labels_dir / 'source.json').read_text())
    for field, filename in [('class_index_sha256', 'class_index.json'),
                             ('ground_truth_sha256', 'validation_wnids.txt')]:
        if source[field] != sha256(labels_dir / filename):
            raise ValueError('devkit label digest mismatch')
    mapping = json.loads((labels_dir / 'class_index.json').read_text())
    if set(mapping) != {str(i) for i in range(1000)}:
        raise ValueError('need all 1000 class indices')
    wnids = [mapping[str(i)][0] for i in range(1000)]
    if wnids != sorted(set(wnids)) or any(not re.fullmatch(r'n\d{8}', w) for w in wnids):
        raise ValueError('class mapping must match torchvision alphabetical WNID order')
    truth = (labels_dir / 'validation_wnids.txt').read_text().split()
    if len(truth) != 50000 or not set(truth) <= set(wnids):
        raise ValueError('invalid validation ground truth')
    if sorted(p.name for p in root.iterdir() if p.is_dir()) != wnids:
        raise ValueError('ImageFolder class directories disagree with devkit')
    rows, numbers = [], set()
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if len(rel.parts) != 2 or path.suffix.lower() not in {'.jpeg', '.jpg', '.png'}:
            raise ValueError('unexpected file in dataset: ' + str(rel))
        m = re.fullmatch(r'ILSVRC2012_val_(\d{8})\.JPEG', path.name)
        if not m:
            raise ValueError('expected original ImageNet validation filenames')
        n = int(m[1])
        if n < 1 or n > 50000 or n in numbers or rel.parts[0] != truth[n - 1]:
            raise ValueError('filename, duplicate, or ground-truth mapping mismatch: ' + str(rel))
        if not path.resolve().is_relative_to(root):
            raise ValueError('dataset symlink escapes root')
        numbers.add(n)
        rows.append(dict(image_id=rel.as_posix(), class_wnid=rel.parts[0],
                         class_idx=wnids.index(rel.parts[0]), file_sha256=sha256(path)))
    if numbers != set(range(1, 50001)):
        raise ValueError('expected exactly 50000 validation images')
    splits = split_rows(rows, seed)
    out.mkdir(parents=True, exist_ok=False)
    for split, values in splits.items():
        with (out / (split + '.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(values)
    write_json(out / 'index.json', dict(schema_version=1, seed=seed, source=source,
               dataset='ILSVRC2012_validation', manifest_hashes={s: sha256(out / (s + '.csv')) for s in SPLITS},
               counts={s: len(v) for s, v in splits.items()}))
    verify_bundle(out)


def read_manifest(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('manifest columns mismatch')
        rows = list(reader)
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('empty manifest or duplicate IDs')
    for r in rows:
        p = PurePosixPath(r['image_id'])
        if p.is_absolute() or '..' in p.parts or '\\' in r['image_id'] or len(p.parts) != 2:
            raise ValueError('unsafe relative image ID')
        if p.parts[0] != r['class_wnid'] or not 0 <= int(r['class_idx']) < 1000:
            raise ValueError('invalid manifest label')
        if r['split'] not in SPLITS or not re.fullmatch('[a-f0-9]{64}', r['file_sha256']):
            raise ValueError('invalid split or image checksum')
    return rows


def verify_bundle(directory):
    directory = Path(directory)
    index = json.loads((directory / 'index.json').read_text())
    rows = {}
    for s in SPLITS:
        path = directory / (s + '.csv')
        if sha256(path) != index['manifest_hashes'][s]:
            raise ValueError('manifest checksum mismatch: ' + s)
        rows[s] = read_manifest(path)
        if any(r['split'] != s for r in rows[s]) or len(rows[s]) != index['counts'][s]:
            raise ValueError('manifest split/count mismatch')
    ids = {s: {r['image_id'] for r in rr} for s, rr in rows.items()}
    if ids['selection'] & ids['holdout'] or not ids['smoke_holdout'] <= ids['holdout']:
        raise ValueError('split overlap or invalid smoke subset')
    by_id = {r['image_id']: r for r in rows['holdout']}
    if any(any(r[k] != by_id[r['image_id']][k] for k in FIELDS if k != 'split') for r in rows['smoke_holdout']):
        raise ValueError('smoke labels/content disagree with holdout')
    return index
