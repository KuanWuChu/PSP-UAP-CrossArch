import csv
import json
import time
import uuid
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data import read_manifest, verify_bundle
from .metrics import Counts, perturb, normalize, load_uap
from .provenance import sha256, write_json, git_identity, digest, check_upstream
from .registry import load_target


class ManifestDataset(Dataset):
    def __init__(self, root, rows, geometry):
        self.root, self.rows, self.geometry = Path(root).resolve(), rows, geometry

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        path = (self.root / r['image_id']).resolve()
        if not path.is_relative_to(self.root) or sha256(path) != r['file_sha256']:
            raise ValueError('image content changed or unsafe path: ' + r['image_id'])
        with Image.open(path) as im:
            image = self.geometry(im.convert('RGB'))
        return image, int(r['class_idx']), r['image_id']


@torch.inference_mode()
def evaluate_batches(model, loader, delta, mean, std, writer=None):
    model.eval().requires_grad_(False)
    counts, seen = Counts(), set()
    for images, labels, ids in loader:
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('duplicate evaluated image IDs')
        seen.update(ids)
        images, labels = images.to(delta.device), labels.to(delta.device)
        adv = perturb(images, delta)
        clean_logits = model(normalize(images, mean, std))
        adv_logits = model(normalize(adv, mean, std))
        if not torch.isfinite(clean_logits).all() or not torch.isfinite(adv_logits).all():
            raise FloatingPointError('non-finite model output')
        clean, attacked = clean_logits.argmax(1), adv_logits.argmax(1)
        counts.update(labels, clean, attacked)
        if writer:
            writer.writerows(zip(ids, labels.cpu().tolist(), clean.cpu().tolist(), attacked.cpu().tolist()))
    return counts.result()


def evaluate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run_id = 'eval-' + uuid.uuid4().hex[:12]
    provenance = dict(git_identity(), run_id=run_id, status='running', error_reason=None,
                      execution_label='target_only_evaluation', precision='float32',
                      environment={'torch': torch.__version__, 'cuda': torch.version.cuda})
    write_json(out / 'provenance.json', provenance)
    try:
        check_upstream()
        if not 0 < args.epsilon <= 1 or args.batch_size < 1 or args.workers < 0:
            raise ValueError('invalid epsilon/batch size/workers')
        index = verify_bundle(Path(args.manifest).parent)
        if index.get('dataset') != 'ILSVRC2012_validation' or index.get('counts') != {'selection': 1000, 'holdout': 49000, 'smoke_holdout': 1000}:
            raise ValueError('expected verified full ImageNet validation split bundle')
        split = Path(args.manifest).stem
        if split not in {'holdout', 'smoke_holdout', 'selection'}:
            raise ValueError('unknown manifest split')
        if args.stage == 'final' and split != 'holdout':
            raise ValueError('final evaluation requires disjoint holdout')
        if args.stage == 'smoke' and split != 'smoke_holdout':
            raise ValueError('smoke evaluation requires fixed smoke_holdout')
        if args.stage == 'selection' and split != 'selection':
            raise ValueError('selection diagnostics require selection manifest')
        if len(set(args.targets)) != len(args.targets):
            raise ValueError('duplicate target names')
        delta = load_uap(args.uap, args.epsilon).to(args.device)
        rows = read_manifest(args.manifest)
        provenance.update(uap_sha256=sha256(args.uap), manifest_sha256=sha256(args.manifest),
                          split=split, stage=args.stage, epsilon=args.epsilon,
                          weights_lock_sha256=sha256(args.weights_lock), data_source=index['source'])
        protocol = dict(split=split, manifest_sha256=provenance['manifest_sha256'],
                        epsilon=args.epsilon, geometry='checkpoint_native_224', precision='float32')
        provenance['protocol_id'] = 'e1-' + digest(protocol)[:16]
        write_json(out / 'config.resolved.json', dict(protocol, targets=args.targets,
                   batch_size=args.batch_size, workers=args.workers, stage=args.stage))
        for target in args.targets:
            tdir = out / target
            tdir.mkdir()
            record = dict(provenance, target=target, status='running')
            try:
                model, geometry, spec = load_target(target, args.weights_lock, args.weights_dir, args.device)
                dataset = ManifestDataset(args.data_root, rows, geometry)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with (tdir / 'predictions.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_id', 'label', 'clean_prediction', 'adv_prediction'])
                    metrics = evaluate_batches(model, loader, delta, spec['mean'], spec['std'], writer)
                if str(args.device).startswith('cuda'):
                    torch.cuda.synchronize()
                record.update(metrics, target_family=spec['family'], weights_id=spec['weights_id'],
                              weights_sha256=spec['weights_sha256'], transform_sha256=spec['transform_sha256'],
                              elapsed_seconds=time.perf_counter() - start, status='complete')
                if metrics['total_count'] != len(rows):
                    raise ValueError('evaluated count mismatch')
                write_json(tdir / 'metrics.json', record)
                del model
            except Exception as e:
                write_json(tdir / 'metrics.json', dict(record, status='failed', error_reason=type(e).__name__ + ': ' + str(e)))
                raise
        provenance['status'] = 'complete'
    except Exception as e:
        provenance.update(status='failed', error_reason=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        write_json(out / 'provenance.json', provenance)
