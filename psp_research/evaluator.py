import csv
import json
import time
import uuid
import importlib.metadata as md
import platform
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data import read_manifest, verify_bundle
from .metrics import Counts, perturb, normalize, load_uap
from .provenance import sha256, write_json, git_identity, digest, check_upstream, ROOT
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
    provenance = dict(git_identity(), evaluation_implementation_sha256=digest({p:sha256(ROOT / p) for p in ['psp_research/evaluator.py','psp_research/data.py','psp_research/metrics.py','psp_research/registry.py']}), run_id=run_id, status='running', error_reason=None,
                      execution_label='target_only_evaluation', precision='float32',
                      environment={'torch': torch.__version__, 'torchvision': md.version('torchvision'), 'pillow': md.version('pillow'), 'python': platform.python_version(), 'cuda': torch.version.cuda, 'tf32': False})
    write_json(out / 'provenance.json', provenance)
    try:
        check_upstream()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
                        epsilon=args.epsilon, geometry='checkpoint_native_224', precision='float32', environment=provenance['environment'])
        provenance['protocol_id'] = 'e1-' + digest(protocol)[:16]
        provenance.update(training_run_id=None, variant=None, source_mode=None, surrogate=None,
                          seed=None, step=None, t_on=None, rw_on=None, crop_min=None, crop_max=None)
        training_record = getattr(args, 'training_record', None)
        if training_record:
            from .config import validate
            training = json.loads(Path(training_record).read_text())
            config = validate(training['config'])
            if (training['status'] != 'complete' or training.get('uap_sha256') != provenance['uap_sha256'] or
                training.get('manifest_hashes') != index['manifest_hashes'] or config['epsilon'] != args.epsilon):
                raise ValueError('training record/UAP/manifest/epsilon mismatch')
            if args.stage == 'final' and (training['purpose'] != 'full' or training['step'] != config['max_steps']):
                raise ValueError('smoke or incomplete training cannot become a formal final evaluation')
            provenance.update(training_run_id=training['run_id'], variant=config['variant'],
                source_mode=config['source_mode'], surrogate=config['surrogate'], seed=training['seed'],
                step=training['step'], t_on=config['input_transform'], rw_on=config['re_weight'],
                crop_min=config['crop_area'][0], crop_max=config['crop_area'][1],
                training_config=config, training_weights_sha256=training['weights_sha256'],
                training_record_sha256=sha256(training_record), purpose=training['purpose'],
                exogenous_draws_sha256=training.get('exogenous_draws_sha256'), prior_draws_sha256=training.get('prior_draws_sha256'),
                training_implementation_sha256=training.get('training_implementation_sha256'))
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
