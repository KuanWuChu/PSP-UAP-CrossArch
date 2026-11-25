"""Aggregate only validated full holdout records; preserve missing/failed cells.

No maximum-FR checkpoint selection; duplicate variant/seed/target results are ambiguous.
"""
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from .config import validate
from .metrics import Counts
from .registry import TARGETS
from .provenance import digest, write_json

VARIANTS = ['e2_noise_crops', 'e2_psp_only', 'full_original', 'full_local', 'full_large']
SEEDS = [126, 2025, 3407]
PILOT_TARGETS = [x for x in TARGETS if x != 'resnet152']
METRICS = ['fr', 'clean_acc', 'adv_acc', 'cc_asr']
COUNTS = list(Counts.__dataclass_fields__)


def validate_metric(r):
    if r.get('status') != 'complete' or r.get('stage') != 'final' or r.get('purpose') != 'full':
        raise ValueError('requires complete full-budget final evaluation; smoke/missing/failed excluded')
    c = validate(r['training_config'])
    if r['variant'] != c['variant'] or type(r['seed']) is not int or r['seed'] not in SEEDS or r['step'] != c['max_steps']:
        raise ValueError('variant/seed/budget mismatch')
    if r['target'] not in TARGETS or r['total_count'] != 49000:
        raise ValueError('requires native registry target on complete 49000-image holdout')
    for k in COUNTS:
        if type(r[k]) is not int or not 0 <= r[k] <= r['total_count']:
            raise ValueError('invalid raw count: ' + k)
    n, cc, ac, wrong = r['total_count'], r['clean_correct_count'], r['adv_correct_count'], r['clean_correct_became_wrong_count']
    if wrong > min(cc, n-ac, r['flip_count']) or ac < cc-wrong:
        raise ValueError('inconsistent joint counts')
    expected = Counts(**{k:r[k] for k in COUNTS}).result()
    for k in METRICS:
        if expected[k] is None:
            if r[k] is not None:
                raise ValueError('zero denominator must be null')
        elif type(r[k]) not in (int,float) or not math.isfinite(r[k]) or not math.isclose(r[k],expected[k],abs_tol=1e-12,rel_tol=0):
            raise ValueError('metric disagrees with raw counts: ' + k)
    for k in ['weights_sha256','transform_sha256','manifest_sha256','uap_sha256','training_weights_sha256',
              'training_implementation_sha256','evaluation_implementation_sha256','exogenous_draws_sha256','prior_draws_sha256']:
        if not isinstance(r.get(k),str) or not re.fullmatch(r'[a-f0-9]{64}',r[k]):
            raise ValueError('missing/invalid provenance hash: ' + k)
    expected_metadata=dict(epsilon=c['epsilon'],source_mode=c['source_mode'],surrogate=c['surrogate'],
                           t_on=c['input_transform'],rw_on=c['re_weight'],crop_min=c['crop_area'][0],crop_max=c['crop_area'][1])
    if any(r.get(k)!=v for k,v in expected_metadata.items()):
        raise ValueError('UAP units/source/T/RW/crop metadata mismatch')
    return r


def comparison_identity(r):
    return {k:r[k] for k in ['protocol_id','target','weights_sha256','transform_sha256','manifest_sha256',
        'precision','environment','training_weights_sha256','training_implementation_sha256','evaluation_implementation_sha256']}


def paired_delta(base, intervention, contrast):
    if comparison_identity(base) != comparison_identity(intervention) or base['seed'] != intervention['seed']:
        raise ValueError('incomparable evaluation/source identities')
    a,b = dict(base['training_config']),dict(intervention['training_config'])
    if contrast == 'e2':
        if (a['variant'], b['variant']) != ('e2_noise_crops','e2_psp_only'):
            raise ValueError('E2 must compare B minus A')
        ignored=['variant','source_mode']
        draw_key='exogenous_draws_sha256'
    elif contrast == 'e3':
        if a['variant'] != 'full_original' or b['variant'] not in {'full_local','full_large'}:
            raise ValueError('E3 requires original vs prespecified crop policies')
        ignored=['variant','crop_area']
        draw_key='prior_draws_sha256'
    else:
        raise ValueError('unknown contrast')
    for k in ignored:
        a.pop(k); b.pop(k)
    if a != b or base[draw_key] != intervention[draw_key]:
        raise ValueError('unmatched config/budget/exogenous draws')
    return (intervention['fr'] - base['fr']) * 100


def summary(values):
    # Seed aggregation has exactly three prespecified seeds, no partial mean presented as final.
    if len(values) != len(SEEDS):
        return dict(status='missing_seeds', n=len(values), mean=None, sample_std=None)
    return dict(status='complete', n=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values))


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def aggregate(root, output, targets=None):
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise ValueError('runs root does not exist')
    targets = targets or PILOT_TARGETS
    if len(set(targets)) != len(targets) or any(t not in TARGETS for t in targets):
        raise ValueError('invalid targets')
    output.mkdir(parents=True,exist_ok=False)
    slots=defaultdict(list)
    rejected=[]
    for p in sorted(root.rglob('metrics.json')):
        try:
            r=validate_metric(json.loads(p.read_text()))
            if r['target'] in targets:
                slots[r['variant'],r['seed'],r['target']].append(r)
        except (ValueError,TypeError,KeyError) as e:
            rejected.append(dict(artifact=p.relative_to(root).as_posix(),reason=str(e)))
    def cell(v,s,t):
        records=slots[v,s,t]
        return records[0] if len(records)==1 else None
    seed_rows=[]
    for v in VARIANTS:
        for t in targets:
            for s in SEEDS:
                records=slots[v,s,t]
                status='complete' if len(records)==1 else 'ambiguous_duplicate' if records else 'missing'
                row=dict(variant=v,target=t,seed=s,status=status,**{k:None for k in ['run_id','training_run_id','uap_sha256','weights_sha256','manifest_sha256',*COUNTS,*METRICS]})
                if status=='complete':
                    row.update({k:records[0][k] for k in ['run_id','training_run_id','uap_sha256','weights_sha256','manifest_sha256',*COUNTS,*METRICS]})
                seed_rows.append(row)
    write_csv(output/'seed_metrics.csv',seed_rows)
    means=[]
    for v in VARIANTS:
        for t in targets:
            records=[cell(v,s,t) for s in SEEDS]
            present=[r for r in records if r]
            matched=len({digest(dict(identity=comparison_identity(r),config=r['training_config'])) for r in present})<=1
            for metric in METRICS:
                vals=[r[metric] for r in present if r[metric] is not None]
                stats=summary(vals) if matched else dict(status='incomparable_identities',n=len(vals),mean=None,sample_std=None)
                if matched and len(present)==3 and len(vals)!=3:
                    stats['status']='undefined_denominator'
                means.append(dict(variant=v,target=t,metric=metric,unit='ratio',**stats))
    write_csv(output/'seed_summary.csv',means)
    write_csv(output/'e1_matrix.csv',[r for r in means if r['variant']=='full_original'])
    deltas=[]
    for t in targets:
        for s in SEEDS:
            for intervention,kind,baseline in [('e2_psp_only','e2','e2_noise_crops'),('full_local','e3','full_original'),('full_large','e3','full_original')]:
                a,b=cell(baseline,s,t),cell(intervention,s,t)
                row=dict(contrast=kind,baseline=baseline,intervention=intervention,target=t,seed=s,
                         unit='percentage_points',delta_fr=None,status='missing_pair',error_reason=None,comparison_sha256=None)
                if a and b:
                    try:
                        row.update(delta_fr=paired_delta(a,b,kind),status='complete')
                        row['comparison_sha256']=digest(dict(identity=comparison_identity(a),
                            matched_config={k:v for k,v in a['training_config'].items() if k not in {'variant','source_mode','crop_area'}}))
                    except ValueError as e:
                        row.update(status='unmatched_pair',error_reason=str(e))
                deltas.append(row)
    write_csv(output/'paired_delta_fr.csv',deltas)
    delta_means=[]
    for t in targets:
        for v in ['e2_psp_only','full_local','full_large']:
            rr=[r for r in deltas if r['target']==t and r['intervention']==v and r['status']=='complete']
            matched=len({r['comparison_sha256'] for r in rr})<=1
            stats=summary([r['delta_fr'] for r in rr]) if matched else dict(status='incomparable_identities',n=len(rr),mean=None,sample_std=None)
            delta_means.append(dict(target=t,intervention=v,unit='percentage_points',**stats))
    write_csv(output/'paired_delta_summary.csv',delta_means)
    write_json(output/'rejected_records.json',rejected)
    counts={s:sum(r['status']==s for r in seed_rows) for s in ['complete','missing','ambiguous_duplicate']}
    result=dict(status='complete' if counts['complete']==len(seed_rows) and all(r['status']=='complete' for r in deltas) and all(r['status'] in {'complete','undefined_denominator'} for r in means) else 'incomplete',
        expected_training_runs=15,expected_target_records=len(seed_rows),record_counts=counts,
        rejected_records=len(rejected),note='Missing/failed/duplicate/unmatched results are never zero-filled. No max-FR selection.')
    write_json(output/'status.json',result)
    return result
