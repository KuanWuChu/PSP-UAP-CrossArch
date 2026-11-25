import copy
import csv
import json
import pytest
from psp_research.aggregate import validate_metric,paired_delta,aggregate,VARIANTS,SEEDS
from psp_research.config import DEFAULTS
from psp_research.metrics import Counts
from psp_research.provenance import write_json


def record(variant='e2_noise_crops',seed=126,flips=9800):
    full=variant.startswith('full_')
    area={'full_local':[.08,.35],'full_large':[.5,1]}.get(variant,[.08,1])
    c=dict(DEFAULTS,variant=variant,source_mode='noise_crops' if variant=='e2_noise_crops' else 'psp_crops',
           input_transform=full,re_weight=full,crop_area=area)
    return dict(status='complete',stage='final',purpose='full',training_config=c,variant=variant,seed=seed,step=10000,
        target='resnet50',precision='float32',environment={'fixture':True},protocol_id='fixture',
        training_run_id=f'fixture-{variant}-{seed}',run_id=f'eval-fixture-{variant}-{seed}',epsilon=10/255,source_mode=c['source_mode'],
        surrogate=c['surrogate'],t_on=c['input_transform'],rw_on=c['re_weight'],crop_min=area[0],crop_max=area[1],
        **Counts(49000,flips,24500,20000,5000).result(),
        **{k:'a'*64 for k in ['weights_sha256','transform_sha256','manifest_sha256','uap_sha256','training_weights_sha256',
            'training_implementation_sha256','evaluation_implementation_sha256','exogenous_draws_sha256','prior_draws_sha256']})


def test_exact_paired_delta_and_reject_unmatched():
    a,b=record(),record('e2_psp_only',flips=14700)
    assert paired_delta(a,b,'e2')==pytest.approx(10)
    b['weights_sha256']='b'*64
    with pytest.raises(ValueError,match='identities'):
        paired_delta(a,b,'e2')
    b=record('e2_psp_only',flips=14700); b['exogenous_draws_sha256']='b'*64
    with pytest.raises(ValueError,match='draws'):
        paired_delta(a,b,'e2')
    b=record('e2_psp_only',flips=14700); b['training_config']['max_steps']=5000
    with pytest.raises(ValueError,match='config'):
        paired_delta(a,b,'e2')
    assert paired_delta(record('full_original'),record('full_local',flips=14700),'e3')==pytest.approx(10)


@pytest.mark.parametrize('change',[dict(stage='smoke'),dict(purpose='smoke'),dict(status='failed'),
    dict(total_count=1000),dict(fr=.9),dict(clean_correct_count=-1),dict(uap_sha256=None),dict(t_on=True)])
def test_invalid_records_rejected(change):
    with pytest.raises(ValueError):
        validate_metric(dict(record(),**change))


def test_missing_is_blank_never_zero(tmp_path):
    root=tmp_path/'runs'; root.mkdir()
    out=tmp_path/'out'
    result=aggregate(root,out,['resnet50'])
    assert result['record_counts']==dict(complete=0,missing=15,ambiguous_duplicate=0)
    with (out/'seed_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==15 and all(r['fr']=='' and r['status']=='missing' for r in rows)
    with (out/'paired_delta_summary.csv').open() as f:
        assert all(r['mean']=='' for r in csv.DictReader(f))


def test_complete_summary_and_no_max_fr_duplicate_selection(tmp_path):
    root=tmp_path/'runs'; root.mkdir()
    for v in VARIANTS:
        for i,s in enumerate(SEEDS):
            flips=9800+i*490
            if v=='e2_psp_only':
                flips+=2450+i*490
            p=root/f'{v}-{s}'/'metrics.json'
            write_json(p,record(v,s,flips))
    out=tmp_path/'out'
    assert aggregate(root,out,['resnet50'])['status']=='complete'
    with (out/'paired_delta_summary.csv').open() as f:
        rows=list(csv.DictReader(f))
    e2=next(r for r in rows if r['intervention']=='e2_psp_only')
    assert float(e2['mean'])==pytest.approx(6) and float(e2['sample_std'])==pytest.approx(1)
    write_json(root/'duplicate'/'metrics.json',record('e2_noise_crops',126,20000))
    result=aggregate(root,tmp_path/'duplicates',['resnet50'])
    assert result['status']=='incomplete' and result['record_counts']['ambiguous_duplicate']==1
