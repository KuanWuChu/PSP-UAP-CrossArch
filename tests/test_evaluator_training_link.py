from argparse import Namespace
import csv
import json
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from psp_research import evaluator
from psp_research.config import DEFAULTS
from psp_research.data import FIELDS
from psp_research.provenance import sha256, write_json


@pytest.mark.parametrize('purpose,expected', [('full','complete'), ('smoke','failed')])
def test_training_record_link_and_smoke_final_guard(tmp_path,monkeypatch,purpose,expected):
    # Small synthetic fixture; fake bundle verification isolates orchestration logic.
    data = tmp_path / 'data'
    (data/'n00000001').mkdir(parents=True)
    img = data/'n00000001/fixture.JPEG'
    Image.new('RGB',(224,224),(125,150,175)).save(img)
    manifest = tmp_path/'holdout.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        w.writerow(dict(image_id='n00000001/fixture.JPEG',class_wnid='n00000001',class_idx=0,
                        file_sha256=sha256(img),split='holdout'))
    index=dict(dataset='ILSVRC2012_validation',counts={'selection':1000,'holdout':49000,'smoke_holdout':1000},
               source={},manifest_hashes={'holdout':sha256(manifest),'selection':'1'*64,'smoke_holdout':'2'*64})
    monkeypatch.setattr(evaluator,'verify_bundle',lambda _:index)
    from torchvision.transforms import ToTensor
    def target(*args):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten()),ToTensor(),dict(
            mean=[0]*3,std=[1]*3,family='fixture',weights_id='fixture',weights_sha256='3'*64,transform_sha256='4'*64)
    monkeypatch.setattr(evaluator,'load_target',target)
    uap=tmp_path/'uap.npy'; np.save(uap,np.zeros((1,3,224,224),dtype=np.float32))
    lock=tmp_path/'lock.json'; lock.write_text('{}')
    training=tmp_path/'training.json'
    write_json(training,dict(config=DEFAULTS,status='complete',uap_sha256=sha256(uap),
        manifest_hashes=index['manifest_hashes'],purpose=purpose,step=10000,run_id='fixture',seed=126,weights_sha256='5'*64))
    out=tmp_path/'out'
    a=Namespace(output=str(out),manifest=str(manifest),stage='final',epsilon=10/255,targets=['resnet50'],
        uap=str(uap),device='cpu',weights_lock=str(lock),weights_dir='unused',training_record=str(training),
        batch_size=1,workers=0,data_root=str(data))
    if expected=='failed':
        with pytest.raises(ValueError,match='smoke'):
            evaluator.evaluate(a)
    else:
        evaluator.evaluate(a)
        metrics=json.loads((out/'resnet50/metrics.json').read_text())
        assert metrics['fr']==0 and metrics['training_run_id']=='fixture' and metrics['variant']=='e2_noise_crops'
    assert json.loads((out/'provenance.json').read_text())['status']==expected
