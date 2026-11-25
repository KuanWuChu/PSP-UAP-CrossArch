import torch
from psp_research.config import load_config
from psp_research.rng import RNGStreams
from psp_research.trainer import build_sources


def test_only_prespecified_crop_range_changes():
    configs=[load_config('configs/experiments/'+v+'.json') for v in ['full_original','full_local','full_large']]
    assert [c['crop_area'] for c in configs]==[[.08,1],[.08,.35],[.5,1]]
    stripped=[{k:v for k,v in c.items() if k not in {'variant','crop_area'}} for c in configs]
    assert stripped[0]==stripped[1]==stripped[2]
    prior=torch.rand(1,3,224,224); delta=torch.zeros_like(prior,requires_grad=True)
    for c in configs:
        with RNGStreams(126).use('crop'):
            images,sources,boxes=build_sources(prior,delta,c['source_mode'],10,c['crop_area'],c['aspect_ratio'])
        areas=[h*w/224**2 for h,w,_,_ in boxes]
        # Integer rounding of crop edges allows a small area discrepancy.
        assert min(areas)>=c['crop_area'][0]-.005 and max(areas)<=c['crop_area'][1]+.005
        assert images.shape==(10,3,224,224) and not sources.requires_grad
