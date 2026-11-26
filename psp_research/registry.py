"""Explicit ImageNet-1k 224 weight enums; never DEFAULT or pretrained=True."""
from pathlib import Path
from urllib.parse import urlparse
import json
import torch
from torchvision import models, transforms
from .provenance import sha256, digest, write_json

# Legacy CNNs retain the checkpoint selected by pretrained=True in torchvision 0.12.
TARGETS = {
    'resnet152': ('ResNet152_Weights', 'IMAGENET1K_V1', 'CNN'),
    'vgg16': ('VGG16_Weights', 'IMAGENET1K_V1', 'CNN'),
    'resnet50': ('ResNet50_Weights', 'IMAGENET1K_V1', 'CNN'),
    'densenet121': ('DenseNet121_Weights', 'IMAGENET1K_V1', 'CNN'),
    'convnext_base': ('ConvNeXt_Base_Weights', 'IMAGENET1K_V1', 'ConvNeXt'),
    'swin_t': ('Swin_T_Weights', 'IMAGENET1K_V1', 'Swin'),
    'vit_b_16': ('ViT_B_16_Weights', 'IMAGENET1K_V1', 'ViT'),
}
E0_TARGETS = {
    'alexnet': ('AlexNet_Weights', 'IMAGENET1K_V1', 'CNN'),
    'vgg19': ('VGG19_Weights', 'IMAGENET1K_V1', 'CNN'),
    'googlenet': ('GoogLeNet_Weights', 'IMAGENET1K_V1', 'CNN'),
}
E0_NAMES = ['alexnet', 'vgg16', 'vgg19', 'resnet152', 'googlenet']


def specification(name):
    enum, member, family = (TARGETS | E0_TARGETS)[name]
    weight = getattr(getattr(models, enum), member)
    t = weight.transforms()
    if t.crop_size != [224]:
        raise ValueError('registry supports only native 224 checkpoints')
    spec = dict(target=name, family=family, weights_id=f'{enum}.{member}', url=weight.url,
                dataset='ImageNet-1k', crop_size=t.crop_size, resize_size=t.resize_size,
                interpolation=t.interpolation.value, antialias=t.antialias,
                mean=t.mean, std=t.std)
    spec['transform_sha256'] = digest({k: spec[k] for k in (
        'crop_size', 'resize_size', 'interpolation', 'antialias', 'mean', 'std')})
    return spec


def lock_weights(names, cache, output):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in names:
        s = specification(name)
        path = cache / Path(urlparse(s['url']).path).name
        # Download API checks the official filename hash prefix; record the full digest.
        prefix = path.stem.rsplit('-', 1)[1]
        if not path.exists():
            torch.hub.download_url_to_file(s['url'], str(path), hash_prefix=prefix)
        actual = sha256(path)
        if not actual.startswith(prefix):
            raise ValueError('cached official checkpoint checksum mismatch')
        result[name] = dict(s, file_name=path.name, weights_sha256=actual)
    write_json(output, {'schema_version': 1, 'targets': result})
    return result


def load_target(name, lock_file, cache, device):
    entry = json.loads(Path(lock_file).read_text())['targets'][name]
    spec = specification(name)
    if any(entry[k] != v for k, v in spec.items()):
        raise ValueError('weight/transform lock disagrees with installed registry')
    path = Path(cache) / entry['file_name']
    if Path(entry['file_name']).name != entry['file_name'] or sha256(path) != entry['weights_sha256']:
        raise ValueError('checkpoint digest mismatch')
    kwargs = dict(transform_input=True, init_weights=False) if name == 'googlenet' else {}
    model = getattr(models, name)(weights=None, **kwargs)
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Historical DenseNet checkpoint keys require torchvision's official adapter.
    if name == 'densenet121':
        import re
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state):
            m = pattern.match(key)
            if m:
                state[m[1] + m[2]] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    geometry = transforms.Compose([
        transforms.Resize(spec['resize_size'], interpolation=transforms.InterpolationMode(spec['interpolation']),
                          antialias=spec['antialias']),
        transforms.CenterCrop(spec['crop_size']), transforms.ToTensor()])
    return model, geometry, entry
