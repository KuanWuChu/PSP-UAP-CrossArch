"""Record hardware/software and distinguish modern compatibility from faithful E0."""
import _bootstrap
import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
from pathlib import Path
from psp_research.provenance import write_json, git_identity


def doctor(data_root=None, lock=None, weights_dir=None):
    result = dict(git_identity(), python=platform.python_version(), execution_label='modern_compatibility',
                  faithful_e0='NOT_RUN: original Python3.8/torch1.11/torchvision0.12/CUDA11.3 unvalidated',
                  data_status='BLOCKED: ImageNet root not supplied', deit3='OPTIONAL_NOT_CONFIGURED')
    result['packages'] = {}
    for name in ['torch', 'torchvision', 'numpy', 'pillow', 'scipy', 'scikit-image', 'opencv-python', 'imageio', 'pytest']:
        try:
            result['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            result['packages'][name] = None
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=15)
        result['nvidia_smi'] = p.stdout.strip() if p.returncode == 0 else 'unavailable'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result['nvidia_smi'] = 'unavailable'
    try:
        import torch
        import torchvision
        result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), compiled_arches=torch.cuda.get_arch_list())
        result['gpus'] = [dict(name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i)),
                              vram_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():
            x = torch.ones(8, device='cuda')
            result['cuda_kernel_probe'] = float((x @ x).cpu()) == 8
        else:
            result['cuda_kernel_probe'] = None
        result['legacy_compatibility'] = ('UNVALIDATED: Ada has no native sm89 in CUDA11.3; older Ampere/PTX kernels may work; '
            'torch1.11/cuDNN coverage not tested. Do not call modern run faithful.')
        if lock:
            from psp_research.registry import specification
            from psp_research.provenance import sha256
            result['weights'] = {}
            for name, entry in json.loads(Path(lock).read_text())['targets'].items():
                s = specification(name)
                path = Path(weights_dir or 'models_cache') / entry['file_name']
                result['weights'][name] = dict(weights_id=entry['weights_id'], weights_sha256=entry['weights_sha256'],
                    verified=path.is_file() and sha256(path) == entry['weights_sha256'] and all(entry[k] == v for k,v in s.items()))
    except Exception as e:
        result['environment_error'] = type(e).__name__ + ': ' + str(e)
    if data_root:
        result['data_status'] = 'PRESENT_UNVERIFIED: run devkit/manifests checks' if Path(data_root).is_dir() else 'BLOCKED: supplied root missing'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/doctor.json')
    p.add_argument('--data-root', default=os.environ.get('IMAGENET_VAL_ROOT'))
    p.add_argument('--weights-lock')
    p.add_argument('--weights-dir')
    a = p.parse_args()
    r = doctor(a.data_root, a.weights_lock, a.weights_dir)
    write_json(a.output, r)
    print(json.dumps(r, indent=2))
