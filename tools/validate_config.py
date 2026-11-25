"""Print a fully resolved strict config; no training or model download."""
import _bootstrap
import argparse
import json
from psp_research.config import load_config
from psp_research.provenance import digest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = load_config(a.config)
    print(json.dumps(dict(config=c, config_sha256=digest(c)), indent=2))
