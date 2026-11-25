"""Build E1 matrix, matched E2/E3 ΔFR and seed mean/sample-std; retain missing cells."""
import _bootstrap
import argparse
import json
from psp_research.aggregate import aggregate, PILOT_TARGETS
from psp_research.registry import TARGETS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--targets',nargs='+',choices=TARGETS,default=PILOT_TARGETS)
    a=p.parse_args()
    print(json.dumps(aggregate(a.runs,a.out,a.targets),indent=2))
