"""Explicit CLI: preflight, train, resume, export. Importing never trains."""
import argparse
import json
from pathlib import Path

import yaml

from kt1k.engine import export_best,json_write,load,preflight_report,train


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','train','export'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,help='Stop after this global update count; probe directory only')
    args=p.parse_args();config=yaml.safe_load(args.config.read_text())
    if args.command=='preflight':
        manifest,spec,model,source,validation,eligible,device=load(args.data,config)
        result=preflight_report(model,source,validation,eligible,device,spec,config)
        args.output.mkdir(parents=True,exist_ok=True)
        json_write(args.output/'feature_acceptance.json',result);print(json.dumps(result,indent=2))
    elif args.command=='train':train(args.data,config,args.output,args.resume,args.stop_after)
    else:export_best(args.data,config,args.output)


if __name__=='__main__':main()
