"""Student-run V3: plan/smoke by default, explicit full optimization on A100."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','smoke','run','status','export'],default='plan')
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/a1_optimization_v3.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max-tasks',type=int,help='Stop cleanly after this many completed new tasks; same study resumes later')
    args=parser.parse_args(argv)
    if args.max_tasks is not None and args.max_tasks<1: parser.error('--max-tasks must be positive')
    from a1opt.study import prepare,smoke,execute,status,export_results
    if args.stage=='export': value=export_results(args.out)
    elif args.stage=='status': value=status(args.out)
    else:
        import torch
        if args.device=='cuda' and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0)):
            raise RuntimeError('Select a real NVIDIA A100 runtime; no CUDA training on a different device.')
        if args.stage=='run': value=execute(args.out,args.protocol,args.data,torch.device(args.device),args.max_tasks)
        else:
            manifest=prepare(args.out,args.protocol,args.data,torch.device(args.device))
            if args.stage=='smoke': smoke(args.out,manifest,args.data,torch.device(args.device))
            value=status(args.out,manifest)
    # The complete task ledger is exported; keep notebook status easy to read.
    display={key:item for key,item in value.items() if key!='tasks'}
    print(json.dumps(display,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
