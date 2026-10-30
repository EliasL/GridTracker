"""Run independent frames in parallel, with one numeric thread per worker."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from pipeline import run_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('testData'))
    parser.add_argument('--output',type=Path,default=Path('output/final_batch'))
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    args=parser.parse_args()
    files=list(args.input.glob('*.jpeg'))
    files.sort(key=lambda p:getattr(p.stat(),'st_birthtime',p.stat().st_mtime))
    if not files:raise ValueError('No JPEG frames found')
    if args.workers<1:raise ValueError('Workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    results=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_frame,(i,str(path),str(args.output))):(i,path) for i,path in enumerate(files)}
        for future in as_completed(futures):
            i,path=futures[future]
            try:
                r=future.result();results.append(r)
                print(f"Frame {i}: width {r['width']:.2f}, {r['post_maxima']} nodes -> {len(r['final_nodes'])}; {len(r['curves'])} curves",flush=True)
            except Exception:
                error={'frame':i,'source':str(path),'traceback':traceback.format_exc()};errors.append(error)
                print(f'Frame {i} FAILED: {error["traceback"]}',flush=True)
    manifest={'workers':args.workers,'threads_per_worker':1,'results':sorted(results,key=lambda r:r['frame']),'errors':errors,
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('.').glob('*.py'))},
              'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if errors:raise RuntimeError(f'{len(errors)} frames failed; see manifest.json')


if __name__=='__main__':main()
