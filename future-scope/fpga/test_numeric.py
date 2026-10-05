"""Generate independent NumPy/C-semantics fixed conversion and rounding cases."""
from pathlib import Path
import subprocess
import numpy as np


def numeric(engine,output):
    engine=Path(engine);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);words=[]
    for i in range(2048):
        max_exp=int(rng.integers(110,145));exp=int(rng.integers(max_exp-60,max_exp+1))
        raw=(int(rng.integers(0,2))<<31)|(exp<<23)|int(rng.integers(0,1<<23))
        value=np.array([raw],dtype=np.uint32).view(np.float32)[0]
        fixed=int(np.ldexp(float(value),166-max_exp))
        total=int(rng.integers(-(1<<45),(1<<45)))
        if i<20:total=(1<<25)+(2 if i%2==0 else 6);total=-total if i%3==0 else total
        if i==20:total=0
        expected=np.float32(np.ldexp(np.float32(total),max_exp-166)).view(np.uint32)
        words.append(f'{raw:08x}{max_exp:02x}{fixed & ((1<<64)-1):016x}{total & ((1<<64)-1):016x}{int(expected):08x}')
    path=output/'numeric.mem';path.write_text('\n'.join(words)+'\n')
    cmd=['iverilog','-g2012','-s','tb_numeric','-o',str(output/'numeric.vvp'),str(engine/'rtl/numeric.sv'),str(engine/'tb/tb_numeric.sv')]
    print('$ '+' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    cmd=['vvp',str(output/'numeric.vvp'),f'+VECTORS={path}',f'+COUNT={len(words)}']
    print('$ '+' '.join(cmd),flush=True);subprocess.run(cmd,check=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();numeric(Path(__file__).resolve().parent,a.output)
