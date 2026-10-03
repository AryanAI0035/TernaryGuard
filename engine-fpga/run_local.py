#!/usr/bin/env python3
"""Icarus fallback validation. NOT a substitute for XSim or synthesis."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(engine,vectors):
    engine=Path(engine).resolve();vectors=Path(vectors).resolve();output=vectors/'local';output.mkdir(exist_ok=True)
    def execute(command,cwd=output):
        print('$ '+' '.join(map(str,command)),flush=True)
        result=subprocess.run(list(map(str,command)),cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,end='',flush=True)
        with (output/'commands.log').open('a') as log:log.write('$ '+' '.join(map(str,command))+'\n'+result.stdout)
        result.check_returncode()
    for tool in ['iverilog','vvp','iverilog-vpi']:
        if not shutil.which(tool):raise RuntimeError('Required local tool missing: '+tool)
    execute(['iverilog','-g2012','-s','tb_ternary_mac','-o',output/'mac.vvp',engine/'rtl/ternary_mac.sv',engine/'tb/tb_ternary_mac.sv'])
    execute(['vvp',output/'mac.vvp'])
    execute(['iverilog-vpi',engine/'tb/fp32_vpi.c'])
    files=[engine/'rtl'/name for name in ['numeric.sv','ternary_mac.sv','mac_array.sv','fp32_unit.sv','control.sv','top.sv']]
    execute(['iverilog','-g2012','-DTG_PORTABLE_SIM','-s','tb_top','-o',output/'top.vvp',*files,vectors/'model_rom.sv',engine/'tb/tb_top.sv'])
    samples=json.loads((vectors/'vectors.json').read_text())['samples']
    execute(['vvp','-M',output,'-m','fp32_vpi',output/'top.vvp',f'+SAMPLES={samples}',f'+INPUT={vectors/"inputs.mem"}',f'+OUTPUT={vectors/"iverilog_logits.mem"}'])
    execute([sys.executable,engine/'compare.py','--vectors',vectors,'--logits',vectors/'iverilog_logits.mem','--backend','iverilog'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vectors',type=Path,required=True)
    args=p.parse_args();run(Path(__file__).resolve().parent,args.vectors)
