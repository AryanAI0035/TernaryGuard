#!/usr/bin/env python3
"""Compiled RTL simulation with a local binary32 model, NOT Vivado acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(engine,vectors):
    engine=Path(engine).resolve();vectors=Path(vectors).resolve();output=vectors/'compiled';output.mkdir(exist_ok=True)
    def execute(command):
        command=list(map(str,command));line='$ '+' '.join(command)
        print(line,flush=True)
        with (output/'commands.log').open('a') as log:
            log.write(line+'\n');log.flush()
            process=subprocess.Popen(command,cwd=output,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if process.wait():raise subprocess.CalledProcessError(process.returncode,command)
    files=[engine/'rtl'/name for name in ['numeric.sv','ternary_mac.sv','mac_array.sv','fp32_unit.sv','control.sv','top.sv']]
    execute(['verilator','--binary','--timing','--build-jobs','4',
        '-DTG_PORTABLE_SIM','--top-module','tb_top','--Mdir',output/'obj',
        '-CFLAGS','-ffp-contract=off -fno-fast-math',*files,vectors/'model_rom.sv',engine/'tb/tb_top.sv',engine/'tb/fp32_dpi.cpp'])
    samples=json.loads((vectors/'vectors.json').read_text())['samples']
    execute([output/'obj/Vtb_top',f'+SAMPLES={samples}',f'+INPUT={vectors/"inputs.mem"}',f'+OUTPUT={vectors/"verilator_logits.mem"}'])
    execute([sys.executable,engine/'compare.py','--vectors',vectors,'--logits',vectors/'verilator_logits.mem','--backend','verilator'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vectors',type=Path,required=True)
    a=p.parse_args();run(Path(__file__).resolve().parent,a.vectors)
