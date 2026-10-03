"""Phase 6 LOCAL simulation regressions; these do not certify Vivado or a board."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'engine-fpga'


def module(name):
    spec=importlib.util.spec_from_file_location('fpga_'+name,ENGINE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def require(tool):
    if not shutil.which(tool):pytest.skip('Local RTL test requires '+tool+'; no Vivado/board claim')


def unit(folder,top,files,args=()):
    require('iverilog');require('vvp')
    executable=folder/(top+'.vvp')
    subprocess.run(['iverilog','-g2012','-s',top,'-o',str(executable),*[str(ENGINE/p) for p in files]],check=True)
    result=subprocess.run(['vvp',str(executable),*args],check=True,text=True,capture_output=True)
    assert '_PASS' in result.stdout,result.stdout


def test_fpga_signed_mac_reserved_code_enable_and_clear(tmp_path):
    unit(tmp_path,'tb_ternary_mac',['rtl/ternary_mac.sv','tb/tb_ternary_mac.sv'])


def test_fpga_eight_lane_mac_array_independent_accumulators(tmp_path):
    unit(tmp_path,'tb_mac_array',['rtl/ternary_mac.sv','rtl/mac_array.sv','tb/tb_mac_array.sv'])


def test_fpga_fixed_conversion_and_rne_restore(tmp_path):
    require('iverilog');require('vvp');module('test_numeric').numeric(ENGINE,tmp_path)


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    folder=tmp_path_factory.mktemp('fpga-golden')
    module('prepare').prepare(ROOT,folder,'golden')
    return folder


@pytest.fixture(scope='module')
def simulated(generated):
    require('verilator');module('run_compiled').run(ENGINE,generated)
    return generated


def test_fpga_frozen_golden_network_and_stream_backpressure(simulated):
    report=json.loads((simulated/'verilator-parity.json').read_text())
    assert report['samples']==264 and report['prediction_disagreements']==0
    assert report['rtl_argmax_disagreements']==0 and report['logits_within_tolerance']


def test_fpga_rom_matches_exact_header_all_packed_weights(generated):
    rom=(generated/'model_rom.sv').read_text();header=(ROOT/'model_weights.h').read_text()
    actual=[int(v,16) for v in re.findall(r"weight_byte_mem\[\d+\]=8'h([0-9a-f]+);",rom)]
    expected=[]
    for i in range(3):
        body=re.search(r'tg_weights_'+str(i)+r'\[\]\s*=\s*\{(.*?)\};',header,re.S).group(1)
        expected.extend(int(v.strip(),0) for v in body.split(',') if v.strip())
    assert actual==expected and len(actual)==920
    codes=[(byte>>(2*k))&3 for byte in actual for k in range(4)]
    assert len(codes)==3680 and set(codes)<={0,1,2}


def test_fpga_generated_rom_tamper_is_rejected(simulated,tmp_path):
    for name in ['vectors.json','model_rom.sv','inputs.mem','reference.npz']:
        shutil.copy2(simulated/name,tmp_path/name)
    rom=tmp_path/'model_rom.sv';rom.write_text(rom.read_text().replace("bias_word_mem[0]=32'h", "bias_word_mem[0]=32'hf",1))
    with pytest.raises(ValueError,match='model_rom.sv'):
        module('compare').compare(tmp_path,simulated/'verilator_logits.mem','verilator')


def test_fpga_wrong_hardware_argmax_is_rejected(simulated,tmp_path):
    logits=tmp_path/'logits.mem';shutil.copy2(simulated/'verilator_logits.mem',logits)
    words=Path(str(simulated/'verilator_logits.mem')+'.predictions').read_text().split()
    words[0]=str((int(words[0])+1)%11)
    Path(str(logits)+'.predictions').write_text('\n'.join(words)+'\n')
    with pytest.raises(AssertionError,match='argmax'):
        module('compare').compare(simulated,logits,'verilator')


def test_fpga_synthesis_wrapper_serialization_and_backpressure(generated,tmp_path):
    require('iverilog');require('vvp');require('iverilog-vpi')
    subprocess.run(['iverilog-vpi',str(ENGINE/'tb/fp32_vpi.c')],cwd=tmp_path,check=True)
    files=[ENGINE/'rtl'/name for name in ['numeric.sv','ternary_mac.sv','mac_array.sv','fp32_unit.sv','control.sv','top.sv','basys3_top.sv']]
    executable=tmp_path/'wrapper.vvp';logits=tmp_path/'wrapper_logits.mem'
    subprocess.run(['iverilog','-g2012','-DTG_PORTABLE_SIM','-s','tb_basys3_top','-o',str(executable),*map(str,files),str(generated/'model_rom.sv'),str(ENGINE/'tb/tb_basys3_top.sv')],check=True)
    subprocess.run(['vvp','-M',str(tmp_path),'-m','fp32_vpi',str(executable),'+SAMPLES=264',f'+INPUT={generated/"inputs.mem"}',f'+OUTPUT={logits}'],check=True)
    report=module('compare').compare(generated,logits,'iverilog')
    assert report['samples']==264 and report['prediction_disagreements']==0
