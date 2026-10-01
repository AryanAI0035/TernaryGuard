#!/usr/bin/env python3
"""Real Nano serial validation. Never substitutes host timing for board timing."""
import argparse
import binascii
import json
from pathlib import Path
import statistics
import struct
import time


def packet(sequence, values, mode=1):
    if mode not in (1,2) or len(values)!=20:
        raise ValueError('Expected mode 1/2 and twenty selected features')
    body=struct.pack('<2sBBI20f',b'TG',1,mode,sequence,*values)
    return body+struct.pack('<H',binascii.crc_hqx(body,0xffff))


def response(frame, sequence):
    if len(frame)!=61: raise ValueError('Truncated board reply')
    if binascii.crc_hqx(frame[:-2],0xffff)!=struct.unpack('<H',frame[-2:])[0]:
        raise ValueError('Board reply CRC mismatch')
    magic,version,status,seq,pred,micros,gap,*logits=struct.unpack('<2sBBIBIH11f',frame[:-2])
    if magic!=b'TR' or version!=1 or seq!=sequence: raise ValueError('Reply identity mismatch')
    if status: raise ValueError('Board reported status '+str(status))
    return dict(sequence=seq,prediction=pred,inference_us=micros,untouched_gap_bytes=gap,logits=logits)


def run(args):
    import numpy as np
    import serial
    cases=np.load(args.cases,allow_pickle=False)
    indices=cases[args.group]
    if args.group=='subset' and (len(indices)!=330 or any(np.sum(cases['labels'][indices]==i)!=30 for i in range(11))):
        raise ValueError('Expected 330 stratified rows, thirty per class')
    if args.group=='tcp' and (len(indices)!=5555 or np.any(cases['labels'][indices]!=9)):
        raise ValueError('TCP verification requires all 5,555 rows')
    rows=cases['raw_selected' if args.mode=='raw' else 'preprocessed']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    received=[]
    with serial.Serial(args.port,115200,timeout=5,write_timeout=5) as board, args.output.open('w') as log:
        time.sleep(2)
        startup=board.read_all().decode('ascii',errors='replace')
        print('BOARD STARTUP '+startup.strip(),flush=True)
        log.write(json.dumps(dict(port=args.port,startup=startup,group=args.group,mode=args.mode,samples=len(indices)))+'\n')
        if 'TG_NANO_V1' not in startup: raise ValueError('Nano firmware handshake missing')
        board.reset_input_buffer()
        for sequence,index in enumerate(indices):
            board.write(packet(sequence,rows[index],1 if args.mode=='raw' else 2));board.flush()
            frame=board.read(61)
            log.write(json.dumps(dict(sequence=sequence,raw_reply_hex=frame.hex()))+'\n');log.flush()
            result=response(frame,sequence)
            result.update(test_index=int(index),true_label=int(cases['labels'][index]),
                expected_prediction=int(cases['expected_predictions'][index]))
            result['max_abs_logit_error']=float(np.max(np.abs(np.asarray(result['logits'])-cases['expected_logits'][index])))
            log.write(json.dumps(result)+'\n');log.flush()
            received.append(result)
            if (sequence+1)%30==0: print('Real board replies:',sequence+1,'/',len(indices),flush=True)
    mismatches=sum(r['prediction']!=r['expected_prediction'] for r in received)
    tcp=[r for r in received if r['true_label']==9]
    summary=dict(scope='PHYSICAL Nano via serial',port=args.port,mode=args.mode,samples=len(received),
        prediction_disagreements=mismatches,tcp_rows=len(tcp),tcp_true_positives=sum(r['prediction']==9 for r in tcp),
        median_inference_us=statistics.median(r['inference_us'] for r in received),
        min_inference_us=min(r['inference_us'] for r in received),max_inference_us=max(r['inference_us'] for r in received),
        min_untouched_gap_bytes=min(r['untouched_gap_bytes'] for r in received),
        watermark_used_sram_bytes=2048-min(r['untouched_gap_bytes'] for r in received),
        max_abs_logit_error=max(r['max_abs_logit_error'] for r in received))
    print(json.dumps(summary,indent=2))
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if mismatches or (args.group=='tcp' and summary['tcp_true_positives']!=1):
        raise AssertionError('Physical-board frozen-reference parity failed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--group',choices=['subset','tcp'],required=True)
    p.add_argument('--mode',choices=['raw','preprocessed'],default='raw')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
