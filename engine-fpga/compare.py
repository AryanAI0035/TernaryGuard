#!/usr/bin/env python3
"""Strict full-row comparison, including TCP identities. No board claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(folder, logits_path, backend):
    folder=Path(folder);metadata=json.loads((folder/'vectors.json').read_text())
    for filename,key in [('inputs.mem','input_sha256'),('model_rom.sv','rom_sha256'),('reference.npz','reference_sha256')]:
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=metadata[key]:
            raise ValueError('Frozen generated artifact changed: '+filename)
    with np.load(folder/'reference.npz') as data:
        expected=data['logits'].copy();labels=data['labels'].copy()
    words=Path(logits_path).read_text().split()
    if len(words)!=metadata['samples']*11:raise ValueError(f'Expected {metadata["samples"]*11} logit words; got {len(words)}')
    actual=np.array([int(word,16) for word in words],dtype=np.uint32).view(np.float32).reshape(-1,11)
    if not np.isfinite(actual).all():raise ValueError('Nonfinite RTL logit')
    prediction=np.array([int(v) for v in Path(str(logits_path)+'.predictions').read_text().split()],dtype=np.int64)
    if prediction.shape!=(metadata['samples'],):raise ValueError('Incomplete RTL prediction stream')
    if not np.array_equal(prediction,actual.argmax(1)):raise AssertionError('RTL argmax output disagrees with RTL logits')
    reference=expected.argmax(1)
    mismatch=np.flatnonzero(prediction!=reference)
    tcp=labels==9;hits=np.flatnonzero(tcp & (prediction==9))
    report=dict(scope='RTL simulation only; no physical FPGA',backend=backend,
        fp_model='Host binary32 arithmetic model, NOT AMD IP' if backend!='xsim' else 'AMD floating-point IP in XSim',
        vectors=metadata['vectors'],samples=len(labels),prediction_matches=int(len(labels)-len(mismatch)),
        prediction_disagreements=int(len(mismatch)),rtl_argmax_disagreements=0,mismatch_indices=mismatch.tolist(),
        max_abs_logit_error_vs_checkpoint=float(np.max(np.abs(actual-expected))),
        tolerance=dict(atol=2e-5,rtol=2e-5),
        logits_within_tolerance=bool(np.allclose(actual,expected,atol=2e-5,rtol=2e-5)),
        binary32_logit_word_matches=int(np.sum(actual.view(np.uint32)==expected.view(np.uint32))),
        accuracy=float(np.mean(prediction==labels)),
        bashlite_tcp=dict(total=int(tcp.sum()),true_positives=len(hits),true_positive_test_indices=hits.tolist()),
        active_hashes=metadata['active_hashes'])
    target=folder/f'{backend}-parity.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if len(mismatch):raise AssertionError('RTL predictions disagree with frozen checkpoint')
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    if metadata['vectors']=='test' and report['bashlite_tcp']!=dict(total=5555,true_positives=1,true_positive_test_indices=[59782]):
        raise AssertionError('Frozen TCP behavior changed')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors',type=Path,required=True);p.add_argument('--logits',type=Path,required=True)
    p.add_argument('--backend',choices=['iverilog','verilator','xsim'],required=True)
    a=p.parse_args();compare(a.vectors,a.logits,a.backend)
