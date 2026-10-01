"""PlatformIO pre-build: validate frozen artifacts and emit float32 PROGMEM scaler."""
import hashlib
import json
from pathlib import Path
import struct

def generate(root, destination):
    root=Path(root)
    active=json.loads((root/'model/active_model.json').read_text())
    manifest=json.loads((root/active['manifest']).read_text())
    cfg=json.loads((root/active['preprocessing']).read_text())
    for key in ('checkpoint','header'):
        if hashlib.sha256((root/active[key]).read_bytes()).hexdigest()!=manifest[key+'_sha256']:
            raise ValueError(key+' hash mismatch')
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if digest!=manifest['preprocessing_sha256'] or cfg['transform']!='signed_log1p':
        raise ValueError('Preprocessing mismatch')
    if active['architecture']!=dict(input_dim=20,hidden_dims=[64,32],output_dim=11,use_rmsnorm=True,ternary_output=True):
        raise ValueError('Architecture changed')
    text=['#ifndef TG_NANO_PREPROCESSING_H','#define TG_NANO_PREPROCESSING_H',
          '/* Generated from frozen JSON; float32 rounding is intentional. */']
    for name,key in [('nano_mean','mean'),('nano_scale','scale')]:
        values=[struct.unpack('<f',struct.pack('<f',v))[0] for v in cfg['scaler'][key]]
        text.append('static const float TGPROGMEM '+name+'[] = {'+', '.join(v.hex()+'f' for v in values)+'};')
    text.append('#endif\n')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    (destination/'nano_preprocessing.h').write_text('\n'.join(text))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    generate(Path(__file__).resolve().parents[1],args.output)
else:
    Import('env')
    root=Path(env.subst('$PROJECT_DIR')).parent
    destination=Path(env.subst('$BUILD_DIR'))/'generated'
    generate(root,destination)
    env.Append(CPPPATH=[str(destination)])
