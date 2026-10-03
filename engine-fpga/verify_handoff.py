#!/usr/bin/env python3
"""Verify every transferred byte before running the manual Vivado phase."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
manifest=json.loads((root/'handoff_manifest.json').read_text())
for name,expected in manifest['files'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('HANDOFF HASH MISMATCH: '+name)
print('HANDOFF_PASS files='+str(len(manifest['files'])))
print('Frozen checkpoint SHA256: '+manifest['active_hashes']['checkpoint_sha256'])
print('Scope: local Phase 6 complete; Phase 7 manual XSim/synthesis/timing/power pending')
