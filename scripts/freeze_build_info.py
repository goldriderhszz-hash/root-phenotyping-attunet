"""Identify the exact source payload even when an executable lacks a Git checkout."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
files=sorted(list((ROOT/'rootscope').rglob('*.py'))+[ROOT/'desktop.py',ROOT/'cli.py'])
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
(ROOT/'rootscope/build_info.json').write_text(json.dumps(dict(application='RootScope',source_fingerprint_sha256=fingerprint,files=hashes),indent=2)+'\n',encoding='utf-8')
print(fingerprint)
