"""Shared helpers for read-only manuscript analyses."""
from pathlib import Path
import ast, hashlib, json, os, random, sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'evaluation'
SECONDARY = ROOT / 'data' / 'images'
SEED = 20260903
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.cache' / 'matplotlib'))
sys.dont_write_bytecode = True

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def split_names():
    names=sorted({p.stem for p in SECONDARY.glob('*.tif') if 'mask' not in p.name.lower()})
    random.Random(42).shuffle(names)
    return names[:40],names[40:]

def read_mask(path):
    import cv2,numpy as np
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError(f'Cannot read {path}')
    return im

def load_functions(path, names, namespace):
    """Compile exact source definitions only: no __main__, directory writes, or demo data."""
    path=Path(path);source=path.read_text(encoding='utf-8-sig')
    tree=ast.parse(source,filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):raise ValueError('Missing required original definitions')
    module=ast.Module(body=nodes,type_ignores=[])
    env={'__name__':'original_functions_read_only',**namespace}
    exec(compile(module,str(path),'exec'),env)
    return env

def bootstrap_ci(values, seed=SEED, n=10000):
    import numpy as np
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),size=(n,len(a)))].mean(axis=1)
    return tuple(np.quantile(means,[.025,.975]))
