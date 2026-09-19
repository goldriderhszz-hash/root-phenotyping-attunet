# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH).resolve().parent
model_dir = root / "software" / "models"

a = Analysis(
    [str(root / "software" / "main.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[
        (str(model_dir / "precision_boost_attention_unet.onnx"), "models"),
        (str(model_dir / "precision_boost_attention_unet.onnx.data"), "models"),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI_Phenotype_Tool",
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="AI_Phenotype_Tool",
)
