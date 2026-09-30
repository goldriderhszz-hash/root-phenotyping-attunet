"""Add offline guides and available third-party notices to the Windows folder."""
from pathlib import Path
import importlib.metadata as metadata
import json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'dist'/'RootScope'
PACKAGES=['numpy','scipy','scikit-image','Pillow','onnxruntime','opencv-python-headless','tkinterdnd2','networkx','imageio','tifffile','lazy-loader','packaging','flatbuffers','protobuf','pyinstaller']
def main():
    assert (DEST/'RootScope.exe').exists()
    docs=DEST/'docs';docs.mkdir(exist_ok=True)
    for name in ['USER_GUIDE.md','USER_GUIDE.zh-CN.md','OUTPUT_SCHEMA.md','LIMITATIONS.md','TROUBLESHOOTING.md']:
        shutil.copy2(ROOT/'docs'/name,docs/name)
    for name in ['LICENSE','MODEL.md','RELEASE_NOTES.md']:shutil.copy2(ROOT/name,DEST/name)
    shutil.copytree(ROOT/'examples',DEST/'examples',dirs_exist_ok=True)
    notices=ROOT/'third_party_licenses';notices.mkdir(exist_ok=True)
    info=[]
    for name in PACKAGES:
        dist=metadata.distribution(name);copied=[]
        for rel in dist.files or []:
            filename=Path(str(rel)).name.lower()
            if not (filename.startswith(('license','copying','thirdpartynotice','notice')) or filename=='license.terms'):continue
            source=dist.locate_file(rel)
            if not source.is_file():continue
            target=notices/name/str(rel).replace('../','parent/')
            target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,target)
            copied.append(target.relative_to(notices).as_posix())
        info.append({'package':name,'version':dist.version,'license_files':copied})
    python_license=Path(sys.base_prefix)/'LICENSE.txt'
    if python_license.exists():shutil.copy2(python_license,notices/'Python-LICENSE.txt')
    tk_license=DEST/'_internal'/'_tk_data'/'license.terms'
    if tk_license.exists():shutil.copy2(tk_license,notices/'Tk-license.terms')
    (notices/'package_inventory.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    shutil.copytree(notices,DEST/'third_party_licenses',dirs_exist_ok=True)
    (DEST/'START_HERE.md').write_text('''# RootScope Desktop 1.1.0

Extract this entire folder and open RootScope.exe. Keep _internal beside the EXE.
No Python or GPU installation is required. The GUI starts in English on first use;
select 中文 while idle for Chinese. See docs/USER_GUIDE.md or USER_GUIDE.zh-CN.md.

Model fold_0 is the default, not an ensemble or an externally validated final model.
Its locked threshold is 0.47. MODEL.md lists all five model hashes and thresholds.
Length is in pixels; counts are operational descriptors and angles exploratory.
No flags is not a biological accuracy verdict.

For a synthetic installation check, run from this folder and wait for the process:

```powershell
.\\RootScope.exe --batch examples/synthetic_root.png --references examples/synthetic_root_mask.png --output demo-results --no-zip
```

The synthetic image is an installation example, not biological validation.
Python research/analysis sources are in the separately supplied source archive.
Third-party notices are retained under third_party_licenses/ and _internal/.
Versioned downloads and source: https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/v1.1.0
''',encoding='utf-8')
    print(json.dumps({'offline_guides':5,'packages_with_inventory':len(info),'license_files':sum(len(r['license_files']) for r in info),'python':sys.version.split()[0]},indent=2))
if __name__=='__main__':main()
