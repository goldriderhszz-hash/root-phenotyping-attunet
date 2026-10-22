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
    for name in ['USER_GUIDE.md','USER_GUIDE.zh-CN.md','OUTPUT_SCHEMA.md','LIMITATIONS.md','TROUBLESHOOTING.md','PAPER_ALIGNMENT.md','JOURNAL_DESIGN.md']:
        shutil.copy2(ROOT/'docs'/name,docs/name)
    for name in ['LICENSE','MODEL.md','DATA_AVAILABILITY.md','CITATION.md']:shutil.copy2(ROOT/name,DEST/name)
    shutil.copytree(ROOT/'examples',DEST/'examples',dirs_exist_ok=True)
    shutil.copytree(ROOT/'templates',DEST/'templates',dirs_exist_ok=True)
    notices=ROOT/'build'/'license_notices'
    assert notices.resolve().is_relative_to((ROOT/'build').resolve())
    if notices.exists():shutil.rmtree(notices)
    notices.mkdir(parents=True)
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
    destination=DEST/'third_party_licenses'
    assert destination.resolve().is_relative_to(DEST.resolve())
    if destination.exists():shutil.rmtree(destination)
    shutil.copytree(notices,destination)
    (DEST/'START_HERE.md').write_text('''# RootScope

Extract this entire folder and open RootScope.exe. Keep _internal beside the EXE.
No Python or GPU installation is required. The GUI starts in English on first use;
select 中文 while idle for Chinese. See docs/USER_GUIDE.md or USER_GUIDE.zh-CN.md.

Automatic mode identifies study images and their recorded D3 fold by SHA-256.
New images use fold 0 and require review. The default pipeline is the paper's
whole-root primary arm E. Model metadata: _internal/model/paper/catalog.json.
Lengths are in pixels. Root identity, counts and angles need independent review.

For a synthetic installation check, run from this folder and wait for the process:

```powershell
.\\RootScope.exe --batch examples/synthetic_root.png --references examples/synthetic_root_mask.png --output demo-results --no-zip
```

The synthetic image is an installation example, not biological validation.
Python research/analysis sources are in the separately supplied source archive.
Third-party notices are retained under third_party_licenses/ and _internal/.
Source and build artifacts: https://github.com/goldriderhszz-hash/root-phenotyping-attunet
''',encoding='utf-8')
    print(json.dumps({'offline_guides':7,'packages_with_inventory':len(info),'license_files':sum(len(r['license_files']) for r in info),'python':sys.version.split()[0]},indent=2))
if __name__=='__main__':main()
