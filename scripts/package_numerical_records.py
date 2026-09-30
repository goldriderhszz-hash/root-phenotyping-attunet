"""Package journal-sized numerical evidence separately from images and models."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT.parent/'submission_files'/'Additional_file_2_Numerical_records.zip'
def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for folder in ['analysis','research','validation']:
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file():continue
            parts=path.relative_to(ROOT).parts
            if any(p in parts for p in ['__pycache__','generated','figures','reference_metrics','mask_cache_before_regeneration','data','manifests']):continue
            if path.suffix.lower() not in ['.csv','.json','.py','.md','.txt']:continue
            if folder=='research' and path.suffix=='.py' and path.name not in ['phenotype_extractor.py','phenotype_extractor_fast.py','phenotype_extractor_grid.py']:continue
            files.append(path)
    files += [ROOT/name for name in ['requirements-analysis.txt','requirements.txt','LICENSE','DATA_AVAILABILITY.md','CITATION.cff']]
    # The geometry audit parses the frozen deployment angle rule from this file.
    # AST extraction needs its source only, not the GUI or model graphs.
    files.append(ROOT/'rootscope'/'phenotype_extractor.py')
    with zipfile.ZipFile(TARGET,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):z.write(path,Path('RootScope')/path.relative_to(ROOT))
        z.writestr('README_Additional_file_2.md','''# Additional file 2 — complete numerical records and replay

RootScope/ contains the frozen numerical evidence, split/hash metadata, all 60
fit histories/selections, analysis programs and verification records. It omits
biological images, masks, prediction image files, model graphs and checkpoints.

To replay table summaries without training:

```powershell
cd RootScope
python -m pip install -r requirements-analysis.txt
python analysis/reproduce_tables.py --output analysis/generated
python analysis/audit_uncertainty.py
python analysis/source_denominators.py
python analysis/angle_geometry_audit.py
python analysis/make_descriptor_figures.py
```

Some retained audit files describe full-software tests and source CPU timings;
their input/model assets are in the separately prepared software/research packages.
They are verification records, not an additional independent biological dataset.
Recomputing from masks or retraining needs owner-authorized access to the research
assets and full source. Numerical records alone reproduce arithmetic, not source
image annotations or independent anatomical truth. These records accompany RootScope Desktop. The main software's MIT license does not automatically
license the research data. See DATA_AVAILABILITY in the full source package.
''')
    assert TARGET.stat().st_size<20_000_000, 'Journal additional-file size exceeded'
    with zipfile.ZipFile(TARGET) as z:assert z.testzip() is None
    with TARGET.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps({'file':str(TARGET),'bytes':TARGET.stat().st_size,'sha256':digest,'files':len(set(files))+1,'CRC_all_entries_checked':True},indent=2))
if __name__=='__main__':main()
