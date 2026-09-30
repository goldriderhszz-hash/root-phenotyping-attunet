"""Audit width-stratum denominators and recoverable existing-image metadata."""
from pathlib import Path
import argparse,csv,json,re
import numpy as np
from PIL import Image,TiffTags
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path);a=parser.parse_args()
    with (ROOT/'tables'/'all_oof_image_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    sources={}
    bins=[('width_le3','≤3 pixels'),('width_3_6','>3–6 pixels'),('width_6_12','>6–12 pixels'),('width_gt12','>12 pixels')]
    for row in rows:
        counts={key:int(row[key+'_n']) for key,_ in bins}
        if row['name'] in sources:assert sources[row['name']]==counts
        else:sources[row['name']]=counts
    assert len(sources)==50
    denominators=[]
    for key,label in bins:
        n=np.array([s[key] for s in sources.values()])
        denominators.append({'stratum':label,'distinct_images_with_reference_points':int((n>0).sum()),'images_without_reference_points':int((n==0).sum()),'total_reference_centerline_points':int(n.sum()),'minimum_points_per_image':int(n.min()),'maximum_points_per_image':int(n.max())})
    target=ROOT/'generated';target.mkdir(exist_ok=True)
    with (target/'width_stratum_denominators.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(denominators[0]));w.writeheader();w.writerows(denominators)
    report={'status':'passed','distinct_images':50,'width_strata':denominators,'unit':'annotation-centerline points on 50 distinct source images; repeated configurations/seeds do not add images','filename_suffixes_do_not_establish_plant_identity':True}
    if a.data:
        metadata=[]
        for name in sorted(sources):
            with Image.open(a.data/(name+'.tif')) as image:
                tags={TiffTags.TAGS.get(k,str(k)):str(v) for k,v in image.tag_v2.items()}
                acquisition={k:v for k,v in tags.items() if k in ['Make','Model','DateTime','Software','Artist','ImageDescription','XResolution','YResolution','ResolutionUnit']}
            annotation=json.loads((a.data/(name+'.json')).read_text(encoding='utf-8'))
            metadata.append({'sample_name':name,'filename_without_parenthetical_suffix':re.sub(r'\(\d+\)$','',name),'TIFF_tags':acquisition,'annotation_version':annotation.get('version'),'shape_count':len(annotation.get('shapes',[])),'shape_types':sorted({s.get('shape_type','unknown') for s in annotation.get('shapes',[])})})
        report['existing_metadata']=metadata
        report['metadata_interpretation']='File suffixes, image/software tags and file times are not validated plant/batch IDs. Resolution tags describe file display density and do not establish botanical pixel-to-length calibration. No camera, annotator identity, acquisition batch or biological grouping is inferred without supporting records.'
    (target/'source_denominators.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'passed','width_strata':denominators,'metadata_images':len(report.get('existing_metadata',[]))},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
