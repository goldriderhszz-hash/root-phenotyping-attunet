import csv,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'whole_root_all250_image_method_rows.csv').open(encoding='utf-8')))
assert len(rows)==250 and len({r['image'] for r in rows})==50
expected=json.loads((p/'whole_root_summary.json').read_text(encoding='utf-8'))
for e in expected:
 g=[r for r in rows if r['method']==e['arm']]; assert len(g)==50
 assert sum(int(r['reference_roots']) for r in g)==3290
 mapping={'output_objects':'predicted_roots','matched50':'matched50','matched75':'matched75','matched90':'matched90','fragmented_references':'fragmented_references','mixed_outputs':'mixed_predictions','partial_outputs':'high_support_partial_objects'}
 for source,target in mapping.items():assert sum(int(r[source]) for r in g)==e[target],(e['arm'],target)
 assert abs(statistics.mean(float(r['count_MAE']) for r in g)-e['count_MAE'])<=1e-12
 print(e['arm'],e['matched75'],e['count_MAE'])
print('PASS: all5arms,250rows,50images,3290reference roots; no fits or inference')
