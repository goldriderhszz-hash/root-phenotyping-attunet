"""Probe the frozen angle rule on constructed geometry with a fixed reference axis.

Synthetic geometry verifies implementation properties; it is not biological validation.
"""
from pathlib import Path
import ast
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def frozen_angle_function():
    # Extract the exact executable block from the frozen production extractor.
    tree=ast.parse((ROOT/'rootscope'/'phenotype_extractor.py').read_text(encoding='utf-8'))
    extractor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_descriptors')
    loop=next(n for n in ast.walk(extractor) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='component')
    body=[]
    for n in loop.body:
        if isinstance(n,ast.If):
            # Convert the production loop's invalid-vector branch into a function return.
            n.body=[ast.Return(value=ast.Constant(None))]
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append':
            n=ast.Return(value=n.value.args[0])
        body.append(n)
    fn=ast.FunctionDef(name='frozen_angle',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='component'),ast.arg(arg='primary_path')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.Assign(targets=[ast.Name(id='path_array',ctx=ast.Store())],value=ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='asarray',ctx=ast.Load()),args=[ast.Name(id='primary_path',ctx=ast.Load())],keywords=[]))]+body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace={'np':np,'math':math};exec(compile(module,'frozen_angle_block','exec'),namespace)
    return namespace['frozen_angle']

def audit():
    angle=frozen_angle_function();axis=np.array([(y,-1.) for y in range(80)])
    cases=[]
    for degrees in [30,45,60,90]:
        r=np.arange(0,31,dtype=float)
        component=np.c_[r*np.cos(np.deg2rad(degrees)),r*np.sin(np.deg2rad(degrees))]
        measured=angle(component,axis)
        assert measured is not None and abs(measured-degrees)<1e-10
        cases.append({'case':f'straight_{degrees}_degrees','expected_deg':degrees,'measured_deg':measured})
    curved=np.array([(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5),(12,5),(13,5),(14,5),(15,5),(16,5),(17,5),(18,5),(19,5),(20,5)],dtype=float)
    ordered=angle(curved,axis);reversed_order=angle(curved[::-1],axis)
    mirrored=curved.copy();mirrored[:,1]*=-1
    mirrored_axis=axis.copy();mirrored_axis[:,1]*=-1
    mirror_angle=angle(mirrored,mirrored_axis)
    assert abs(ordered-mirror_angle)<1e-10
    assert abs(ordered-reversed_order)>1, 'The intended coordinate-order probe did not expose the design property.'
    cases.extend([{'case':'curved_original_order','measured_deg':ordered},{'case':'same_geometry_reversed_coordinate_order','measured_deg':reversed_order},{'case':'horizontal_mirror_with_matching_fixed_axis','measured_deg':mirror_angle}])
    empty_vector=angle(np.array([(0,0)],dtype=float),axis)
    assert empty_vector is None
    cases.append({'case':'zero_length_lateral_vector','measured_deg':None,'expected':'unavailable'})
    report={'status':'passed','n_cases':len(cases),'cases':cases,'coordinate_order_difference_deg':abs(ordered-reversed_order),'interpretation':'The implementation matches straight geometry and mirror transformation with corresponding axis. The same curved geometry changes angle under coordinate reordering. The deterministic argwhere ordering is part of the frozen definition, not a newly discovered coding mismatch. Local angles should be treated as exploratory operational descriptors. No parameter was retuned.'}
    return report

if __name__=='__main__':
    output=ROOT/'analysis'/'generated';output.mkdir(parents=True,exist_ok=True)
    report=audit();(output/'angle_geometry_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
