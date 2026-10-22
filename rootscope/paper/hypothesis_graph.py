"""Root-identity route candidates; no lowest-tip or root-wide shortest-path rule.

Experimental routing uses a graph with short burrs suppressed and narrow
enclosed lumens filled. Open gaps remain open. Image coordinates and pixel
arc lengths are retained; biological root identity still needs validation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import copy
import cv2
import numpy as np
from skimage.morphology import skeletonize

VERSION = 'mainroot-direction-hypotheses-20261015.13'


@dataclass(frozen=True)
class RouteSettings:
    up_direction: tuple[float, float] = (-1., 0.)
    scale: float = 1.
    max_candidates: int = 32
    max_expansions: int = 20000
    ambiguity_cost: float = .16
    lumen_radius: float = 5.
    suppress_burrs: bool = True
    basal_top_fraction: float = .12
    canonical_support: bool = True
    rebuild_skeleton: bool = True
    zero_boundary_radius: bool = True
    bounded_contacts: bool = False
    crown_contacts: int = 6
    crown_trace_budget: int = 192
    crown_display_candidates: int = 6


def arc(points):
    p = np.asarray(points, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _graph(skeleton, mask, up_direction=(-1.,0.), suppress_burrs=True, zero_boundary_radius=True, bounded_contacts=False):
    pixels = {tuple(p) for p in np.argwhere(skeleton)}
    # Contacts often contain equally central raster pixels. Image-row/column
    # tie breaks change their bookkeeping connector under a rigid transform.
    # Use the known basal orientation and intrinsic principal direction only
    # to order graph construction; this does not define the primary root.
    up=np.asarray(up_direction,float)
    cloud=np.asarray(list(pixels),float) if pixels else np.empty((0,2),float)
    center=cloud.mean(axis=0) if len(cloud) else np.zeros(2)
    intrinsic=up.copy()
    if len(cloud)>2:
        _,v=np.linalg.eigh((cloud-center).T@(cloud-center));intrinsic=v[:,-1]
        if intrinsic@up<0:intrinsic=-intrinsic
        if abs(intrinsic@up)<1e-8:intrinsic=up.copy()
    # Cache the intrinsic ordering once; group centroids cannot change it.
    point_keys={p:(-round(float((np.asarray(p)-center)@up),9),-round(float((np.asarray(p)-center)@intrinsic),9),p) for p in pixels}
    def point_key(p):return point_keys[p]
    adjacency = {}
    for y, x in pixels:
        ns = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y+dy, x+dx)
                if q not in pixels or not (dy or dx):
                    continue
                # Diagonal triangles are graph redundancy, not an extra root.
                if dy and dx and ((y+dy, x) in pixels or (y, x+dx) in pixels):
                    continue
                ns.append(q)
        adjacency[(y, x)] = sorted(ns,key=point_key)
    radius_mask=np.asarray(mask,np.uint8)
    radius=(cv2.distanceTransform(np.pad(radius_mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            if zero_boundary_radius else cv2.distanceTransform(radius_mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
    _,component_labels=cv2.connectedComponents(skeleton,8)
    component_tops={}
    for p in pixels:
        label=int(component_labels[p]);projection=float(np.asarray(p)@np.asarray(up_direction))
        component_tops[label]=max(component_tops.get(label,-math.inf),projection)
    # Skeleton burrs are shorter than the thickness of the contact that
    # generated them. Remove only dangling chains, never bridge a gap.
    # The upper entry of each component is retained using specimen coordinates,
    # so padding the photograph cannot change pruning decisions.
    removed=set()
    for _ in range(8 if suppress_burrs else 0):
        prune=set()
        for origin in sorted((p for p in pixels if len(adjacency[p])==1),key=point_key):
            if component_tops[int(component_labels[origin])]-float(np.asarray(origin)@np.asarray(up_direction))<=3:
                continue
            chain=[origin];previous=origin;current=adjacency[origin][0];length=1.
            while len(adjacency[current])==2:
                chain.append(current);nxt=next(p for p in adjacency[current] if p!=previous)
                length+=float(np.linalg.norm(np.asarray(nxt)-current));previous,current=current,nxt
            if len(adjacency[current])>=3 and length<3*float(radius[current]):
                prune.update(chain)
        if not prune:break
        removed.update(prune);pixels-=prune
        adjacency={p:[q for q in ns if q not in prune] for p,ns in adjacency.items() if p not in prune}
    critical = {p for p in pixels if len(adjacency[p]) != 2}
    # A single rasterised crossing can have two adjacent Y nodes separated by
    # a very short link. Bundle only junction-to-junction links within the local
    # contact width. Endpoints are never extended or pruned here.
    expanded=set(critical)
    for origin in sorted(critical,key=point_key):
        if len(adjacency[origin])<3:
            continue
        limit=max(1.5,1.5*float(radius[origin]))
        for first in adjacency[origin]:
            previous=origin;current=first;chain=[];distance=float(np.linalg.norm(np.asarray(first)-origin))
            while current not in critical and distance<=limit:
                chain.append(current)
                nxt=next(p for p in adjacency[current] if p!=previous)
                distance+=float(np.linalg.norm(np.asarray(nxt)-current))
                previous,current=current,nxt
            if current in critical and len(adjacency[current])>=3 and distance<=limit:
                expanded.update(chain)
    critical=expanded
    pending = set(pixels)
    pure_loop = False
    while pending:
        first = min(pending,key=point_key); stack = [first]; component = set()
        while stack:
            p = stack.pop()
            if p not in pending:
                continue
            pending.remove(p); component.add(p); stack.extend(adjacency[p])
        if not (component & critical):
            critical.add(min(component,key=point_key)); pure_loop = True
    groups = []; pending = set(critical)
    while pending:
        first = min(pending,key=point_key); pending.remove(first); group = {first}; stack = [first]
        while stack:
            for q in adjacency[stack.pop()]:
                if q in pending:
                    if bounded_contacts:
                        # Keep endpoints as endpoints; do not absorb them into
                        # a contact. Bound the complete cluster, preventing an
                        # arbitrarily long chain of short links from merging.
                        if len(adjacency[q])==1 or any(len(adjacency[p])==1 for p in group):continue
                        proposed=np.asarray(list(group)+[q])
                        local_width=max(1.,float(np.median(2*radius[proposed[:,0],proposed[:,1]])))
                        if float(np.linalg.norm(np.ptp(proposed,axis=0)))>2*local_width:continue
                    pending.remove(q); group.add(q); stack.append(q)
        groups.append(group)
    belongs = {p: i for i, group in enumerate(groups) for p in group}
    nodes = []; paths = []; internal_sizes = []
    for group in groups:
        center = np.mean(list(group), axis=0)
        rep = min(group, key=lambda p: (round(float(np.sum((p-center)**2)),9), point_key(p)))
        nodes.append(rep); previous = {rep: None}; queue = deque([rep])
        while queue:
            p = queue.popleft()
            for q in adjacency[p]:
                if q in group and q not in previous:
                    previous[q] = p; queue.append(q)
        local_paths = {}
        for p in group:
            chain = []; q = p
            while q is not None:
                chain.append(q); q = previous[q]
            local_paths[p] = chain[::-1]
        paths.append(local_paths)
        internal_sizes.append(float(np.linalg.norm(np.ptp(np.asarray(list(group)), axis=0))))
    edges = []; incidence = defaultdict(list); used = set()
    for i, group in enumerate(groups):
        for p in sorted(group,key=point_key):
            for q in adjacency[p]:
                if q in group or (p, q) in used:
                    continue
                chain = paths[i][p].copy(); before = p; cur = q
                used.add((p, q)); used.add((q, p))
                while cur not in belongs:
                    chain.append(cur)
                    nxt = next(n for n in adjacency[cur] if n != before)
                    used.add((cur, nxt)); used.add((nxt, cur)); before, cur = cur, nxt
                j = belongs[cur]; chain.extend(paths[j][cur][::-1])
                points = np.asarray(chain, int)
                eid = len(edges)
                edges.append({'a': i, 'b': j, 'points': points, 'length': float(arc(points)[-1]),
                              'widths': 2*radius[points[:, 0], points[:, 1]],
                              'contact_trims': (len(paths[i][p])-1, len(paths[j][cur])-1)})
                incidence[i].append((eid, 0)); incidence[j].append((eid, 1))
    return {'nodes': nodes, 'edges': edges, 'incidence': incidence, 'pure_loop': pure_loop,
            'internal_sizes': internal_sizes, 'radius':radius,'suppressed_burr_pixels':len(removed)}


def _arm(graph, port, settings):
    eid, end = port; edge = graph['edges'][eid]
    full = edge['points'][::(-1 if end else 1)]
    full_widths = edge['widths'][::(-1 if end else 1)]
    # The connector inside a bundled contact is a bookkeeping path. Its
    # direction is not the direction of the root entering/leaving the contact.
    trims = edge['contact_trims'][::(-1 if end else 1)]
    stop_index = len(full)-trims[1]
    p = full[trims[0]:stop_index]; widths = full_widths[trims[0]:stop_index]
    if len(p)<2:
        return None
    a = arc(p); width = max(settings.scale, float(np.quantile(widths, .25)))
    stop = min(a[-1], max(7*settings.scale, 3*width))
    start = min(.5*width, .25*stop)
    samples = np.linspace(start, stop, 15)
    q = np.c_[np.interp(samples, a, p[:, 0]), np.interp(samples, a, p[:, 1])]
    delta = q[-1]-q[0]
    if np.linalg.norm(delta) < 1e-8:
        delta = p[-1]-p[0]
    if np.linalg.norm(delta) < 1e-8:
        return None
    _, _, v = np.linalg.svd(q-q.mean(axis=0), full_matrices=False)
    direction = v[0]
    if direction @ delta < 0:
        direction = -direction
    local = widths[(a >= start) & (a <= stop)]
    width = float(np.quantile(local if len(local) else widths, .25))
    return {'direction': direction, 'width': max(width, settings.scale), 'points': p,
            'end': end, 'external_length':float(a[-1]), 'contact_trim':trims[0]}


def _pair_cost(a, b):
    # Arms point OUT of the contact. Straight continuation has opposite vectors.
    angle = math.acos(float(np.clip(-a['direction'] @ b['direction'], -1., 1.)))
    width_change = abs(math.log(a['width']/b['width']))
    return (angle/(math.pi/2))**2 + .55*width_change


def _contact_costs(graph, node, arms):
    ports = [p for p in graph['incidence'][node] if p in arms]
    if len(ports) > 8:
        return {}, True
    costs = {(i, j): _pair_cost(arms[ports[i]], arms[ports[j]])
             for i in range(len(ports)) for j in range(i+1, len(ports))}
    @lru_cache(None)
    def pairings(indices):
        if len(indices) < 2:
            return [(0., ())]
        first, rest = indices[0], indices[1:]; results = []
        for j in rest:
            tail = tuple(k for k in rest if k != j)
            for score, pairs in pairings(tail):
                results.append((costs[min(first,j), max(first,j)] + score, ((first,j),)+pairs))
        return results
    options = []
    if len(ports) % 2:
        for unpaired in range(len(ports)):
            options.extend(pairings(tuple(i for i in range(len(ports)) if i != unpaired)))
    else:
        options = pairings(tuple(range(len(ports))))
    result = {}
    for score, pairs in options:
        for i, j in pairs:
            key = frozenset((ports[i], ports[j]))
            result[key] = min(result.get(key, math.inf), score)
    return result, False


def _select_path_core(skeleton, mask, minimum_pixels=10, settings=RouteSettings(), base_hint=None, image=None):
    skeleton = np.asarray(skeleton, np.uint8)
    up = np.asarray(settings.up_direction, float)
    if up.shape != (2,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8 or settings.scale <= 0:
        raise ValueError('Invalid orientation or scale')
    up = up / np.linalg.norm(up)
    if image is not None and np.asarray(image).shape[:2]!=skeleton.shape:
        raise ValueError('Photograph and mask dimensions differ')
    # Predictions can represent the two margins of a pale root, leaving a
    # narrow enclosed lumen. Fill only bounded, narrow holes for routing.
    # Open gaps and broad biological loops remain untouched.
    mask=np.asarray(mask,np.uint8)
    support=mask.copy();hole_pixels=0
    if settings.lumen_radius>0 and mask.any():
        n,hlabels,hstats,_=cv2.connectedComponentsWithStats(1-(mask>0).astype(np.uint8),8)
        distance=cv2.distanceTransform(1-(mask>0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        maxima=np.zeros(n);np.maximum.at(maxima,hlabels.ravel(),distance.ravel())
        fill_labels=np.zeros(n,bool)
        for label in range(1,n):
            x,y,w,h,area=hstats[label]
            if x==0 or y==0 or x+w==mask.shape[1] or y+h==mask.shape[0]:continue
            if maxima[label]<=settings.lumen_radius*settings.scale:
                fill_labels[label]=True;hole_pixels+=int(area)
        support[fill_labels[hlabels]]=1
        if hole_pixels:skeleton=skeletonize(support>0).astype(np.uint8)
    mask=support
    graph = _graph(skeleton, mask,up,settings.suppress_burrs,settings.zero_boundary_radius,settings.bounded_contacts)
    arms = {port: arm for n in graph['incidence'] for port in graph['incidence'][n]
            if (arm := _arm(graph, port, settings)) is not None}
    # Component reach is evidence of basal connection, never a tip or primary-root definition.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, 8)
    endpoints = [(n, ports[0]) for n, ports in graph['incidence'].items()
                 if len(ports) == 1 and ports[0] in arms]
    flags = []
    if hole_pixels:flags.append('narrow_enclosed_lumen_filled')
    if image is None:
        flags.append('photograph_not_supplied')
    if not endpoints:
        return {'path': [], 'length': None, 'status': 'no_visible_endpoint', 'flags': ['closed_or_unresolved_network'], 'alternatives': []}
    projected = np.asarray(graph['nodes'], float) @ up
    top = float(projected.max()); extent = max(1., float(projected.max()-projected.min()))
    median_width = max(settings.scale, float(np.median([v['width'] for v in arms.values()])))
    bases = [];basal_evidence=[]
    for node, port in endpoints:
        point = graph['nodes'][node]; size = int(stats[labels[point], cv2.CC_STAT_AREA])
        if size < minimum_pixels:
            continue
        if base_hint is None and extent>100 and stats[labels[point],cv2.CC_STAT_HEIGHT]-1<.12*extent:
            continue
        arm = arms[port]
        position = (top-projected[node])/extent
        inward = round(float(-up @ arm['direction']),9)
        # A basal candidate must enter the network, rather than head farther upward.
        if base_hint is None and (position > settings.basal_top_fraction or inward < -.25):
            continue
        score = 12*position + .3*(1-inward) - .4*math.log(max(1.,size)/max(1.,minimum_pixels)) - .65*math.log(arm['width']/median_width)
        colour_support=0.
        if image is not None and np.asarray(image).ndim==3:
            p=arm['points'];a=arc(p)
            sample=p[a<=max(24*settings.scale,6*arm['width'])]
            values=[]
            for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                yy=np.clip(sample[:,0]+dy,0,image.shape[0]-1)
                xx=np.clip(sample[:,1]+dx,0,image.shape[1]-1)
                values.append(np.asarray(image)[yy,xx,:3].astype(float))
            pixel=np.mean(values,axis=0);b,g,r=pixel.T
            evidence=np.clip((2*g-r-b-20)/35,0,1)
            evidence[(g-b)<8]=0;evidence[(g-r)<-8]=0
            colour_support=float(np.mean(evidence)) if len(evidence) else 0.
            score-=3*colour_support
        if base_hint is not None:
            score = float(np.linalg.norm(np.asarray(point)-np.asarray(base_hint)))/max(7*settings.scale, median_width)
        basal_evidence.append({'point':list(map(int,point)),'score':float(score),
                               'photograph_support':colour_support,'entry_width_px':arm['width'],
                               'connected_pixels':size,'role':'mapped_raw_origin' if base_hint is not None else 'basal_candidate'})
        bases.append((round(score,9), node, port))
    if not bases:
        return {'path': [], 'length': None, 'status': 'base_not_supported', 'flags': ['base_not_supported'], 'alternatives': []}
    bases.sort(key=lambda t: t[0])
    best_base_score = bases[0][0]
    bases = bases[:4] if settings.crown_contacts else [b for b in bases if b[0] <= best_base_score + .4][:3]
    if len(bases) > 1:
        flags.append('crown_multiple_basal_hypotheses' if settings.crown_contacts else 'multiple_base_candidates')
    def incoming_history(path_points, fallback):
        outside=path_points[:len(path_points)-fallback['contact_trim']] if fallback['contact_trim'] else path_points
        p=np.asarray(outside[-max(32,int(8*fallback['width'])):],float)[::-1]
        a=arc(p);stop=min(a[-1],max(7*settings.scale,4*fallback['width']))
        first=min(.5*fallback['width'],.25*stop)
        sample=np.linspace(first,stop,16)
        q=np.c_[np.interp(sample,a,p[:,0]),np.interp(sample,a,p[:,1])]
        delta=q[-1]-q[0]
        if np.linalg.norm(delta)<1e-8:
            return fallback
        _,_,v=np.linalg.svd(q-q.mean(axis=0),full_matrices=False)
        direction=v[0]
        if direction@delta<0:direction=-direction
        integer=np.rint(p).astype(int)
        width_values=graph['radius'][integer[:,0],integer[:,1]]*2
        chosen=width_values[(a>=first)&(a<=stop)]
        width=max(settings.scale,float(np.quantile(chosen if len(chosen) else width_values,.25)))
        return {**fallback,'direction':direction,'width':width}

    def trace(base, forced=None):
        _,base_node,port=base
        used=set();points=[];records=[];choices=[];route_flags=[];costs=[];walked=0.
        endpoint_status='budget_stop'
        for _ in range(settings.max_expansions):
            eid,side=port
            if eid in used:
                endpoint_status='graph_block';route_flags.append('loop_or_unresolved_contact');break
            used.add(eid);edge=graph['edges'][eid]
            walked+=edge['length']
            segment=edge['points'][::(-1 if side else 1)]
            points.extend(map(tuple,segment[1:] if points else segment))
            target=edge['a'] if side else edge['b'];incoming=(eid,1-side)
            if len(graph['incidence'][target])==1:
                endpoint_status='visible_endpoint';break
            ports=[p for p in graph['incidence'][target] if p!=incoming and p[0] not in used and p in arms]
            if not ports or incoming not in arms:
                endpoint_status='graph_block';route_flags.append('loop_or_unresolved_contact');break
            history=incoming_history(points,arms[incoming])
            local_arms={p:arms[p] for p in graph['incidence'][target] if p in arms}
            local_arms[incoming]=history
            matching,crowded=_contact_costs(graph,target,local_arms)
            if crowded:route_flags.append('crowded_contact')
            scores=[]
            for candidate in ports:
                single=_pair_cost(history,arms[candidate])
                pair=matching.get(frozenset((incoming,candidate)),single)/max(1,len(local_arms)//2)
                # Terminal burrs on a thick predicted root should not outvote
                # a supported continuation solely because they point straight.
                other=graph['edges'][candidate[0]]
                terminal=other['a'] if candidate[1] else other['b']
                support=arms[candidate]['external_length']/max(history['width'],arms[candidate]['width'])
                spur_penalty=2*max(0.,1-support/2) if len(graph['incidence'][terminal])==1 else 0.
                scores.append((.65*single+.35*pair+spur_penalty,candidate))
            scores.sort(key=lambda item:(round(item[0],9),item[1]))
            selected=scores[0]
            forced_port=forced.get(len(records)) if isinstance(forced,dict) else (forced[1] if forced is not None and forced[0]==len(records) else None)
            if forced_port is not None:
                selected=next((item for item in scores if item[1]==forced_port),selected)
            if len(scores)>1 and scores[1][0]-scores[0][0]<=settings.ambiguity_cost:
                route_flags.append('root_identity_ambiguous')
            for score,candidate in scores[1:]:
                choices.append((len(records),candidate,score-scores[0][0]<=settings.ambiguity_cost))
            if len(local_arms)>=4:route_flags.append('crossing_identity_unverified')
            if graph['internal_sizes'][target]>4*median_width:route_flags.append('extended_contact')
            record={'node':target,'incoming':incoming,'outgoing':selected[1],
                    'continuation_cost':float(selected[0]),
                    'runner_up_gap':float(scores[1][0]-scores[0][0]) if len(scores)>1 else None,
                    'incoming_direction':history['direction'].tolist(),
                    'path_arc_position_px':float(walked),'contact_point_yx':list(map(int,graph['nodes'][target])),
                    'incoming_width_px':history['width'],'outgoing_width_px':arms[selected[1]]['width'],
                    'turn_angle_deg':float(math.degrees(math.acos(float(np.clip(-history['direction']@arms[selected[1]]['direction'],-1,1))))),
                    'contact_span_px':graph['internal_sizes'][target]}
            records.append(record);costs.append(selected[0]);port=selected[1]
        else:
            route_flags.append('search_budget_exhausted')
        if not points:
            return None,[]
        tip=points[-1];h,w=skeleton.shape
        if min(tip[0],tip[1],h-1-tip[0],w-1-tip[1])<=2*settings.scale:
            route_flags.append('visible_tip_at_image_border')
        result={'path':[list(map(int,p)) for p in points],'length':float(arc(points)[-1]),
                'base':list(map(int,points[0])),'tip':list(map(int,points[-1])),
                'contacts':records,'flags':sorted(set(route_flags)),
                'endpoint_status':endpoint_status,
                'cost':float(np.mean(costs)) if costs else 0.,
                'cost_role':'diagnostic only; selected by incoming continuity, not whole-route cost'}
        return result,choices

    first,choices=trace(bases[0])
    if first is None:
        return {'path':[],'length':None,'status':'no_supported_route','flags':flags+['no_supported_route'],'alternatives':[]}
    first['kind']='selected'
    alternatives=[first]
    enumeration=None;all_crown=[]
    if settings.crown_contacts:
        # Branch unconditionally in the first six contacts. Keep the existing
        # selected path. Breadth-first trace budget is explicit and finite.
        pool=[first];queue=deque([(0,{},first,choices)]);trace_count=1;seen={(0,())}
        for bi,base in enumerate(bases[1:],1):
            other,other_choices=trace(base,{})
            trace_count+=1
            if other:other['kind']='crown_candidate';pool.append(other);queue.append((bi,{},other,other_choices))
        while queue and trace_count<settings.crown_trace_budget:
            bi,forced,previous_route,local_choices=queue.popleft()
            last=max(forced,default=-1)
            for depth,port,_ in local_choices:
                if not (last<depth<settings.crown_contacts):continue
                updated={**forced,depth:port};key=(bi,tuple(sorted(updated.items())))
                if key in seen:continue
                seen.add(key);candidate,next_choices=trace(bases[bi],updated);trace_count+=1
                if candidate:
                    candidate['kind']='crown_candidate';pool.append(candidate);queue.append((bi,updated,candidate,next_choices))
                if trace_count>=settings.crown_trace_budget:break
        # Diversity is only for displaying a bounded set for human review.
        # It never replaces the selected route or declares a primary root.
        unique=[]
        from scipy.spatial import cKDTree
        for candidate in pool:
            p=np.asarray(candidate['path'],float);duplicate=False
            for kept in unique:
                if np.linalg.norm(p[-1]-kept['tip'])>18*settings.scale:continue
                q=np.asarray(kept['path'],float)
                d=max(cKDTree(p).query(q)[0].mean(),cKDTree(q).query(p)[0].mean())
                if d<=12*settings.scale:duplicate=True;break
            if not duplicate:unique.append(candidate)
        alternatives=[first];pending=[a for a in unique if a is not first]
        while pending and len(alternatives)<settings.crown_display_candidates:
            chosen=max(range(len(pending)),key=lambda i:round(min(np.linalg.norm(np.asarray(pending[i]['tip'])-a['tip']) for a in alternatives),9))
            alternatives.append(pending.pop(chosen))
        enumeration=dict(contacts=settings.crown_contacts,basal_candidates=len(bases),trace_budget=settings.crown_trace_budget,traces=trace_count,truncated=bool(queue),unique_routes=len(unique),displayed=len(alternatives),display_rule='Selected route plus farthest endpoint diversity; not primary-root scoring',human_selection_required=True)
        all_crown=copy.deepcopy(unique)
    # Alternatives audit uncertain LOCAL decisions; they are not a claim of
    # globally optimised biological identity. The selected route cannot change
    # to a short early lateral because it has fewer contacts later on.
    for choice in ([] if settings.crown_contacts else choices[:max(0,settings.max_candidates-len(bases))]):
        alternative,_=trace(bases[0],choice)
        if alternative and alternative['path']!=first['path']:
            alternative['kind']='uncertain_alternative' if choice[2] else 'rejected_alternative'
            alternatives.append(alternative)
    for base in ([] if settings.crown_contacts else bases[1:]):
        alternative,_=trace(base)
        if alternative:
            alternative['kind']='alternative_base'
            alternatives.append(alternative)
    alternatives=alternatives[:settings.max_candidates]
    flags+=first['flags']
    supported=[a for a in alternatives if a['kind']!='rejected_alternative']
    return {'path':[tuple(p) for p in first['path']],'length':first['length'],
            'status':'candidate_uncertain' if flags else 'candidate','flags':sorted(set(flags)),
            'endpoint_status':first['endpoint_status'],'alternatives':alternatives,
            'length_range':None if settings.crown_contacts else [min(a['length'] for a in supported),max(a['length'] for a in supported)],
            'display_candidate_length_range':[min(a['length'] for a in supported),max(a['length'] for a in supported)] if settings.crown_contacts else None,
            'legacy_uncertainty_fields_comparable':not bool(settings.crown_contacts),
            'all_crown_candidates':all_crown,
            'settings':asdict(settings),'biological_identity_validated':False,
            'basal_candidates':sorted(basal_evidence,key=lambda item:item['score'])[:8],
            'component_label':int(labels[tuple(first['base'])]),
            'search_expansions':sum(len(a['contacts'])+1 for a in alternatives),
            'suppressed_burr_pixels':graph['suppressed_burr_pixels'],
            'filled_lumen_pixels':hole_pixels,
            'candidate_enumeration':enumeration,
            'selection_rule':'local incoming-history continuation with joint contact pairing'}



def select_path(skeleton, mask, minimum_pixels=10, settings=RouteSettings(), base_hint=None, image=None):
    support=(np.asarray(mask)>0).astype(np.uint8)
    if not settings.canonical_support or not support.any():
        return _select_path_core(skeleton,support,minimum_pixels,settings,base_hint,image)
    if np.asarray(skeleton).shape!=support.shape:
        raise ValueError('Skeleton and support dimensions differ')
    if image is not None and np.asarray(image).shape[:2]!=support.shape:
        raise ValueError('Photograph and support dimensions differ')
    yy,xx=np.where(support);y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
    border=max(8,int(math.ceil(16*settings.scale)))
    local=np.pad(support[y0:y1,x0:x1],border)
    if settings.rebuild_skeleton:
        # Skeletonization is performed in identical root-relative surroundings,
        # regardless of blank padding or contact with the input image border.
        sk=skeletonize(local>0).astype(np.uint8)
    else:sk=np.pad(np.asarray(skeleton,np.uint8)[y0:y1,x0:x1],border)
    photo=None if image is None else cv2.copyMakeBorder(np.asarray(image)[y0:y1,x0:x1],border,border,border,border,cv2.BORDER_CONSTANT,value=0)
    offset=np.array([y0-border,x0-border])
    hint=None if base_hint is None else np.asarray(base_hint)-offset
    r=_select_path_core(sk,local,minimum_pixels,settings,hint,photo)
    if r.get('path'):r['path']=[tuple(map(int,np.asarray(p)+offset)) for p in r['path']]
    for a in r.get('alternatives',[])+r.get('all_crown_candidates',[]):
        a['path']=[list(map(int,np.asarray(p)+offset)) for p in a['path']]
        for key in ['base','tip']:
            if key in a:a[key]=list(map(int,np.asarray(a[key])+offset))
        for contact in a.get('contacts',[]):
            if 'contact_point_yx' in contact:contact['contact_point_yx']=list(map(int,np.asarray(contact['contact_point_yx'])+offset))
    for candidate in r.get('basal_candidates',[]):candidate['point']=list(map(int,np.asarray(candidate['point'])+offset))
    r['canonical_support_bbox_yxyx']=[y0,x0,y1,x1];r['canonical_border']=border
    r['algorithm_version']=VERSION
    return r


def draw_path(image, result):
    view = np.asarray(image).copy()
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    p = np.asarray(result.get('path', []), int)
    if len(p):
        cv2.polylines(view, [p[:, ::-1].astype(np.int32)], False, (0, 165, 255), 2)
        cv2.circle(view, tuple(p[0, ::-1]), 5, (0, 200, 0), -1)
        cv2.circle(view, tuple(p[-1, ::-1]), 5, (0, 0, 230), -1)
    return view
