import argparse
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

# =================================================================
# 1. Configuration & Setup
# =================================================================
# Define workspace directories (modify these paths according to your environment)
WORKSPACE_DIR = "./runs/manuscript/phenotypes"
GT_DIR = "./runs/manuscript/ground_truth"
PRED_DIR = "./runs/manuscript/predictions"

def robust_imread(path):
    """Safely read an image from a path, returning a grayscale numpy array."""
    if not os.path.exists(path): 
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def find_shortest_path(img, start, end):
    """
    Find the shortest path between two points in a skeletonized binary image 
    using unweighted Breadth-First Search (BFS) in an 8-neighborhood.
    """
    queue = [start]
    visited = {start: None}
    h, w = img.shape
    found = False
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while queue:
        curr = queue.pop(0)
        if curr == end:
            found = True
            break
        y, x = curr
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and img[ny, nx] == 1:
                if (ny, nx) not in visited:
                    visited[(ny, nx)] = curr
                    queue.append((ny, nx))
                    
    if not found: 
        return None
    
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = visited[curr]
    return path[::-1]

def prune_skeleton(skel, num_iter=15):
    """
    Iteratively remove skeletal endpoints to eliminate pseudo-lateral 
    root artifacts caused by edge jaggedness.
    """
    skel_pruned = skel.copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(num_iter):
        neighbor_count = cv2.filter2D(skel_pruned, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (skel_pruned == 1) & (neighbor_count == 1)
        skel_pruned[endpoints] = 0
    return skel_pruned

# =================================================================
# 2. Core Engine: Feature-Decoupled Phenotype Extraction
# =================================================================
def extract_phenotypes_decoupled(mask):
    """
    Extract root phenotypes using a dual-channel strategy:
    - Channel A: High-fidelity for primary root length.
    - Channel B: Topology-robust for lateral root count, junctions, and angle.
    """
    binary = (mask > 127).astype(np.uint8)
    if np.sum(binary) == 0: 
        return 0, 0, 0, 0

    # --- Channel A: High-fidelity channel (Primary Root Length) ---
    raw_skel = skeletonize(binary).astype(np.uint8)
    num_labels_raw, labels_raw = cv2.connectedComponents(raw_skel)
    
    primary_length = 0
    if num_labels_raw > 1:
        max_span_raw = 0
        main_label_raw = 1
        for i in range(1, num_labels_raw):
            pts = np.argwhere(labels_raw == i)
            span = pts[:, 0].max() - pts[:, 0].min()
            if span > max_span_raw:
                max_span_raw = span
                main_label_raw = i
                
        main_skel_raw = (labels_raw == main_label_raw).astype(np.uint8)
        main_pts_raw = np.argwhere(main_skel_raw > 0)
        
        if len(main_pts_raw) >= 10:
            top_pt_raw = tuple(main_pts_raw[np.argmin(main_pts_raw[:, 0])])
            bot_pt_raw = tuple(main_pts_raw[np.argmax(main_pts_raw[:, 0])])
            
            primary_path_raw = find_shortest_path(main_skel_raw, top_pt_raw, bot_pt_raw)
            if primary_path_raw:
                primary_length = len(primary_path_raw) * 1.12 
            else:
                primary_length = max_span_raw * 1.12

    # --- Channel B: Topology-robust channel (Lateral Traits) ---
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smooth_bin = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel)
    smooth_skel = skeletonize(smooth_bin).astype(np.uint8)
    pruned_skel = prune_skeleton(smooth_skel, num_iter=15)

    if np.sum(pruned_skel) == 0:
        return primary_length, 0, 0, 0

    # Extract branch points (junctions)
    kernel_cross = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(pruned_skel, -1, kernel_cross, borderType=cv2.BORDER_CONSTANT) * pruned_skel
    junction_pixels = (neighbor_count >= 3).astype(np.uint8)
    true_junction_count, _ = cv2.connectedComponents(junction_pixels)
    true_junction_count = max(0, true_junction_count - 1)

    # Identify primary root in the pruned skeleton
    num_labels_p, labels_p = cv2.connectedComponents(pruned_skel)
    pruned_primary_mask = np.zeros_like(pruned_skel)
    pruned_primary_path = None
    
    if num_labels_p > 1:
        max_span_p = 0
        main_label_p = 1
        for i in range(1, num_labels_p):
            pts = np.argwhere(labels_p == i)
            span = pts[:, 0].max() - pts[:, 0].min()
            if span > max_span_p:
                max_span_p = span
                main_label_p = i
                
        main_skel_p = (labels_p == main_label_p).astype(np.uint8)
        main_pts_p = np.argwhere(main_skel_p > 0)
        
        if len(main_pts_p) >= 10:
            top_pt_p = tuple(main_pts_p[np.argmin(main_pts_p[:, 0])])
            bot_pt_p = tuple(main_pts_p[np.argmax(main_pts_p[:, 0])])
            pruned_primary_path = find_shortest_path(main_skel_p, top_pt_p, bot_pt_p)
            
            if pruned_primary_path:
                for y, x in pruned_primary_path: 
                    pruned_primary_mask[y, x] = 1
            else:
                pruned_primary_mask = main_skel_p

    # Extract lateral root count
    lateral_skel = ((pruned_skel == 1) & (pruned_primary_mask == 0)).astype(np.uint8)
    lateral_skel[junction_pixels == 1] = 0
    
    num_lat, labeled_lat = cv2.connectedComponents(lateral_skel)
    min_lateral_pixels = 10 
    true_lateral_count = 0
    valid_lateral_components = []
    
    for i in range(1, num_lat):
        comp_pts = np.argwhere(labeled_lat == i)
        if len(comp_pts) >= min_lateral_pixels:
            true_lateral_count += 1
            valid_lateral_components.append(comp_pts)

    # Extract branching angles via local vector differencing
    angles = []
    if pruned_primary_path and len(valid_lateral_components) > 0:
        path_arr = np.array(pruned_primary_path)
        for comp in valid_lateral_components:
            dists_to_path = np.linalg.norm(comp[:, None, :] - path_arr[None, :, :], axis=2)
            min_idx = np.unravel_index(np.argmin(dists_to_path), dists_to_path.shape)
            base_pt = comp[min_idx[0]]
            closest_path_idx = min_idx[1]
            
            dists_from_base = np.linalg.norm(comp - base_pt, axis=1)
            target_lat_indices = np.where((dists_from_base >= 8) & (dists_from_base <= 15))[0]
            
            if len(target_lat_indices) > 0:
                lat_vec = comp[target_lat_indices[0]] - base_pt
            else:
                lat_vec = comp[-1] - base_pt
                
            idx_down = min(len(pruned_primary_path) - 1, closest_path_idx + 12)
            prim_vec = path_arr[idx_down] - path_arr[closest_path_idx]
            
            norm_p = np.linalg.norm(prim_vec)
            norm_l = np.linalg.norm(lat_vec)
            
            if norm_p > 0 and norm_l > 0:
                dot = np.dot(prim_vec, lat_vec) / (norm_p * norm_l)
                angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
                if angle > 90: 
                    angle = 180 - angle
                angles.append(angle)
                
    mean_angle = np.mean(angles) if angles else 60.0
    return primary_length, true_lateral_count, true_junction_count, mean_angle

# =================================================================
# 3. Evaluation and Visualization Pipeline
# =================================================================
def main():
    import matplotlib.pyplot as plt
    from scipy import stats

    # Academic plotting configuration (sans-serif fonts)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    global WORKSPACE_DIR, GT_DIR, PRED_DIR
    parser = argparse.ArgumentParser(description="Extract and compare the four mask-derived descriptors.")
    parser.add_argument("--ground-truth-dir", default=GT_DIR)
    parser.add_argument("--prediction-dir", default=PRED_DIR)
    parser.add_argument("--output-dir", default=WORKSPACE_DIR)
    args = parser.parse_args()
    GT_DIR = os.path.abspath(args.ground_truth_dir)
    PRED_DIR = os.path.abspath(args.prediction_dir)
    WORKSPACE_DIR = os.path.abspath(args.output_dir)

    print("Initializing phenotype extraction and evaluating metrics...")
    
    traits_gt = {"length": [], "count": [], "junctions": [], "angle": []}
    traits_pred = {"length": [], "count": [], "junctions": [], "angle": []}
    
    # Process actual data. Missing inputs are treated as an error so that a
    # manuscript rerun can never silently switch to synthetic demonstration data.
    if os.path.exists(PRED_DIR) and os.path.exists(GT_DIR):
        files = [f for f in os.listdir(PRED_DIR) if f.endswith('.tif')]
        for name in files:
            gt_path = os.path.join(GT_DIR, name.replace(".tif", "_mask.png"))
            pred_path = os.path.join(PRED_DIR, name)
            if not os.path.exists(gt_path): 
                continue
            
            gt_mask = robust_imread(gt_path)
            pred_mask = robust_imread(pred_path)
            if gt_mask is None or pred_mask is None: 
                continue
            
            gl, gc, gj, ga = extract_phenotypes_decoupled(gt_mask)
            pl, pc, pj, pa = extract_phenotypes_decoupled(pred_mask)
            
            traits_gt["length"].append(gl); traits_pred["length"].append(pl)
            traits_gt["count"].append(gc);  traits_pred["count"].append(pc)
            traits_gt["junctions"].append(gj); traits_pred["junctions"].append(pj)
            traits_gt["angle"].append(ga);  traits_pred["angle"].append(pa)
    else:
        raise FileNotFoundError(f"Expected ground-truth and prediction directories: {GT_DIR}, {PRED_DIR}")

    # Global plot typography adjustments
    plt.rcParams.update({
        'font.size': 20,              
        'axes.labelsize': 22,         
        'xtick.labelsize': 18,        
        'ytick.labelsize': 18,        
        'axes.linewidth': 2.0         
    })

    keys = ["length", "count", "junctions", "angle"]
    file_names = [
        "Fig_Primary_Length.png", 
        "Fig_Lateral_Count.png", 
        "Fig_Branch_Points.png", 
        "Fig_Branching_Angle.png"
    ]
    
    x_labels = ["Auto primary root length", "Auto lateral root number", "Auto branch points", "Auto root branching angle"]
    y_labels = ["Manual primary root length", "Manual lateral root number", "Manual branch points", "Manual root branching angle"]
    
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    
    for i, key in enumerate(keys):
        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=600)
        
        x = np.array(traits_pred[key])
        y = np.array(traits_gt[key])
        
        # Statistical calculations
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        y_safe = np.where(y == 0, 1e-5, y)
        mape = np.mean(np.abs((y - x) / y_safe)) * 100
        rmse = np.sqrt(np.mean((y - x) ** 2))
        
        # Axis limits based on data range
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        padding = (max_val - min_val) * 0.05
        if padding == 0: 
            padding = 1.0
        
        lim_min = min_val - padding
        lim_max = max_val + padding
        
        # Plot regression line
        line_x = np.linspace(lim_min, lim_max, 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color='#e31a1c', linewidth=2.5, zorder=2)
        
        # Plot scatter points
        ax.scatter(x, y, facecolors='none', edgecolors='#204a87', s=80, linewidths=2.0, zorder=3)
        
        # Clean formatting for publication
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out', width=2.0, length=6)
        
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])
        
        # Metrics annotation
        sign = "+" if intercept >= 0 else "-"
        text_str = f"y = {slope:.3f} * x {sign} {abs(intercept):.2f}\n$R^2$ = {r_squared:.3f}\nMAPE = {mape:.2f}%\nRMSE = {rmse:.2f}"
        
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='left')
        
        ax.grid(False)
        plt.tight_layout()
        
        output_png = os.path.join(WORKSPACE_DIR, file_names[i])
        plt.savefig(output_png, dpi=600, bbox_inches='tight', transparent=False, facecolor='w')
        plt.close(fig)
        
        print(f"Generated: {file_names[i]}")

    print(f"\nAll figures successfully saved to: {WORKSPACE_DIR}")


if __name__ == "__main__":
    main()
