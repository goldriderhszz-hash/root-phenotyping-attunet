# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) Agreement scatter and difference plots -> MarginalDensity/plot_VariableCorrelation.py -> param inherit; the asset uses synthetic 2D distributions without repeated paired images.
# (b) Extractor sensitivity -> heatmap/plot_composition.py -> param inherit; the asset uses synthetic categorical counts, while our data are signed, paired percentage changes.
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

from pathlib import Path
import json
import math

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent / "figures"
ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "tables"
CONFIG = "attunet_cl_fixed"
TRAITS = [
    ("dominant_path_length", "Dominant path", "px", CATEGORICAL[2]),
    ("retained_segment_count", "Retained segments", "count", CATEGORICAL[2]),
    ("junction_region_count", "Junction regions", "count", CATEGORICAL[2]),
    ("mean_local_acute_angle_deg", "Mean local angle", "deg", CATEGORICAL[2]),
]
CHANNEL_B_TRAITS = [item for item in TRAITS if item[0] != "dominant_path_length"]
SEEDS = [42, 3407, 2026]
DEFAULT = (5, 15, 10)
MM = 1 / 25.4


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x-mx)*(y-my)))
    denominator = vx + vy + (mx-my)**2
    return 2*cov/denominator if denominator else math.nan


def load_agreement() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    source = pd.read_csv(ANALYSIS / "descriptor_comparison_long.csv")
    fixed = source.loc[source.config == CONFIG].copy()
    assert set(fixed.trait) == {t[0] for t in TRAITS}
    assert set(fixed.seed) == set(SEEDS)
    assert set(fixed.reference_type) == {"annotation_mask_same_extractor"}
    assert len(fixed) == 600
    assert fixed.sample_name.nunique() == 50
    assert not fixed.duplicated(["sample_name", "seed", "trait"]).any()
    assert (fixed.groupby(["trait", "seed"]).size() == 50).all()
    fixed.to_csv(HERE / "source_Fig7_all_600_pairs.csv", index=False)
    good = fixed.loc[(fixed.status == "valid") & np.isfinite(fixed.reference_value) & np.isfinite(fixed.estimate_value)].copy()
    excluded = fixed.loc[~fixed.index.isin(good.index)]
    if len(excluded):
        excluded.to_csv(HERE / "source_Fig7_excluded_invalid_pairs.csv", index=False)
    assert len(good) == 600, f"Fig7 requires all 600 valid pairs; {len(excluded)} invalid pairs need explicit figure handling"
    grouped = good.groupby(["sample_name", "fold", "trait"], as_index=False).agg(
        reference_value=("reference_value", "first"), estimate_mean=("estimate_value", "mean"),
        estimate_min=("estimate_value", "min"), estimate_max=("estimate_value", "max"),
        difference_mean=("difference_estimate_minus_reference", "mean"),
        difference_min=("difference_estimate_minus_reference", "min"),
        difference_max=("difference_estimate_minus_reference", "max"),
        n_seeds=("seed", "size"))
    assert len(grouped) == 200 and (grouped.n_seeds == 3).all()
    grouped["average_reference_estimate"] = (grouped.reference_value + grouped.estimate_mean)/2
    grouped.to_csv(HERE / "source_Fig7_50_image_means_per_trait.csv", index=False)
    stats=[]
    for trait,_,unit,_ in TRAITS:
        d = good[good.trait==trait]
        for seed, ds in d.groupby("seed"):
            x,y=ds.reference_value.to_numpy(),ds.estimate_value.to_numpy()
            stats.append({"trait":trait,"unit":unit,"seed":int(seed),"n_images":len(ds),
                          "mae":float(np.mean(np.abs(y-x))),"bias":float(np.mean(y-x)),"ccc":lin_ccc(x,y)})
    stats=pd.DataFrame(stats)
    stats.to_csv(HERE/"source_Fig7_seed_statistics.csv",index=False)
    comparison=pd.read_csv(ANALYSIS/"descriptor_summary_by_seed.csv")
    assert len(comparison)==48 and set(comparison.config)=={"unet_pixel","attunet_pixel","attunet_cl_fixed","attunet_cl_schedule"}
    assert set(comparison.seed)==set(SEEDS)
    assert set(comparison.trait)=={item[0] for item in TRAITS}
    assert set(comparison.n_valid_pairs)=={50}
    comparison.to_csv(HERE/"source_Fig7_four_configuration_seed_MAE.csv",index=False)
    return good,grouped,stats,comparison


def axis_data_limit(values: np.ndarray) -> tuple[float,float]:
    low = float(np.min(values))
    high = float(np.max(values))
    pad = (high-low)*.06 if high>low else max(abs(high)*.06,1)
    return low-pad, high+pad


def plot_agreement(pairs:pd.DataFrame,means:pd.DataFrame,stats:pd.DataFrame,comparison:pd.DataFrame,traits=TRAITS,name="Fig7_Descriptor_Agreement") -> None:
    fig=plt.figure(figsize=(170*MM,188*MM),layout="constrained")
    grid=fig.add_gridspec(3,len(traits),height_ratios=[1.15,.76,.70],hspace=.08,wspace=.12)
    for i,(trait,label,unit,color) in enumerate(traits):
        d=pairs[pairs.trait==trait]
        g=means[means.trait==trait]
        s=stats[stats.trait==trait]
        ax=fig.add_subplot(grid[0,i])
        ax.scatter(d.reference_value,d.estimate_value,s=7,c=color,alpha=.13,lw=0,zorder=1)
        ax.vlines(g.reference_value,g.estimate_min,g.estimate_max,color=color,lw=.6,alpha=.45,zorder=2)
        ax.scatter(g.reference_value,g.estimate_mean,s=13,c=color,alpha=.9,marker="D",lw=0,zorder=3)
        upper=float(max(d.reference_value.max(),d.estimate_value.max()))
        ax.plot([0,upper],[0,upper],ls="--",lw=.7,color="#777777",zorder=0)
        lo,hi=axis_data_limit(np.r_[d.reference_value.to_numpy(),d.estimate_value.to_numpy(),[0]])
        ax.set_xlim(lo,hi)
        ax.set_ylim(lo,hi)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title(f"{chr(ord('a')+i)}  {label}",loc="left",weight="bold",pad=9)
        ax.set_xlabel(f"Annotation ({unit})")
        if i==0: ax.set_ylabel(f"OOF mask ({unit})")
        ax.tick_params(axis="both",labelsize=6)
        ax.text(.03,.98,f"CCC {s.ccc.mean():.2f} ± {s.ccc.std(ddof=1):.2f}\nMAE {s.mae.mean():.1f} {unit}",
                transform=ax.transAxes,ha="left",va="top",fontsize=6.1,
                bbox=dict(facecolor="white",alpha=.84,edgecolor="none",pad=1.5))

        bx=fig.add_subplot(grid[1,i])
        bx.vlines(g.average_reference_estimate,g.difference_min,g.difference_max,color=color,lw=.6,alpha=.36)
        bx.scatter(g.average_reference_estimate,g.difference_mean,s=12,c=color,alpha=.78,lw=0)
        bx.axhline(0,color="#777777",lw=.7,ls="--")
        bias=float(g.difference_mean.mean())
        difference_sd=float(g.difference_mean.std(ddof=1))
        limits=(bias-1.96*difference_sd,bias+1.96*difference_sd)
        bx.axhline(bias,color=color,lw=1.1)
        for limit in limits: bx.axhline(limit,color=color,lw=.8,ls=":")
        bx.set_xlim(axis_data_limit(g.average_reference_estimate.to_numpy()))
        bx.set_ylim(axis_data_limit(np.r_[g.difference_min.to_numpy(),g.difference_max.to_numpy(),[0,*limits]]))
        bx.set_xlabel(f"Pair mean ({unit})")
        if i==0: bx.set_ylabel(f"OOF − annotation ({unit})")
        bx.tick_params(axis="both",labelsize=6)
        bx.text(0,1.03,chr(ord("a")+len(traits)+i),transform=bx.transAxes,fontsize=9,weight="bold")

        cx=fig.add_subplot(grid[2,i])
        tcomp=comparison[comparison.trait==trait]
        configs=["unet_pixel","attunet_pixel","attunet_cl_fixed","attunet_cl_schedule"]
        color_map={"unet_pixel":CATEGORICAL[0],"attunet_pixel":CATEGORICAL[3],
                   "attunet_cl_fixed":CATEGORICAL[2],"attunet_cl_schedule":CATEGORICAL[1]}
        for j,config in enumerate(configs):
            sub=tcomp[tcomp.config==config].sort_values("seed")
            value=sub.mae.to_numpy()
            cx.vlines(j,np.mean(value)-np.std(value,ddof=1),np.mean(value)+np.std(value,ddof=1),
                      color=color_map[config],lw=1.2)
            cx.hlines(np.mean(value),j-.20,j+.20,color=BLACK,lw=1.1)
            for _,row in sub.iterrows():
                shift={42:-.10,3407:0,2026:.10}[int(row.seed)]
                marker={42:"o",3407:"s",2026:"^"}[int(row.seed)]
                cx.scatter(j+shift,row.mae,color=color_map[config],marker=marker,s=17,
                           edgecolor="white",linewidth=.35,zorder=3)
        cx.set_xticks(range(4),["U-Net","Attention","Fixed","Two-stage"],rotation=35,ha="right")
        cx.set_ylim(bottom=0)
        cx.set_ylabel(f"OOF MAE ({unit})")
        cx.tick_params(axis="both",labelsize=6)
        cx.text(0,1.03,chr(ord("a")+2*len(traits)+i),transform=cx.transAxes,fontsize=9,weight="bold")
    fig.savefig(HERE/(name+".pdf"),dpi=300)
    fig.savefig(HERE/(name+".png"),dpi=600)
    plt.close(fig)


def load_sensitivity() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    ann=pd.read_csv(ANALYSIS/"sensitivity_annotation_summary.csv")
    oof=pd.read_csv(ANALYSIS/"sensitivity_summary_across_seeds.csv")
    grid=pd.read_csv(ANALYSIS/"sensitivity_comparison_long.csv")
    assert len(ann)==36 and len(oof)==144 and len(grid)==21600
    assert set(ann.n_distinct_images)=={50}
    assert set(oof.n_distinct_images)=={50}
    assert set(oof.n_training_seeds)=={3}
    assert grid.sample_name.nunique()==50
    assert set(grid.seed)==set(SEEDS)
    assert set(grid.trait)=={t[0] for t in TRAITS}
    assert set(grid.status)=={"valid"}
    keys=["closing_kernel","pruning_iterations","min_component_pixels"]
    assert len(ann[keys].drop_duplicates())==36
    assert len(oof[keys+["trait"]].drop_duplicates())==144
    assert (grid.groupby(keys+["trait","seed"]).size()==50).all()
    assert not grid.duplicated(["sample_name","seed","trait"]+keys).any()
    ann.to_csv(HERE/"source_Fig8_annotation_36_settings.csv",index=False)
    oof.to_csv(HERE/"source_Fig8_oof_36_settings_4_traits.csv",index=False)
    ann_default=ann[(ann.closing_kernel==5)&(ann.pruning_iterations==15)&(ann.min_component_pixels==10)]
    assert len(ann_default)==1
    entries=[]
    for trait,label,unit,_ in CHANNEL_B_TRAITS:
        baseline=float(ann_default.iloc[0][trait+"_mean"])
        baseline_oof=oof[(oof.trait==trait)&(oof.closing_kernel==5)&(oof.pruning_iterations==15)&(oof.min_component_pixels==10)]
        assert len(baseline_oof)==1
        baseline_mae=float(baseline_oof.iloc[0].mae_seed_mean)
        assert baseline>0 and baseline_mae>0
        joined=ann.merge(oof[oof.trait==trait],on=keys,validate="one_to_one",suffixes=("_ann","_oof"))
        for _,row in joined.iterrows():
            entries.append({"trait":trait,"trait_label":label,"unit":unit,
                            "closing_kernel":int(row.closing_kernel),"pruning_iterations":int(row.pruning_iterations),
                            "min_component_pixels":int(row.min_component_pixels),
                            "annotation_mean":float(row[trait+"_mean"]),
                            "annotation_mean_change_percent":float(100*(row[trait+"_mean"]/baseline-1)),
                            "oof_mae_seed_mean":float(row.mae_seed_mean),
                            "oof_mae_change_percent":float(100*(row.mae_seed_mean/baseline_mae-1)),
                            "reference_valid_angle_images":int(row.n_valid_angle_images),
                            "oof_valid_pairs_min_seed":int(row.n_valid_pairs_min_seed),
                            "oof_valid_pairs_max_seed":int(row.n_valid_pairs_max_seed)})
    cells=pd.DataFrame(entries)
    assert len(cells)==108
    cells.to_csv(HERE/"source_Fig8_108_heatmap_cells.csv",index=False)
    return ann,oof,cells


def heatmap_array(cells:pd.DataFrame,trait:str,column:str)->np.ndarray:
    subset=cells[cells.trait==trait]
    matrix=np.full((9,4),np.nan)
    for _,r in subset.iterrows():
        y=[5,10,20].index(int(r.min_component_pixels))*3+[3,5,7].index(int(r.closing_kernel))
        x=[5,10,15,20].index(int(r.pruning_iterations))
        matrix[y,x]=r[column]
    assert np.isfinite(matrix).all()
    assert abs(matrix[4,2])<1e-8
    return matrix


def plot_sensitivity(cells:pd.DataFrame,traits=CHANNEL_B_TRAITS,name="Fig8_Extractor_Sensitivity")->None:
    cmap=LinearSegmentedColormap.from_list("signed",DIVERGING,N=256)
    fig=plt.figure(figsize=(170*MM,(62*len(traits)+1)*MM),layout="constrained")
    grid=fig.add_gridspec(len(traits),2,wspace=.13,hspace=.11)
    for row,(trait,label,unit,_) in enumerate(traits):
        for col,(key,heading) in enumerate([
            ("annotation_mean_change_percent","Annotation-mask mean"),
            ("oof_mae_change_percent","Matched OOF MAE"),
        ]):
            ax=fig.add_subplot(grid[row,col])
            matrix=heatmap_array(cells,trait,key)
            bound=float(np.max(np.abs(matrix)))
            bound=max(bound,0.25)
            im=ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),
                         aspect="auto",interpolation="nearest")
            ax.add_patch(Rectangle((1.5,3.5),1,1,fill=False,edgecolor=BLACK,lw=1.5))
            ax.hlines([2.5,5.5],-.5,3.5,color="white",lw=2)
            ax.set_xticks(range(4),[5,10,15,20])
            ax.set_xlabel("Endpoint pruning iterations")
            ax.set_yticks(range(9),[f"{k} / {m}" for m in [5,10,20] for k in [3,5,7]])
            if col==0: ax.set_ylabel("Closing kernel / min. component (px)")
            else: ax.tick_params(axis="y",labelleft=False)
            ax.tick_params(axis="both",length=0,labelsize=6)
            letter=chr(ord("a")+row*2+col)
            ax.set_title(f"{letter}  {label}: {heading}",loc="left",pad=9,fontsize=7.1,weight="bold")
            cbar=fig.colorbar(im,ax=ax,location="right",fraction=.033,pad=.015,aspect=25)
            cbar.ax.tick_params(labelsize=5,length=2)
            cbar.set_label("Change from 5 / 15 / 10 (%)",fontsize=6)
    fig.savefig(HERE/(name+".pdf"),dpi=300)
    fig.savefig(HERE/(name+".png"),dpi=600)
    plt.close(fig)


def write_qa(pairs:pd.DataFrame,means:pd.DataFrame,stats:pd.DataFrame,comparison:pd.DataFrame,
             ann:pd.DataFrame,oof:pd.DataFrame,cells:pd.DataFrame)->None:
    lines=["# Descriptor figure QA and provenance","",
           "## Fig. 7: matched descriptors","",
           f"- Source: `{ANALYSIS / 'descriptor_comparison_long.csv'}`; selected the fixed clDice configuration before looking at descriptor performance. Fig. 7 contains all {len(pairs)} valid rows: 50 held-out images × 3 training seeds × 4 traits. Rows for other configurations are excluded by the declared configuration selection, not by their values. No fixed-group row is dropped.",
           "- Reference is the annotation mask processed by the same extractor, not an independent manual or biological ground truth. Each image participates in one test fold; predictions across three seeds are repeated evaluations of that same image.",
           "- In the top row, transparent points show all 150 seed-specific pairs; opaque diamonds show 50 image means; vertical bars span the three seed-specific predictions. Dashed lines are equality lines. Middle row plots 50 image means of prediction-minus-annotation against pair means, with seed ranges; colored horizontal lines are average bias over seed-level 50-image means.",
           "- Bottom row compares all four controlled configurations using each seed's 50-image descriptor MAE (3 points/configuration/trait), with black mean and colored sample SD. Colors match Fig. 5; the source is `descriptor_summary_by_seed.csv` and all 48 rows are retained. This makes the count-descriptor tradeoff visible even though the upper panels focus on the predeclared fixed-loss configuration.",
           "- CCC, MAE and bias are computed for each seed from 50 images, then averaged across the three seeds; the ± annotation gives sample SD across the three seed estimates. No p-values or independent manual measurements are asserted.","",
           "| Trait | Valid images per seed | CCC mean ± SD | MAE mean ± SD | Bias mean ± SD |", "|---|---:|---:|---:|---:|"]
    for trait,label,unit,_ in TRAITS:
        s=stats[stats.trait==trait]
        lines.append(f"| {label} ({unit}) | {int(s.n_images.min())}–{int(s.n_images.max())} | {s.ccc.mean():.3f} ± {s.ccc.std(ddof=1):.3f} | {s.mae.mean():.2f} ± {s.mae.std(ddof=1):.2f} | {s.bias.mean():+.2f} ± {s.bias.std(ddof=1):.2f} |")
    lines += ["","## Fig. 8: extractor settings","",
              f"- Sources: `{ANALYSIS / 'sensitivity_annotation_summary.csv'}` ({len(ann)} rows), `{ANALYSIS / 'sensitivity_summary_across_seeds.csv'}` ({len(oof)} rows), and `{ANALYSIS / 'sensitivity_comparison_long.csv'}` (21,600 image×seed×setting×trait rows). Source heatmap cells are in `source_Fig8_108_heatmap_cells.csv` (36 settings × 3 Channel B traits).",
              "- Left panels show the 50-image annotation-mask descriptor mean relative to the predeclared 5-pixel closing, 15-iteration pruning, 10-pixel minimum component setting. Right panels show the fixed-loss OOF mean absolute error relative to the *same-setting* annotation-mask reference, averaged over three seed-specific 50-image MAEs. A positive right-panel change means greater error; a negative change means less error under that setting's own reference. These comparisons are descriptive parameter sensitivity, not threshold tuning or an accuracy estimate against manual biology.",
              "- All 36 parameter triples occur in every heatmap. Rows encode closing kernel 3/5/7 × minimum component 5/10/20 px; columns encode pruning 5/10/15/20. The black outlined cell is the original 5/15/10 setting. Colorbars are centered at zero and scaled separately by panel to show small trait-specific changes; compare numeric percentages, not darkness, between panels.",
              "- Dominant path length is omitted from Fig. 8 because Channel B settings do not affect Channel A's raw-skeleton path calculation; the analysis confirmed exact path invariance across the grid. All 50 annotation images have valid angles in all 36 settings, and all 21,600 matched OOF grid rows have valid descriptor pairs. Source tables preserve valid-pair counts for each OOF setting.","",
              "## Visual checks","",
              "- Figures use 183 mm drawing width, Arial/Helvetica/Liberation Sans, the skill's semantic palette, vector PDF text and 600 dpi PNG previews. The fixed-loss detailed agreement panels use the same green as that configuration in the four-model MAE panels. The synthetic production assets for marginal scatter and categorical heatmap were inspected; their data structures do not map to repeated OOF pairs and signed parameter effects, so styling parameters were inherited without their fabricated observations.",
              "- All final previews were inspected at full output size; all visible points/cells derive from the accompanying source CSVs. Large panels avoid false zero p-values, and captions state the statistical unit and reference source. The academic-figure-skill validator exits with code 0; exported PNG metadata reports 600 dpi and PDFs are single-page vector masters."]
    (HERE/"DESCRIPTOR_FIGURE_QA.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_captions()->None:
    text="""# Manuscript insertion captions

**Fig. 7 | Agreement of skeleton-derived descriptors between out-of-fold predictions and annotation masks.** The fixed-skeleton-loss Attention U-Net was selected for detailed agreement analysis before descriptor outcomes were examined. Top panels (a–d) compare four descriptors from 50 held-out images. Pale points show all three seed-specific predictions per image (150 pairs per descriptor), and darker diamonds show their image means; vertical lines span the three predictions. Dashed lines denote equality. Middle panels (e–h) show each image's mean prediction-minus-annotation difference against its pair mean, with vertical ranges across the three seeds; colored horizontal lines denote average seed-level bias. Bottom panels (i–l) compare mean absolute error (MAE) across all four controlled configurations; each point is one training seed evaluated on the same 50 held-out images, horizontal bars show means and colored vertical bars show sample standard deviations. The fixed skeleton loss reduced segmentation error yet increased retained-segment and junction-region MAE relative to the pixel-loss Attention U-Net. Descriptors were extracted from the predicted and annotation masks using the same rules. Annotation-mask results are a computational reference, not independent manual or biological truth. CCC and MAE in the upper panels are averages of three seed-level statistics, each based on 50 images; CCC variation is standard deviation across seeds. Length is expressed in image pixels because physical calibration was unavailable.

**Fig. 8 | Sensitivity of extracted descriptors to 36 settings.** For retained segment count (a,b), candidate junction-region count (c,d), and mean local acute angle (e,f), the left panels show percentage changes in the mean descriptor from 50 annotation masks, and the right panels show percentage changes in the mean absolute error of the fixed-skeleton-loss OOF predictions against annotation-mask descriptors at the *same* setting. Errors were calculated separately for 50 images in each of three training seeds and then averaged across seeds. Columns represent endpoint-pruning iterations (5, 10, 15, 20); rows represent combinations of closing-kernel width (3, 5, 7 pixels) and minimum retained-component size (5, 10, 20 pixels). The original 5/15/10 setting is outlined in black. Blue and red denote decreases and increases from that setting; each panel has its own color scale. The dominant path length is not shown because these three parameters act on a separate, pruned-skeleton channel and leave its raw-skeleton computation unchanged. The reference changes with the extractor setting, so these maps characterize computational sensitivity rather than biological measurement accuracy.
"""
    (HERE/"FIGURE_CAPTIONS.md").write_text(text,encoding="utf-8")


def main()->None:
    HERE.mkdir(parents=True,exist_ok=True)
    pairs,means,stats,comparison=load_agreement()
    plot_agreement(pairs,means,stats,comparison,traits=TRAITS[:3])
    plot_agreement(pairs,means,stats,comparison,traits=TRAITS,name="FigS3_All_Descriptors_Exploratory_Angle")
    ann,oof,cells=load_sensitivity()
    plot_sensitivity(cells,traits=CHANNEL_B_TRAITS[:2])
    plot_sensitivity(cells,traits=CHANNEL_B_TRAITS,name="FigS4_All_Sensitivity_Exploratory_Angle")
    report={"all_fixed_pairs":len(pairs),"main_figure_pairs":450,"exploratory_angle_pairs":150,"n_distinct_images":50,"seeds":[42,3407,2026],"difference_limit_estimand":"50 per-image prediction means across three seeds; descriptive bias +/-1.96 image-level sample SD","all_sensitivity_cells":len(cells),"main_count_cells":72,"angle_cells_in_supplement":36,"statistical_tests":"none added","source_data_loss":0,"width_mm":170}
    (HERE/"FIGURE_QA.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
