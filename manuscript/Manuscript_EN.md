# Segmentation and contour-guided root reconstruction in soybean rhizobag images

Zeen Wang1,4, Junyao Tian2,4, He Zhang2,4, Fanchuang Li2,4, Tianyuan Song3,4, Zhanguo Zhang2,4, Yang Li2,4, Qingshan Chen2,4, Rongsheng Zhu2,4*

1 College of Intelligence Science and Engineering, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

2 College of Agriculture, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

3 School of Resources and Environment, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

4 State Key Laboratory of Smart Farm Technologies and Systems, Harbin 150030, China

*Address correspondence to: rshzhu@126.com

## Abstract

Individual-root measurement in rhizobag images requires both recovery of root pixels and reconstruction of continuous root paths. We combined a controlled segmentation comparison with a contour-supervised reconstruction method that integrates ordered-curve continuation, learned candidate quality and global path selection. Fifty annotated soybean seedling images were assessed using five image-level folds. Four segmentation configurations shared a 50-epoch budget and three training seeds. Fixed-skeleton-loss Attention U-Net achieved the highest mean out-of-fold Dice, 58.43%, exceeding pixel-loss Attention U-Net by 2.53 percentage points. However, under a common geometric extractor, its mean absolute errors were higher by 35.03 retained segments and 50.05 junction regions per image. In a separate reconstruction experiment using fixed segmentation inputs and 3290 whole-root polygon references, learned candidate quality and global selection increased one-to-one path matches from 586 to 682 at 75% reciprocal support and reduced image-level root-count mean absolute error from 63.00 to 21.36. Fragmented references decreased from 258 to 154. Although mixed outputs decreased from 674 to 562, their proportion among retained outputs increased from 10.47% to 16.82%. Ordered continuation lengthened one returning axis from 4077.01 to 4982.83 pixels and preserved a complete U-shaped axis, whereas another visible return remained untraced. These results show that pixel recovery and root-object reconstruction require complementary evaluation, and that whole-root contour supervision can improve path selection within the development collection while retaining identifiable structural errors.

Keywords: soybean; rhizobag imaging; Attention U-Net; whole-root reconstruction; skeleton graph; curve length

## 1. Introduction

Root system architecture influences the spatial distribution of resource acquisition and the response of crops to water and nutrient availability [1–3]. Image-based measurements of root extension, abundance and branching can support the characterization of this variation when the measured image structures are clearly defined. Rhizobags provide an accessible surface on which seedling roots can be photographed repeatedly, complementing washed-root scanning and three-dimensional imaging approaches [4–8]. For soybean, two-dimensional imaging platforms have demonstrated the value of root architectural measurements in genetic analysis [9].

Existing analysis tools approach root structure at different levels. SmartRoot supports interactive tracing, while ChronoRoot, RootNav 2.0 and the Root Phenotyping Toolbox combine learned segmentation with geometric or temporal reconstruction [10–13]. Directed structural searches have also been used to recover soybean stem phenotypes from image-derived landmarks [14]. These approaches illustrate the importance of combining local image evidence with an explicit representation of plant structure. Deep learning provides increasingly effective pixel classifiers, but the output required for measuring an individual root is an ordered path with a consistent identity [15–17].

The relationship between these two representations is particularly important in rhizobag images. A short missing connector may divide one root into several components, while an added foreground bridge can merge adjacent roots. Both changes may affect a skeleton graph more strongly than a whole-image overlap score. Recent comparisons have shown that segmentation rankings and root-length or diameter rankings need not coincide [18–20]. Skeleton-sensitive losses such as clDice address connectivity during segmentation, but their practical effect on subsequent object reconstruction depends on the image domain and the extraction procedure [21, 22]. A useful evaluation must therefore follow segmentation changes through the downstream measurements and examine root objects as well as pixels.

Here, we connect pixel recovery with root-object measurement through a contour-supervised reconstruction workflow for soybean rhizobag images. The study makes three contributions. First, a paired four-configuration comparison quantifies how segmentation changes propagate through an unchanged four-descriptor operator. Second, ordered-axis continuation and whole-candidate quality learning combine the existing polygon identities with global edge-ownership and root-count constraints. Third, complementary pixel, path and object evaluations retain all images and examine both returning axes and unresolved contacts, allowing improvements in count to be assessed alongside structural errors.

## 2. Materials and Methods

### 2.1. Workflow

Figure 1 summarizes image preparation, segmentation, root-path reconstruction and evaluation. The controlled segmentation analysis used complete-image folds before tile generation. Its outputs entered a common reference extractor to assess the propagation of mask differences. The subsequent reconstruction analysis used a separately developed, fixed three-channel segmentation pipeline and compared candidate-selection methods on identical inputs. The original root polygons supplied object-level supervision and computational path references. These analyses address different stages of the workflow and were evaluated separately.

![FIG1](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG1.png)

**Fig. 1.** Workflow for segmentation and root-object reconstruction. The upper stages show image preparation and the four-configuration segmentation comparison. A separate fixed three-channel input supplies the reconstruction analysis. Ordered axes, candidate quality prediction and global selection lead to curve and object-level outputs. Quality and count forests are fitted within their designated training folds. A development-stage oracle using all 50 annotations screened candidate-pool feasibility before these fits.

### 2.2. Images and root annotations

The dataset comprised 50 TIFF images of untreated Wm82 soybean seedlings photographed approximately 7 days after germination. Each image contained 3209 × 2311 pixels (height × width). The supplied LabelMe files delineated visible root regions with polygons. The data provider identified each polygon as one complete independent root, giving 3290 root objects across the collection. Foreground root pixels occupied 0.41–8.22% of the images. File-hash and pixel comparisons identified no duplicate source images.

Two representations were derived from the same annotations. For semantic segmentation, all root polygons were combined into one foreground mask, with root and background intensities of 255 and 0. For object-level reconstruction, the polygons retained their separate within-image identities. An isolated computational centerline was derived from each polygon for path comparison. All 3290 objects remained in the evaluation denominator, including two for which a usable reference curve could not be exported. The polygon references describe whole-root regions; branching order and parent–child attachment labels were not supplied. Figure 2 illustrates the distinction between the merged mask and the separate root objects.

![FIG2](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG2.png)

**Fig. 2.** Image and annotation representations for 10-1-9(3), which contains 33 root polygons and also appears in Figs. 4 and 8a, d. (a) Source photograph. (b) Merged root mask used for segmentation. (c) Separate computational centerlines derived from the root polygons; colors distinguish objects. Panels share image coordinates and scale (bar, 500 pixels).

### 2.3. Segmentation

#### 2.3.1. Image folds and tile preparation

The 50 image–mask pairs were assigned to five folds of ten images. Images were sorted by annotation foreground fraction, divided into ten blocks of five and randomly allocated within blocks using split seed 20260920. In rotation k, fold k supplied the test images, fold (k + 1) mod 5 supplied validation images, and the remaining 30 images supplied training tiles. All tiles from an image retained its fold assignment.

The controlled comparison used grayscale inputs enhanced by contrast-limited adaptive histogram equalization (CLAHE; clip limit 2.0, 16 × 16 grid). Training tiles measured 256 × 256 pixels at a 128-pixel stride. Tiles with more than 150 foreground pixels were retained; those with 31–150 pixels were retained with probability 0.5, and those with at most 30 pixels were excluded. Online augmentation applied right-angle rotations, independent horizontal and vertical flips, and gamma adjustment with probability 0.5 using an exponent between 0.8 and 1.2. The training seeds were 42, 3407 and 2026.

For complete-image prediction, the enhanced image was scaled to [0, 1] and reflection-padded on its lower and right boundaries. Every 256 × 256-pixel tile at the same stride entered inference. A two-dimensional Gaussian weighting window reduced tile-edge contributions during fusion. With center μ = 128 pixels and standard deviation σ = 64 pixels, the window was defined as follows:


$$G_{raw}(x,y)=\exp\left[-\frac{(x-\mu)^2+(y-\mu)^2}{2\sigma^2}\right]\tag{1}$$


$$G(x,y)=0.1+0.9\frac{G_{raw}(x,y)-\min G_{raw}}{\max G_{raw}-\min G_{raw}+\epsilon}\tag{2}$$


where ε = 10⁻⁸ and the second expression rescales the raw window to retain a nonzero boundary weight. The fused probability at each pixel was the weighted sum of tile predictions divided by the sum of their weights plus ε. Padding was removed after fusion.

#### 2.3.2. Network and loss configurations

U-Net and Attention U-Net shared the same encoder–decoder implementation [23, 24]. Each convolutional block contained two bias-free 3 × 3 convolutions with batch normalization and ReLU. Encoder widths were 32, 64, 128 and 256 channels, followed by a 512-channel bottleneck. Four decoder stages used bilinear upsampling and produced 256, 128, 64 and 32 channels. A 1 × 1 convolution and sigmoid supplied the foreground probability. The single-channel Attention U-Net in the controlled comparison contained 7,981,277 trainable parameters.

At each decoder level, an attention gate combined the deeper gating signal g and the encoder feature x:


$$q=\operatorname{ReLU}[\operatorname{BN}(W_g g)+\operatorname{BN}(W_x x)]\tag{3}$$


$$\alpha=\sigma[\operatorname{BN}(W_\psi q)]\tag{4}$$


$$\hat{x}=x\odot\alpha\tag{5}$$


where Wg, Wx and Wψ denote learned projections, BN denotes batch normalization, σ denotes the sigmoid, and ⊙ denotes element-wise multiplication. Intermediate gate widths were 128, 64, 32 and 16 channels from the deepest to the shallowest gate. Figure 3 shows the shared architecture and identifies the input-channel change used in the subsequent three-channel reconstruction pipeline.

![FIG3](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG3.png)

**Fig. 3.** Attention U-Net architecture. Encoder features pass through attention gates before concatenation with the corresponding decoder features. The controlled comparison uses a grayscale input; the later reconstruction input has three channels with the same encoder–decoder topology. Checkpoint and threshold selection are performed separately from the network architecture.

The four controlled configurations were U-Net with pixel loss, Attention U-Net with pixel loss, Attention U-Net with fixed skeleton loss, and Attention U-Net with a two-stage loss schedule. Pixel loss combined 0.2 binary cross-entropy and 0.8 soft Dice loss. Dice loss was calculated over the batch with smoothing constant 1:


$$L_{Dice}=1-\frac{2\sum_iP_iY_i+1}{\sum_iP_i+\sum_iY_i+1}\tag{6}$$


The fixed skeleton-loss configuration used the following coefficients throughout training:


$$L_1=0.2L_{BCE}+0.8L_{Dice}+0.3L_{skel}\tag{7}$$


Soft-clDice used 12 corrected soft-skeletonization iterations, with topology precision and sensitivity computed from the predicted and target skeletons [21]. The two-stage configuration used Eq. (7) through epoch 25 and then changed to:


$$L_2=0.4L_{BCE}+0.8L_{Dice}+0.3L_{skel}\tag{8}$$


All four configurations used AdamW, an initial learning rate of 10⁻³, weight decay of 10⁻⁴, cosine annealing toward 10⁻⁵, a batch size of 12 and 50 epochs. The checkpoint with highest validation-image mean Dice at threshold 0.50 was retained. For that checkpoint, a threshold between 0.10 and 0.90 at increments of 0.01 maximized validation-image mean Dice; exact ties favored the lower threshold. Test images were evaluated after these choices were fixed.

#### 2.3.3. Fixed three-channel input for reconstruction

The root-object reconstruction experiment used saved out-of-fold predictions from a three-channel Attention U-Net pipeline, designated D3. Its channels were CLAHE-enhanced grayscale, the Lab b channel, and absolute local grayscale contrast. The latter was the absolute difference between Gaussian-smoothed images at σ = 1 and σ = 12, multiplied by 8 and clipped to the 8-bit range. D3 was initialized from the corresponding validation-selected D1 fold checkpoint. D1 was a grayscale continuation of the original seed-42 fixed-skeleton-loss model, trained for 20 sampling rounds of 1024 crops with a batch size of 12, AdamW and a cosine learning rate from 10⁻⁴ to 10⁻⁵. Its augmentation and checkpoint-selection settings are provided in Additional file 1. The first convolution copied the D1 grayscale weights and initialized the additional channel weights to zero.

D3 used seed 42 and eight sampling rounds of 512 crops, each 256 × 256 pixels, with a batch size of 12. Half of the crop selections were guided by available training-image axis preferences with coordinate jitter of ±80 pixels; the remainder sampled root regions with jitter of ±112 pixels. Pixel targets remained the original combined root masks. Training used Eq. (7), with initial learning rates of 10⁻⁴ for the input convolution and 2 × 10⁻⁵ for the remaining network, reduced by cosine scheduling to 10% of their initial values. Checkpoints at rounds 0, 4 and 8 and the probability threshold were selected on the validation fold. A deterministic image orientation used the sign of the grayscale horizontal intensity moment, with SHA-256 comparison as a tie-break; all input channels were transformed together and predictions were restored to the original coordinates. The saved masks and probabilities were held constant throughout all reconstruction comparisons. The D3 reconstruction experiment and controlled segmentation comparison were analyzed separately because their inputs and training histories differed.

### 2.4. Geometric reference pipeline

A common deterministic extractor was applied to the annotation and prediction masks from the four controlled configurations [25]. Its first channel skeletonized the mask without closing or pruning. The eight-connected component with the greatest vertical span was selected, with ties resolved by component size and order. All its skeleton pixels were sorted lexicographically by row and column; the first and last pixels defined the search start and finish, irrespective of their graph degree. Dijkstra search used edge weights of 1 pixel for orthogonal steps and √2 pixels for diagonal steps. The resulting path descriptor was:


$$L_{path}=\sum_{e\in P}w_e\tag{9}$$


Components with fewer than ten skeleton pixels and unavailable paths were recorded as missing. This channel measured one dominant path in the selected connected component.

A separate channel applied 5 × 5-pixel elliptical closing, skeletonization and 15 simultaneous endpoint-removal iterations. Pixels with at least three eight-neighbors formed junction candidates:


$$J=S\odot\mathbf1[K_8*S\geq3]\tag{10}$$


where K8 counts the eight neighbors. Eight-connected groups of candidate pixels defined junction regions. After reconstructing an axis by the same weighted-path rule, the axis and junction pixels were removed. Remaining components with at least ten pixels defined the retained-segment count:


$$N_{seg}=\left|\{C_k:|C_k|\geq10\}\right|\tag{11}$$


For each retained component, its closest pixel to the ordered axis defined a segment base. The segment vector ended at the first component-array pixel 8–15 pixels from that base, or at the last array pixel if none met the interval. The axis vector extended 12 path steps downstream or to the endpoint. The angle was calculated by:


$$\theta=\arccos\left(\frac{v_p\cdot v_l}{\|v_p\|\|v_l\|}\right)\frac{180^\circ}{\pi}\tag{12}$$


The normalized dot product was clipped to [−1, 1], angles above 90° were folded to their acute equivalents, and valid values were averaged. These four reference descriptors were dominant path length, retained-segment count, junction-region count and mean local acute angle. The comparison retained this definition unchanged. Its parameter sensitivity was examined over 36 combinations of closing width (3, 5 or 7 pixels), endpoint-removal iterations (5, 10, 15 or 20) and minimum component size (5, 10 or 20 pixels).

### 2.5. Ordered axes and whole-root candidate learning

#### 2.5.1. Axis continuation and curve measurement

An existing incoming-direction selector supplied an ordered dominant-axis candidate for each fixed D3 input. Axis preferences used during its earlier development were choices among machine-generated paths. The reconstruction procedure retained the selected axis and considered the six saved alternatives from a fixed, more permissive foreground-support mask. Continuation required ordered overlap with the last max(64 pixels, 12 terminal radii) of the axis, an available foreground connector, initial directional agreement, a sufficiently long tail and no repeated old pixels. Only the supported tail was appended; the previous axis was preserved point for point.

The ordered path was resampled at approximately one-pixel arc intervals and smoothed with a seven-sample, second-order Savitzky–Golay filter. Its endpoints were restored after smoothing. Curve length was the sum of Euclidean distances between consecutive smoothed positions:


$$L_{curve}=\sum_{k=1}^{K-1}\|\widetilde{x}_{k+1}-\widetilde{x}_k\|_2\tag{13}$$


where x̃k is the kth ordered resampled position and K is the number of positions. The 8-neighbor raster arc was retained separately. This procedure includes upward-returning portions of a selected path because its endpoint is determined by path order rather than minimum image height. Algorithm 1 summarizes the procedure.

![ALG1](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/ALG1.png)

**Algorithm 1.** Ordered-axis continuation and curve measurement. A supported tail is appended to the existing axis without replacing its earlier pixels. The final ordered curve supplies a pixel-space length estimate.

#### 2.5.2. Complete-path candidates and features

The fixed D3 mask was converted to a skeleton graph in which edges represented root segments between endpoints or contact regions. Binary masks were processed in a deterministic orientation obtained by comparing SHA-256 hashes of the mask and its horizontal reflection; paths were restored to source-image coordinates before image-feature extraction. Four unlearned contact-pairing explanations considered direction and width continuity with different fixed combinations. Their complete traces were pooled with individual graph edges, and duplicate edge sets were removed. The preserved dominant axis was included as a protected candidate. Graph edges with at least half of their pixels within two pixels of the protected axis were assigned to it, and other candidates containing any assigned edge were excluded. Across the 50 images, this process produced 19,914 path candidates.

Before quality learning, an annotation-informed oracle assessed the complete candidate pool. The recorded development protocol permitted fitting when one-to-one matching at 75% reciprocal support reached at least 645 references. The pool yielded 1158 matches across all 50 images. This calculation used all reference labels and ignored edge-ownership and count constraints; it was a development feasibility screen and an upper bound for this candidate pool, not a deployable prediction.

Each candidate was described by 18 geometric and 12 image-derived features. Geometric features summarized length, displacement, spatial extent, radius distributions, curvature, edge coverage and contact gaps. Image features summarized grayscale and Lab values along the path, local contrasts between the path and its sides, and changes along the ordered curve. Full feature definitions and candidate-generation settings are provided with the numerical package. Figure 4 illustrates the relationship between the image, graph candidates and jointly selected paths.

![FIG4](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG4.png)

**Fig. 4.** Root-path reconstruction for image 10-1-9(3) (fold 4; 33 reference polygons) from the fixed D3 mask. (a) All 229 saved complete-path candidates. (b) The 49 objects selected by the primary method; colors distinguish objects, and 11 references were matched at 75% reciprocal support. (c) The protected axis (blue) and remaining selected paths (gray); graph edges assigned to the protected axis cannot be owned by another path. Saved image and path coordinates are shown without redrawing.

#### 2.5.3. Contour-derived targets and joint selection

Candidate quality was supervised using the independent polygon identities. For each candidate–reference pair, prediction support Pij measured the candidate arc fraction inside polygon i expanded by two pixels. Reference coverage Rij measured the polygon-derived centerline arc fraction within max(2 pixels, median reference radius + 1 pixel) of the candidate. The complete-root target was max_i min(Pij, Rij). A fragmentation surrogate used Pi*j(1 − Ri*j) for the best reciprocal-support reference i*, and a mixing surrogate used the smaller of the two largest candidate-support values across references.

Multi-output Extra-Trees regressors predicted these three targets from either geometric features alone or all 30 features [26]. Each regressor used 128 trees, maximum depth 12, minimum leaf size 6, a feature fraction of 0.75 and random state 20261019. Candidate weights summed to one within each training image. A separate image-level Extra-Trees regressor predicted root count from mask and graph features using 128 trees, maximum depth 6, minimum leaf size 3, absolute-error splitting and random state 20261019; count predictions below one were set to one. Five image-fold rotations fitted geometric-quality, image-plus-geometric-quality and count forests in each fold, giving 15 fits in total. Each fit excluded all ten images in the corresponding test fold.

For candidate j, the decoder combined predicted completeness q̂j, fragmentation f̂j, mixing m̂j and normalized graph-edge coverage aj. Binary variables zj indicated selection, and n̂ was the predicted image-level count. The objective was:


$$\max_{z}\ \sum_jz_j(2\hat q_j-1-\hat f_j-\hat m_j+a_j)-\left|\sum_jz_j-\hat n\right|,\quad\sum_{j:e\in E_j}z_j\leq1,\ z_0=1\tag{14}$$


The constraints required each graph edge to occur in at most one selected candidate and fixed the protected-axis variable to one. Coverage aj was the summed length of candidate edges divided by max(1, ℓ̄), where ℓ̄ was total graph-edge length divided by max(1, n̂). Two nonnegative slack variables represented absolute count deviation. SciPy mixed-integer optimization used an eight-second limit and relative gap tolerance of 10⁻⁶ [27]. Reference root identities and test-image root counts were not passed to decoding.

The primary method combined image-plus-geometric quality with the count penalty. Comparators were the previous fixed-pairing output, an unlearned geometric candidate score using the same learned count prior, geometric quality learning, and image-plus-geometric learning without the explicit count-deviation penalty. Removing that penalty retained the predicted count in coverage normalization. The unlearned geometric score set q̂j = ℓj/(ℓj + s), where ℓj was candidate path length and s = max(10 pixels, median candidate length in that image), with f̂j = m̂j = 0. All methods used the same fixed inputs within their designated comparison. Algorithm 2 gives the training and selection sequence.

![ALG2](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/ALG2.png)

**Algorithm 2.** Contour-supervised whole-root candidate quality and global selection after candidate-pool screening. Training images supply target values and count supervision. Within this fold-wise fitting and decoding procedure, the held-out image supplies features and fixed candidate paths only; its reference identities and count enter evaluation after decoding. The preceding oracle screen used the full development collection.

### 2.6. Evaluation and statistical analysis

#### 2.6.1. Segmentation and geometric agreement

Precision, recall, Dice and intersection over union were calculated from foreground true positives TP, false positives FP and false negatives FN:


$$\operatorname{Precision}=\frac{TP}{TP+FP}\tag{15}$$


$$\operatorname{Recall}=\frac{TP}{TP+FN}\tag{16}$$


$$F_1=\frac{2TP}{2TP+FP+FN}\tag{17}$$


$$\operatorname{IoU}=\frac{TP}{TP+FP+FN}\tag{18}$$


Complete-image scores were averaged over the 50 held-out images within each seed, followed by the mean and sample standard deviation across seeds. Hard clDice used binary-thinned masks. Narrow-root recovery was measured as exact coverage of annotation skeleton pixels by predicted foreground, stratified by annotation width ≤3, >3–6, >6–12 and >12 pixels. Width was twice the local distance to background; strata absent from an image were recorded separately.

Prediction and annotation masks entered the same reference extractor. Descriptor agreement was summarized by mean absolute error (MAE), root mean square error, bias, mean absolute percentage error (MAPE) for nonzero references and Lin’s concordance correlation coefficient (CCC) [28]. Image-mean difference plots (Additional file 1) used the prediction averaged over three seeds and descriptive limits at bias ± 1.96 sample standard deviations [29].

The interval procedures reflected the corresponding statistical units. Segmentation differences used 10,000 independent resamples of the three seed indices and 50 image indices, maintaining configuration pairing on their Cartesian product (seed 20260920). Descriptor differences first averaged seeds within an image and then resampled images separately within the five fixed folds (10,000 draws; seed 20260925). Deployment Dice resampled the 50 held-out image scores (10,000 draws; seed 20260926). Reported 95% descriptive intervals use the 2.5th and 97.5th percentiles of the corresponding resampled estimates. These intervals describe the recorded image collection and predictions. No hypothesis-test significance claims were made.

#### 2.6.2. Whole-root reconstruction

Whole-root paths were evaluated by one-to-one matching between references and outputs. A pair qualified at threshold t when both its prediction support and reference coverage were at least t. Hungarian assignment awarded max(reference count, output count) + 1 per qualifying pair, plus min(Pij, Rij) over every assigned pair. Pairs below threshold were discarded after assignment. This prioritizes qualifying-pair cardinality, with similarity over the full assignment as the secondary criterion. Matching was evaluated at t = 0.50, 0.75 and 0.90, with t = 0.75 used for the primary comparison. Matching proportions were reported against all 3290 references and, separately, against the number of output objects.

The count error for an image was the absolute difference between output-object count and polygon count. A fragmented-reference diagnostic identified a reference supported by two or more partial outputs (candidate support ≥0.75 and reference coverage ≥0.10). A mixed-output diagnostic identified an output with substantial support from at least two references (candidate support ≥0.20 and reference coverage ≥0.10). Fragmentation proportions used the reference count as denominator; mixing proportions used output count. A high-support incomplete output had Pij ≥ 0.75 but Rij < 0.75 for its best reference, defined by the largest min(Pij, Rij). These are contour-based structural diagnostics rather than independent anatomical adjudications.

All 50 images were retained in the reconstruction comparison. The reconstruction learners excluded the evaluated image fold, while the upstream D3 masks and selected axes remained fixed development inputs and candidate-pool feasibility had been screened using all reference labels. Consequently, this analysis assessed conditional internal reconstruction performance rather than a newly nested test of the entire image-to-root pipeline. Post-evaluation paired intervals resampled images within their fixed folds using 10,000 draws and seed 20261019. Per-image outcomes, ablations and changes in each fold were retained.

#### 2.6.3. Implementation

RootScope Desktop provided a reference implementation of the original segmentation and descriptor pipeline. Its five seed-42 ONNX models were evaluated on their respective held-out images using their validation-selected thresholds. Numerical verification compared source predictions and deployment outputs. CPU timing used the existing 150 image-runs, comprising three repetitions of each of 50 images, on an AMD Ryzen 7 7840H system with eight physical cores, Windows 11 and ONNX Runtime CPUExecutionProvider with eight intra-operation threads. Timing included image decoding, CLAHE, tiled inference and fusion, thresholding and the reference extractor; model loading, warm-up, reference evaluation and disk output were excluded. Stage quantiles and total-time quantiles were calculated independently.

The reconstruction prototype remained separate from the packaged reference application. Its analyses were reproduced from saved candidate, decoder and per-image evaluation records. Additional file 2 contains source tables, scripts, model/version identifiers and file hashes. Additional file 1 provides complete sensitivity results and development comparisons, including the later terminal-rollback experiment.

## 3. Results

### 3.1. Controlled segmentation comparison

All 60 controlled model fits completed, producing 600 held-out image predictions. Fixed-skeleton-loss Attention U-Net had the highest mean Dice, IoU, hard clDice and ≤3-pixel centerline recall among the four configurations (Table 1). Its mean Dice was 58.43% ± 0.46 percentage points, compared with 55.90% ± 0.61 for pixel-loss Attention U-Net and 55.04% ± 1.08 for U-Net. The two-stage configuration achieved 57.54% ± 2.81, with greater variation across training seeds.

**Table 1.** Held-out segmentation performance under the shared controlled protocol. Values are means ± sample standard deviations of three seed-specific image means. Each seed mean uses the same 50 source images.

| Configuration | Dice (%) | IoU (%) | Hard clDice (%) | ≤3-pixel centerline recall (%) |
|---|---|---|---|---|
| U-Net, pixel loss | 55.04 ± 1.08 | 38.21 ± 0.96 | 62.07 ± 4.55 | 58.83 ± 1.37 |
| Attention U-Net, pixel loss | 55.90 ± 0.61 | 38.99 ± 0.61 | 64.63 ± 1.52 | 59.03 ± 0.41 |
| Attention U-Net, fixed skeleton loss | 58.43 ± 0.46 | 41.49 ± 0.43 | 67.43 ± 0.67 | 65.64 ± 1.17 |
| Attention U-Net, two-stage loss | 57.54 ± 2.81 | 40.65 ± 2.74 | 65.86 ± 3.28 | 63.48 ± 8.80 |

Attention gates alone changed paired mean Dice by 0.86 percentage points relative to U-Net, with a descriptive interval of −0.68 to 2.85 percentage points. Fixed skeleton loss increased paired mean Dice by 2.53 percentage points relative to pixel-loss Attention U-Net, with an interval of 1.37–3.40 percentage points and positive fold means in all five folds. The two-stage configuration differed from fixed weighting by −0.89 percentage points (interval −3.48 to 1.58), with a positive fold mean in one fold. Figure 5 presents the seed-specific complete-image averages and paired contrasts.

![FIG5](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG5.png)

**Fig. 5.** Controlled segmentation results. (a) Out-of-fold macro Dice; symbols indicate training seeds (circle, 42; square, 3407; triangle, 2026), with mean ± SD of the three seed values. (b) Paired image-level Dice differences in percentage points (pp): Attention, Attention U-Net minus U-Net (pixel loss); +fixed clDice, fixed skeleton loss minus pixel loss; +two-stage, two-stage minus fixed skeleton loss. Points are three-seed image means, vertical lines span seed values and black bars are 50-image means. (c) Fold-by-seed means (open symbols); short bars are fold means, long bars overall means, and labels give folds with a positive mean. Fixed clDice denotes the fixed skeleton-loss configuration (Eq. 7).

Exact centerline recall for fixed skeleton loss was 65.64% for roots ≤3 pixels wide, 72.44% at >3–6 pixels, 74.89% at >6–12 pixels and 75.77% above 12 pixels. Matched source and error-map crops illustrate the variation in narrow-root recovery (Fig. 6). Fourteen of 15 two-stage runs selected checkpoints at or before epoch 25, and one selected epoch 27. Thus, most selected two-stage predictions originated before the coefficient change.

![FIG6](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG6.png)

**Fig. 6.** Root-width-dependent centerline recovery. (a) Exact centerline recall by reference width, defined as twice the distance-transform value at the reference skeleton pixel. Lines are means of three seed-specific 50-image means; bands show ±1 SD across seeds. (b, c) Seed-42 fixed-skeleton-loss predictions for the lowest and highest whole-image ≤3-pixel recall among images with at least 750 such reference skeleton pixels; percentages are whole-image values. Each 384 × 384-pixel crop maximized missed (b) or recovered (c) narrow reference pixels, illustrating contrasting cases rather than typical performance. Left, grayscale source (1st–99th percentile display stretch); middle, reference thin centerline; right, prediction errors. Scale bars, 100 pixels.

### 3.2. Propagation to reference descriptors

All four reference descriptors had 50 evaluable image pairs per seed. For fixed skeleton loss, dominant-path MAE was 1210.52 pixels and mean bias was −1137.20 pixels (Table 2). Retained-segment and candidate-junction counts exceeded annotation-mask outputs by 95.47 and 109.85 on average. The local acute-angle MAE was 6.84°, measuring agreement of the computational image-level angle summaries.

**Table 2.** Agreement between fixed-skeleton-loss predictions and annotation masks under the same reference extractor. MAE and bias retain the descriptor’s unit; CCC and MAPE summarize three seed-specific calculations. Angle values concern the computational local-angle descriptor.

| Descriptor | Unit | MAE ± SD | Mean bias | CCC ± SD | MAPE (%) |
|---|---|---|---|---|---|
| Dominant path length | pixel | 1210.52 ± 56.26 | −1137.20 | 0.321 ± 0.047 | 41.2 |
| Retained segment count | count | 97.43 ± 3.98 | +95.47 | 0.575 ± 0.023 | 100.5 |
| Candidate junction-region count | count | 111.71 ± 5.32 | +109.85 | 0.430 ± 0.034 | 147.8 |
| Mean local acute angle | degree | 6.84 ± 0.42 | −3.99 | 0.261 ± 0.097 | 12.1 |

Configuration rankings differed between segmentation overlap and descriptor error (Table 3). Fixed skeleton loss increased retained-segment and junction-region MAEs by 35.03 and 50.05 relative to pixel-loss Attention U-Net. Both increases were observed in all three seeds and all five fold means. Corresponding descriptive intervals were 25.89–44.75 retained segments and 37.75–63.03 junction regions. The paired descriptor comparisons therefore identified a trade-off that was not apparent from Dice alone.

The corresponding path-length MAE difference was −32.93 pixels (descriptive interval −117.53 to 48.27), whereas the exploratory local-angle MAE difference was −1.18° (−1.83 to −0.55°). Thus, the descriptor responses differed in both direction and consistency: the two count errors increased, the angle summary had a lower absolute error against the same computational operator, and the path-error interval included changes in either direction.

**Table 3.** Descriptor MAEs across the controlled segmentation configurations. Each cell is the mean ± sample standard deviation of three seed-specific 50-image MAEs. References and predictions use the same unchanged reference operator.

| Configuration | Path (pixel) | Segments (count) | Junctions (count) | Angle (degree) |
|---|---|---|---|---|
| U-Net, pixel loss | 1258.8 ± 243.2 | 68.2 ± 26.7 | 62.1 ± 28.0 | 8.0 ± 1.5 |
| Attention U-Net, pixel loss | 1243.5 ± 152.1 | 62.4 ± 16.1 | 61.7 ± 23.8 | 8.0 ± 0.9 |
| Attention U-Net, fixed skeleton loss | 1210.5 ± 56.3 | 97.4 ± 4.0 | 111.7 ± 5.3 | 6.8 ± 0.4 |
| Attention U-Net, two-stage loss | 1200.2 ± 130.6 | 107.4 ± 12.1 | 126.1 ± 19.1 | 7.3 ± 0.3 |

Extractor settings also changed the measurements. Across the 36-setting grid, mean annotation-derived retained-segment count ranged from 111.96 to 165.74, compared with 144.72 at the default. Segment MAE ranged from 45.00 to 217.21. Complete four-descriptor agreement and sensitivity results are reported in Additional file 1. Constructed full-mask examples reproduced a 54.16° output for a known 45° branch and a shortened returning path, confirming that extraction rules can introduce errors independently of segmentation.

### 3.3. Whole-root candidate learning and selection

The annotation-informed candidate-pool oracle matched 1158 of 3290 reference roots (35.20%) at 75% reciprocal support before edge conflicts and count constraints were imposed. The primary reconstruction method retained 3342 output objects, compared with 6440 for fixed pairing (Table 4). One-to-one matches increased from 586 to 682, corresponding to 17.81% and 20.73% of the reference roots. The fraction of outputs participating in these matches increased from 9.10% to 20.41%. Image-level count MAE decreased from 63.00 to 21.36.

**Table 4.** Whole-root reconstruction on the same fixed D3 inputs. Matching and fragmentation percentages use 3290 reference roots; mixed-output percentages use each method’s output count. The primary method is whole-root image-plus-geometric learning with global selection and the count-deviation penalty.

| Method | Output objects | Matched references, n (%) | Count MAE | Fragmented references, n (%) | Mixed outputs, n (%) |
|---|---|---|---|---|---|
| Fixed pairing | 6440 | 586 (17.81) | 63.00 | 258 (7.84) | 674 (10.47) |
| Geometric score + count prior | 3453 | 518 (15.74) | 22.74 | 185 (5.62) | 666 (19.29) |
| Whole-root geometric learning | 3334 | 675 (20.52) | 21.20 | 152 (4.62) | 572 (17.16) |
| Image learning without count penalty | 1571 | 365 (11.09) | 35.22 | 55 (1.67) | 424 (26.99) |
| Whole-root image learning (primary) | 3342 | 682 (20.73) | 21.36 | 154 (4.68) | 562 (16.82) |

Fragmented references decreased from 258 to 154, equivalent to 7.84% and 4.68% of references. Mixed outputs decreased from 674 to 562, but their proportion increased from 10.47% to 16.82% because fewer objects were retained. High-support incomplete outputs decreased from 1690 to 872; their output proportions were approximately 26% in both methods. Thus, count and complete-path matching improved without a corresponding reduction in the frequency of every error type among retained objects.

Count error improved in 45 images and worsened in five. Path matching increased in 35 images, was unchanged in five and decreased in ten. Count MAE decreased in all five folds. Matches increased in folds 1–4 (129 to 171, 79 to 111, 111 to 123 and 100 to 117), whereas fold 0 matches decreased from 167 to 160 while count MAE fell from 63.70 to 17.30. Mean paired changes per image were +1.92 matches (descriptive interval 0.96–2.94) and −41.64 absolute count error (−53.92 to −30.54). Figure 7 displays all image-level changes, including regressions.

![FIG7](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG7.png)

**Fig. 7.** Whole-root reconstruction on identical fixed D3 inputs. A, fixed pairing; B, unlearned geometric score with the learned count prior; C, whole-root geometric learning; D, image-plus-geometric learning without the count-deviation penalty; E, image-plus-geometric learning with the penalty (primary). (a) References matched one-to-one at 75% reciprocal support. (b) Root-count MAE. (c) Fragmented references (% of 3290). (d) Mixed outputs (% of each method’s outputs: A, 6440; B, 3453; C, 3334; D, 1571; E, 3342). (e, f) Per-image changes, E minus A, for all 50 images in alphabetical order; shapes distinguish image folds, and negative values in (e) are regressions.

The ablations separated improvements in count from improvements in path selection. An unlearned geometric score with the same count prior reduced count MAE to 22.74 but yielded only 518 matches, below the fixed-pairing result. Learned geometric quality produced 675 matches and count MAE 21.20; adding image features produced 682 and 21.36. Removing the explicit count-deviation penalty retained 1571 objects with 365 matches. The main improvement therefore arose from combining whole-candidate quality with joint selection, while the incremental contribution of image features was small in this comparison.

### 3.4. Returning axes and image-level examples

Ordered continuation recovered a visible upward-returning segment in image 10-1-9(3), increasing its curve estimate from 4077.01 to 4982.83 pixels. The previously complete U-shaped axis in 235-25-7(1) retained a length of 5028.08 pixels. All whole-root selection methods subsequently preserved these axes point for point, separating curve continuation from the selection of the remaining root objects.

Continuation remained incomplete in image 235-25-9(1), where the route ended in a short branch near the bottom and a visible return remained untraced; its curve estimate remained 3134.07 pixels. Figure 8 contrasts the recovered return, the complete U and the unresolved case using saved paths. A fixed terminal-rollback comparison on all 50 images added path arc but introduced local detours and was not adopted; the complete results are given in Additional file 1.

![FIG8](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/submission-en-20261020/FIG8.png)

**Fig. 8.** Returning-root examples from saved paths. The upper row shows the input ordered axis and the lower row shows the retained or extended axis on the same photograph. Columns show the recovered return in 10-1-9(3), the retained U in 235-25-7(1) and the unresolved return in 235-25-9(1). Blue marks the retained input path and orange marks an appended tail; circles and squares mark path starts and ends. The panels share source-image scale (bar, 500 pixels). Curve estimates are in pixels. The examples illustrate the reconstruction behavior and do not estimate an independent success rate.

### 3.5. Reference implementation

The original seed-42 ONNX pipeline achieved image-level macro Dice of 58.10% with a descriptive image-bootstrap interval of 56.63–59.52% on its 50 corresponding held-out images. Maximum complete-image Dice and IoU differences from the PyTorch research pipeline were 0.00046 and 0.00043 on the 0–1 scale. The existing CPU records gave a total median of 52.26 s per image, an interquartile range of 49.05–57.41 s and P90 of 66.55 s across 150 image-runs. Stage-level quantiles are given in Additional file 1. These measurements characterize the reference deployment pipeline; the reconstruction prototype was evaluated separately.

## 4. Discussion

### 4.1. Evaluating pixels and reconstructed root objects

The controlled comparison showed that fixed skeleton loss improved mask overlap and exact recovery of narrow annotation centerlines, while increasing segment- and junction-count errors under the common geometric operator. These outcomes concern different image representations. Pixel overlap summarizes local classification, whereas graph measurements depend on how foreground pixels connect. A small missing connector or an added bridge can therefore change the object structure without dominating Dice. This observation complements multi-dataset comparisons of segmentation and root-trait rankings [18] and supports evaluation of the downstream quantity for which a model will be used [20].

The two-stage comparison also emphasizes the importance of model-selection behavior. Most selected checkpoints preceded the weighting change, and the later stage altered both relative weights and total loss scale. The available contrast therefore does not establish an independent benefit of that stage. Fixed skeleton loss performed better for the segmentation endpoints in this collection, but its selection as a deployment reference should not be interpreted as optimality for all geometric descriptors.

### 4.2. Whole-root supervision and the count–structure trade-off

Retaining separate polygon identities made it possible to evaluate and learn properties of complete candidates rather than only local contacts. The learned targets penalized incomplete support and ambiguity across root regions, while global selection prevented multiple selected candidates from owning the same graph edge. The improved matching and count error indicate a useful contribution of this combined representation under fixed inputs. The geometric-score control is informative: count MAE approached the learned result while matches fell below fixed pairing. A plausible number of outputs is therefore insufficient evidence that the individual paths are correct.

Candidate availability also constrained reconstruction. Even annotation-informed matching over the full pool reached only 1158 references at 75% reciprocal support, before enforcing the decoder's edge conflicts or count penalty. Learning a better selector within this fixed pool cannot recover a path that the pool does not adequately represent. The remaining gap therefore involves candidate construction as well as candidate ranking.

The absolute number and relative frequency of mixed outputs give complementary information. Although the primary method removed many extraneous objects, the remaining set contained a larger fraction of mixed paths. Likewise, a lower total number of incomplete objects did not remove their substantial output proportion. These results motivate reporting object counts, reference coverage and structural error diagnostics together. The small difference between geometric and image-plus-geometric learning also argues for a measured interpretation of the image features rather than attributing the improvement to color information alone.

### 4.3. Curves, contacts and retained axes

Representing a root as an ordered curve separates its length from image height. A route may turn upward while continuing to accumulate arc length, and preserving its earlier coordinates avoids cutting a complete U when a later stage changes candidate selection. The returning-root examples demonstrate this practical benefit. Protection also retains unresolved portions of the input route, however, as shown by the incomplete axis in 235-25-9(1).

The local failure is not explained by curvature alone. At a crowded contact, a smoother or wider continuation may enter a nearby root, and permissive foreground support may preserve a false connection. The terminal-rollback experiment illustrates this distinction: allowing an earlier decision to be reversed can recover additional arc, but does not establish the identity of the continuation. More complete architectures require evidence linking root order, endpoints and attachment locations, as incorporated in tools such as RootNav 2.0 and standardized structural descriptions [12, 30]. RootNav, SmartRoot and RhizoVision Explorer also provide different reconstruction or measurement definitions, so comparisons must align the intended outputs [10, 12, 31, 32].

### 4.4. Practical use

For practical screening, the results favor retaining explicit route and missing-value information alongside numerical outputs. Root length and object counts should be interpreted with image overlays when contacts or weak foreground make reconstruction uncertain. Software provenance and repeatable numerical records facilitate this inspection, while broader phenotyping applications require validation appropriate to the targeted biological quantity [9, 20, 33–37]. The paired analyses provide a defined basis for testing pixel, object and curve outputs before their use as biological measurements.

### 4.5. Limitations

The study used 50 development images of one soybean genotype, without verified plant or acquisition-batch grouping. The reconstruction analysis reused this collection, a single-seed D3 input and fixed axes, and its candidate-pool feasibility screen used all reference labels before fold-wise learning. It therefore supplies conditional internal evidence rather than new-cohort validation. Polygon-derived centerlines and structural diagnostics are computational references; the annotations do not independently establish anatomical root order, attachment sites or emergence angles. Length remains a two-dimensional pixel-space estimate without physical calibration. These conditions limit biological and cross-condition interpretation, while supporting reproducible comparison within the stated acquisition domain.

## 5. Conclusions

Fixed skeleton loss improved segmentation overlap in the controlled soybean rhizobag comparison, but the corresponding changes did not consistently reduce reference-descriptor errors. Whole-root polygon supervision, complete-path quality learning and global selection increased matches from 586 to 682 and reduced image-level count MAE from 63.00 to 21.36 on fixed development inputs. Ordered curve measurement preserved selected returning axes, while the retained mixed-output proportion and unresolved contacts defined the remaining reconstruction errors. Joint pixel-, object- and curve-level evaluation provides a practical framework for assessing these distinct stages of root-image analysis.

## Abbreviations

AG, attention gate; BCE, binary cross-entropy; BN, batch normalization; CCC, concordance correlation coefficient; CLAHE, contrast-limited adaptive histogram equalization; CPU, central processing unit; CSV, comma-separated values; IoU, intersection over union; IQR, interquartile range; MAE, mean absolute error; MAPE, mean absolute percentage error; ONNX, Open Neural Network Exchange; OOF, out-of-fold; P90, 90th percentile; pp, percentage points; RMSE, root mean square error; SD, standard deviation; SHA-256, Secure Hash Algorithm 256-bit.

## Statements and Declarations

### Availability of data and materials

The datasets supporting the conclusions of this article are available in the versioned GitHub research collection [38]. It includes the 50 original TIFF photographs, their LabelMe annotations and merged reference masks, the frozen segmentation and reconstruction records, study-specific code and selected models. File manifests identify image folds, annotation-object counts and SHA-256 hashes. The large image, model and source archives are separate repository assets; Additional files 2 and 3 provide portable numerical and synthetic-geometry replay without downloading the photographs or model weights.

The original RootScope Desktop implementation and its five ONNX models retain their historical software version. The reconstruction prototype is identified separately and does not replace the packaged application. Source code is written in Python; the reference desktop package was evaluated on Windows 11, and analysis dependencies and reproduction commands are recorded with the corresponding source snapshot. Software retains the repository's MIT licence. Images and annotations retain the provider's copyright, with public access authorized and reuse terms stated in the accompanying data-licence file.

[Repository](https://github.com/goldriderhszz-hash/root-phenotyping-attunet) · [Historical desktop software](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop) · [Historical source archive](https://archive.softwareheritage.org/swh:1:rev:c9103a920e84e69ee6c2280a9cc75b4830f5a286/)

### Competing interests

The authors declare that they have no competing interests.

### Funding

This work was supported by the National Key R&D Program Project funded by the Ministry of Agriculture and Rural Affairs of the People’s Republic of China under Grant No. 2025YFD2300401; the Research and Application of Key Technologies for Intelligent Farming Decision Platform (an Open Competition Project of Heilongjiang Province, China) under Grant No. 2021ZXJ05A03; the Programs for Science and Technology Development of Heilongjiang Province under Grant No. 2024ZX01A07; and the Heilongjiang Province Key Research and Development Plan (Innovation Bases), China under Grant No. JD2023GJ01-01.

### Authors’ contributions

Z.W. conceptualized the methodology, developed the algorithm, and wrote the main manuscript text. J.T., H.Z., and F.L. conducted the rhizobag cultivation experiments, collected the high-resolution root images, and organized the datasets. T.S., Z.Z., Y.L., and Q.C. assisted with data curation, manual annotation, and figure preparation. R.Z. supervised the overall project, acquired funding, and guided the research design. All authors reviewed the manuscript.

### Ethics approval and consent to participate

Not applicable.

### Consent for publication

Not applicable.

### Acknowledgements

Not applicable.

### AI assistance

OpenAI Codex and Notion AI assisted with manuscript drafting, numerical verification and figure preparation. The authors remain responsible for the scientific content and final approval. Workflow guidance used Scientific Agent Skills [39].

## References

1. Mehra P, Banda J, Ogorek LLP, Fusi R, Castrillo G, Colombi T, et al. Root growth and development in “real life”: advances and challenges in studying root–environment interactions. Annu Rev Plant Biol. 2025;76:467–492. https://doi.org/10.1146/annurev-arplant-083123-074506

2. Uga Y, Sugimoto K, Ogawa S, Rane J, Ishitani M, Hara N, et al. Control of root system architecture by DEEPER ROOTING 1 increases rice yield under drought conditions. Nat Genet. 2013;45(9):1097–1102. https://doi.org/10.1038/ng.2725

3. Gamuyao R, Chin JH, Pariasca-Tanaka J, Pesaresi P, Catausan S, Dalid C, et al. The protein kinase Pstol1 from traditional rice confers tolerance of phosphorus deficiency. Nature. 2012;488:535–539. https://doi.org/10.1038/nature11346

4. Atkinson JA, Pound MP, Bennett MJ, Wells DM. Uncovering the hidden half of plants using new advances in root phenotyping. Curr Opin Biotechnol. 2019;55:1–8. https://doi.org/10.1016/j.copbio.2018.06.002

5. Tracy SR, Nagel KA, Postma JA, Fassbender H, Wasson A, Watt M. Crop improvement from phenotyping roots: highlights reveal expanding opportunities. Trends Plant Sci. 2020;25(1):105–118. https://doi.org/10.1016/j.tplants.2019.10.015

6. Yang W, Feng H, Zhang X, Zhang J, Doonan JH, Batchelor WD, et al. Crop phenomics and high-throughput phenotyping: past decades, current challenges, and future perspectives. Mol Plant. 2020;13(2):187–214. https://doi.org/10.1016/j.molp.2020.01.008

7. van Dusschoten D, Metzner R, Kochs J, Postma JA, Pflugfelder D, Bühler J, et al. Quantitative 3D analysis of plant roots growing in soil using magnetic resonance imaging. Plant Physiol. 2016;170(3):1176–1188. https://doi.org/10.1104/pp.15.01388

8. Mairhofer S, Zappala S, Tracy SR, Sturrock C, Bennett M, Mooney SJ, et al. RooTrak: automated recovery of three-dimensional plant root architecture in soil from X-ray microcomputed tomography images using visual tracking. Plant Physiol. 2012;158(2):561–569. https://doi.org/10.1104/pp.111.186221

9. Chandnani R, Qin T, Ye H, Hu H, Panjvani K, Tokizawa M, et al. Application of an improved 2-dimensional high-throughput soybean root phenotyping platform to identify novel genetic variants regulating root architecture traits. Plant Phenomics. 2023;5:0097. https://doi.org/10.34133/plantphenomics.0097

10. Lobet G, Pagès L, Draye X. A novel image-analysis toolbox enabling quantitative analysis of root system architecture. Plant Physiol. 2011;157(1):29–39. https://doi.org/10.1104/pp.111.179895

11. Gaggion N, Ariel F, Daric V, Lambert É, Legendre S, Roulé T, et al. ChronoRoot: high-throughput phenotyping by deep segmentation networks reveals novel temporal parameters of plant root system architecture. GigaScience. 2021;10(7):giab052. https://doi.org/10.1093/gigascience/giab052

12. Yasrab R, Atkinson JA, Wells DM, French AP, Pridmore TP, Pound MP. RootNav 2.0: deep learning for automatic navigation of complex plant root architectures. GigaScience. 2019;8(11):giz123. https://doi.org/10.1093/gigascience/giz123

13. Shi J, Xie S, Li W, Wang X, Wang J, Chen Y, et al. RPT: an integrated root phenotyping toolbox for segmenting and quantifying root system architecture. Plant Biotechnol J. 2025;23(6):2095–2109. https://doi.org/10.1111/pbi.70040

14. Guo Y, Gao Z, Zhang Z, Li Y, Hu Z, Xin D, et al. Automatic and accurate acquisition of stem-related phenotypes of mature soybean based on deep learning and directed search algorithms. Front Plant Sci. 2022;13:906751. https://doi.org/10.3389/fpls.2022.906751

15. Murphy KM, Ludwig E, Gutierrez J, Gehan MA. Deep learning in image-based plant phenotyping. Annu Rev Plant Biol. 2024;75:771–795. https://doi.org/10.1146/annurev-arplant-070523-042828

16. Pound MP, Atkinson JA, Townsend AJ, Wilson MH, Griffiths M, Jackson AS, et al. Deep machine learning provides state-of-the-art performance in image-based plant phenotyping. GigaScience. 2017;6(10):gix083. https://doi.org/10.1093/gigascience/gix083

17. Zeng D, Li M, Jiang N, Ju Y, Schreiber H, Chambers E, et al. TopoRoot: a method for computing hierarchy and fine-grained traits of maize roots from 3D imaging. Plant Methods. 2021;17:127. https://doi.org/10.1186/s13007-021-00829-z

18. Smith AG, Lamprinidis S, Seethepalli A, York LM, Han E, Möhl P, et al. A systematic comparison of transformers and ConvNets for root segmentation across nine datasets. Plant Methods. 2026;22:52. https://doi.org/10.1186/s13007-026-01533-6

19. Baykalov P, Bussmann B, Nair R, Smith AG, Bodner G, Hadar O, et al. Semantic segmentation of plant roots from RGB (mini-) rhizotron images—generalisation potential and false positives of established methods and advanced deep-learning models. Plant Methods. 2023;19:122. https://doi.org/10.1186/s13007-023-01101-2

20. Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21:195–212. https://doi.org/10.1038/s41592-023-02151-z

21. Shit S, Paetzold JC, Sekuboyina A, Ezhov I, Unger A, Zhylka A, et al. clDice – a novel topology-preserving loss function for tubular structure segmentation. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021. p. 16560–16569. https://doi.org/10.1109/CVPR46437.2021.01629

22. Ma J, Chen J, Ng M, Huang R, Li Y, Li C, et al. Loss odyssey in medical image segmentation. Med Image Anal. 2021;71:102035. https://doi.org/10.1016/j.media.2021.102035

23. Oktay O, Schlemper J, Le Folgoc L, Lee M, Heinrich M, Misawa K, et al. Attention U-Net: learning where to look for the pancreas. arXiv. 2018;1804.03999. https://doi.org/10.48550/arXiv.1804.03999

24. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Lecture Notes in Computer Science, vol. 9351. Cham: Springer; 2015. p. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

25. van der Walt S, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2:e453. https://doi.org/10.7717/peerj.453

26. Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Machine Learning. 2006;63:3–42. https://doi.org/10.1007/s10994-006-6226-1

27. Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy1.0: fundamental algorithms for scientific computing in Python. Nat Methods. 2020;17:261–272. https://doi.org/10.1038/s41592-019-0686-2

28. Lin LI. A concordance correlation coefficient to evaluate reproducibility. Biometrics. 1989;45(1):255–268. https://doi.org/10.2307/2532051

29. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;327(8476):307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

30. Lobet G, Pound MP, Diener J, Pradal C, Draye X, Godin C, et al. Root system markup language: toward a unified root architecture description language. Plant Physiol. 2015;167(3):617–627. https://doi.org/10.1104/pp.114.253625

31. Seethepalli A, York LM. RhizoVision Explorer: interactive software for generalized root image analysis designed for everyone. Version 2.0.3. Zenodo; 2021. https://doi.org/10.5281/zenodo.5121845

32. Seethepalli A, Dhakal K, Griffiths M, Guo H, Freschet GT, York LM. RhizoVision Explorer: open-source software for root image analysis and measurement standardization. AoB PLANTS. 2021;13(6):plab056. https://doi.org/10.1093/aobpla/plab056

33. Handy G, Carter I, Mackenzie AR, Esquivel-Muelbert A, Smith AG, Yaffar D, et al. Variation in forest root image annotation by experts, novices, and AI. Plant Methods. 2024;20:154. https://doi.org/10.1186/s13007-024-01279-z

34. Lobet G, Koevoets IT, Noll M, Meyer PE, Tocquin P, Pagès L, et al. Using a structural root system model to evaluate and improve the accuracy of root image analysis pipelines. Front Plant Sci. 2017;8:447. https://doi.org/10.3389/fpls.2017.00447

35. Topp CN, Iyer-Pascuzzi AS, Anderson JT, Lee CR, Zurek PR, Symonova O, et al. 3D phenotyping and quantitative trait locus mapping identify core regions of the rice genome controlling root architecture. Proc Natl Acad Sci U S A. 2013;110(18):E1695–E1704. https://doi.org/10.1073/pnas.1304354110

36. Li J, Bo W, Bu C, Zhou J, Li P, Wang M, et al. Integrating 3D imaging, GWAS, and single-cell transcriptome approaches to elucidate root system architecture in Populus. Plant Physiol. 2025;199(2):kiaf432. https://doi.org/10.1093/plphys/kiaf432

37. Li P, Zhu T, Wang Y, Zhang X, Yang X, Fang S, et al. Natural variation in a cortex/epidermis-specific transcription factor bZIP89 determines lateral root development and drought resilience in maize. Sci Adv. 2025;11(15):eadt1113. https://doi.org/10.1126/sciadv.adt1113

38. Soybean rhizobag image dataset, study code and frozen reconstruction records. Version submission-en-20261020. root-phenotyping-attunet, GitHub. 2026. https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/submission-en-20261020. Accessed 20 Oct 2026.

39. Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv. 2026;2609.00065. https://doi.org/10.48550/arXiv.2609.00065
