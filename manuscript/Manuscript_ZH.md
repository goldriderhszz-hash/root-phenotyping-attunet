# 大豆萌发袋图像的分割与轮廓引导根系重建

Zeen Wang1,4, Junyao Tian2,4, He Zhang2,4, Fanchuang Li2,4, Tianyuan Song3,4, Zhanguo Zhang2,4, Yang Li2,4, Qingshan Chen2,4, Rongsheng Zhu2,4*

1 College of Intelligence Science and Engineering, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

2 College of Agriculture, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

3 School of Resources and Environment, Northeast Agricultural University, Harbin 150030, Heilongjiang, China;

4 State Key Laboratory of Smart Farm Technologies and Systems, Harbin 150030, China

*Address correspondence to: rshzhu@126.com

## 摘要

萌发袋图像中的单根测量既需要恢复根像素，也需要重建连续根路径。本研究将受控分割对照与轮廓监督的重建方法结合，后者包含有序曲线续接、候选质量学习及全局路径选择。对50张标注大豆幼苗图像采用五个图像级折进行评价；四种分割配置具有相同的50轮训练预算，并分别使用三个训练种子。固定骨架损失Attention U-Net获得最高折外平均Dice，为58.43%，比像素损失Attention U-Net高2.53个百分点。但在共同几何提取器下，每图保留根段及交点区域MAE分别高35.03与50.05。在另一项采用固定分割输入与3290个整根多边形参照的重建实验中，候选质量学习与全局选择使75%双向支持标准下的一对一路径匹配由586增至682，图像级根数平均绝对误差由63.00降至21.36。被切碎的参照根由258减至154；混合输出虽由674减至562，其在保留输出中的比例却由10.47%升至16.82%。有序续接使一条返回根轴由4077.01增至4982.83像素，并保留完整U形根轴，但另一处可见返回仍未追踪。结果表明，像素恢复与根对象重建需要互补评价，整根轮廓监督能够改善开发图集内的路径选择，同时仍存在可识别的结构性错误。

关键词：大豆；萌发袋成像；Attention U-Net；整根重建；骨架图；曲线长度

## 1. 引言

根系构型影响作物获取资源的空间分布及其对水分、养分供给的响应[1–3]。在明确所测图像结构的前提下，基于图像的根系伸长、数量和分枝测量可用于描述这类差异。萌发袋提供了便于重复拍摄幼苗根系的表面，与洗根扫描和三维成像等方法形成互补[4–8]。针对大豆，已有二维成像平台展示了根系构型测量在遗传分析中的价值[9]。

现有工具在不同层面处理根系结构。SmartRoot支持交互式描迹；ChronoRoot、RootNav 2.0和Root Phenotyping Toolbox将学习式分割与几何或时序重建结合[10–13]。定向结构搜索也被用于从图像标志点提取大豆茎部表型[14]。这些研究表明，局部图像证据需要与明确的植物结构表示结合。深度学习不断提高像素分类能力，但测量单条根所需的输出，是身份一致的有序路径[15–17]。

在萌发袋图像中，这两类表示之间的关系尤其重要。一个很短的缺口可能将一条根切成多个连通部分，一座额外的前景桥也可能合并相邻根。这些变化对骨架图的影响，可能大于其对整幅图像重叠指标的影响。近期比较表明，分割性能的排序未必与根长或根径测量的排序一致[18–20]。clDice等骨架敏感损失在分割阶段关注连通性，但其对后续对象重建的实际作用仍取决于图像场景和提取过程[21, 22]。因此，有效评价应追踪分割变化如何传递到后续测量，并同时考察像素和根对象。

本研究通过轮廓监督的重建流程，将大豆萌发袋图像中的像素恢复与根对象测量联系起来，主要贡献有三方面。第一，四配置配对比较量化了分割变化如何传递到保持不变的四描述量算子。第二，有序根轴续接与整候选质量学习将已有多边形身份和全局边归属、根数约束结合。第三，像素、路径及对象的互补评价保留全部图像，并同时考察返回根轴及未解决接触，使计数改善能够与结构性错误共同判断。

## 2. 材料与方法

### 2.1. 总体流程

图1概括了图像准备、分割、根路径重建与评价。受控分割分析在切块前按完整图像分折，其输出进入共同参照提取器，以考察掩膜差异的传递。后续重建分析采用另行开发并固定的三通道分割流程，在相同输入上比较候选选择方法。原始整根多边形提供对象级监督和计算路径参照。两项分析关注流程的不同阶段，分别评价。

![FIG1](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG1.png)

**图1.** 分割与根对象重建流程。上部展示图像准备与四配置分割对照；另一组固定三通道输入用于重建分析。有序根轴、候选质量预测和全局选择形成曲线及对象级输出。质量与计数森林在指定训练折内拟合；此前的开发阶段参照上限筛选使用全部50张图像的标注，判断候选池是否具备继续拟合的可行性。

### 2.2. 图像与根系标注

数据集包含50张未经处理的Wm82大豆幼苗TIFF图像，拍摄于萌发后约7天。每幅图像为3209×2311像素（高×宽）。配套LabelMe文件以多边形圈出可见根区域。数据提供者确认每个多边形代表一整条独立根，全图集共3290个根对象。根前景占各图像的0.41%–8.22%。文件哈希与像素比较未发现重复源图像。

同一份标注生成两种表示。语义分割将所有根多边形合并为一个前景掩膜，根和背景灰度值分别为255和0。对象级重建保留每个多边形在图像内的独立身份，并从各多边形提取单独的计算中心线用于路径比较。全部3290个对象均保留在评价分母中，包括两个无法导出可用参照曲线的对象。多边形参照描述整根区域，但不包含根序及母子连接标签。图2展示合并掩膜与独立根对象的区别。

![FIG2](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG2.png)

**图2.** 10-1-9(3)的图像与标注表示。该图含33个整根多边形，也用于图4及图8a、d。(a)原始照片；(b)分割所用的合并根掩膜；(c)各多边形派生的独立计算中心线，颜色仅区分对象。面板共享图像坐标与尺度，比例尺为500像素。

### 2.3. 分割

#### 2.3.1. 图像分折与切块准备

将50组图像与掩膜分为五折，每折十张。按标注前景比例排序，划分为十个含五张图像的区块，再以划分种子20260920在各区块内随机分配。第k次轮转使用第k折测试、第(k+1) mod 5折验证，其余30张图像提供训练图块。同一图像的全部图块保持相同分折归属。

受控对照采用灰度输入，并以对比度受限自适应直方图均衡化增强（CLAHE，裁剪限制2.0，16×16网格）。训练图块为256×256像素，步长128像素。保留前景像素超过150的图块；含31–150个前景像素的图块以0.5概率保留；前景像素不超过30的图块排除。在线增强包括直角旋转、独立水平与垂直翻转，以及以0.5概率执行的gamma变换，指数范围0.8–1.2。训练种子为42、3407和2026。

完整图像预测时，增强图像归一化至[0,1]，在下边和右边反射填充。全部256×256像素图块按相同步长参与推理。二维高斯权重窗降低图块边缘在融合中的权重。中心μ=128像素、标准差σ=64像素，权重窗定义如下：


$$G_{raw}(x,y)=\exp\left[-\frac{(x-\mu)^2+(y-\mu)^2}{2\sigma^2}\right]\tag{1}$$


$$G(x,y)=0.1+0.9\frac{G_{raw}(x,y)-\min G_{raw}}{\max G_{raw}-\min G_{raw}+\epsilon}\tag{2}$$


其中ε=10⁻⁸，第二式将原始权重窗重新缩放，使边界保留非零权重。每个像素的融合概率为图块预测加权和除以权重和加ε。融合后移除填充区域。

#### 2.3.2. 网络与损失配置

U-Net与Attention U-Net共用编码器–解码器实现[23, 24]。每个卷积块包含两个不带偏置的3×3卷积，随后接批归一化与ReLU。编码器通道数为32、64、128、256，瓶颈为512通道。四级解码器采用双线性上采样，输出通道数依次为256、128、64、32。1×1卷积与sigmoid产生前景概率。受控对照中的单通道Attention U-Net包含7,981,277个可训练参数。

每一级解码器的注意力门将深层门控信号g与编码器特征x结合：


$$q=\operatorname{ReLU}[\operatorname{BN}(W_g g)+\operatorname{BN}(W_x x)]\tag{3}$$


$$\alpha=\sigma[\operatorname{BN}(W_\psi q)]\tag{4}$$


$$\hat{x}=x\odot\alpha\tag{5}$$


其中Wg、Wx、Wψ为学习投影，BN为批归一化，σ为sigmoid，⊙表示逐元素乘法。从深到浅，各注意力门中间通道数为128、64、32、16。图3展示共同网络结构，并标明后续三通道重建流程的输入通道变化。

![FIG3](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG3.png)

**图3.** Attention U-Net结构。编码器特征先经过注意力门，再与对应解码器特征拼接。受控对照采用灰度输入；后续重建输入为三通道，编码器–解码器拓扑不变。检查点及阈值选择与网络结构分别说明。

四种受控配置为像素损失U-Net、像素损失Attention U-Net、固定骨架损失Attention U-Net以及两阶段损失Attention U-Net。像素损失由0.2倍二元交叉熵和0.8倍软Dice损失组成。Dice损失在批次内计算，平滑常数为1：


$$L_{Dice}=1-\frac{2\sum_iP_iY_i+1}{\sum_iP_i+\sum_iY_i+1}\tag{6}$$


固定骨架损失配置在整个训练期间使用下列系数：


$$L_1=0.2L_{BCE}+0.8L_{Dice}+0.3L_{skel}\tag{7}$$


Soft-clDice采用12次经修正的软骨架化迭代，通过预测与目标骨架计算拓扑精确率和敏感度[21]。两阶段配置在前25轮使用式（7），随后改为：


$$L_2=0.4L_{BCE}+0.8L_{Dice}+0.3L_{skel}\tag{8}$$


四种配置均采用AdamW，初始学习率10⁻³、权重衰减10⁻⁴，余弦退火至10⁻⁵，批大小12，共训练50轮。保留在0.50阈值下取得最高验证图像平均Dice的检查点。对该检查点，在0.10–0.90范围以0.01步长选择使验证图像平均Dice最大的阈值；完全相同时选较低阈值。上述选择固定后再评价测试图像。

#### 2.3.3. 用于重建的固定三通道输入

根对象重建采用三通道Attention U-Net流程D3保存的折外预测。三个通道为CLAHE增强灰度、Lab b通道和灰度绝对局部对比。最后一个通道为σ=1与σ=12高斯平滑图像之差的绝对值，乘以8后裁剪至8位范围。D3由对应折经验证选择的D1检查点初始化。D1是原始种子42固定骨架损失模型的灰度续训版本，采用20个采样轮次，每轮1024个图块、批大小12，AdamW学习率按余弦由10⁻⁴降至10⁻⁵。其增强和检查点选择设置见附加文件1。首层卷积复制D1灰度权重，并将额外通道权重置零。

D3使用种子42进行八个采样轮次，每轮512个256×256像素图块，批大小12。半数图块由可用训练图像根轴偏好引导，坐标抖动±80像素；其余从根区域采样，抖动±112像素。像素目标仍为原始合并根掩膜。训练采用式（7），输入卷积初始学习率10⁻⁴，其余网络为2×10⁻⁵，并按余弦降至各自初值的10%。第0、4、8轮检查点及概率阈值由验证折选择。确定性图像朝向由灰度水平强度矩的符号决定，相同时再比较SHA-256；所有输入通道同步变换，预测还原至原坐标。全部重建比较均固定这些已保存掩膜与概率。由于输入及训练历史不同，D3重建实验与受控分割对照分别分析。

### 2.4. 几何参照流程

共同确定性提取器分别处理四种受控配置的标注及预测掩膜[25]。第一通道直接骨架化，不进行闭运算或剪枝。选择垂直跨度最大的八连通分量，跨度相同时按分量大小及顺序确定；其全部骨架像素按行、列的字典序排序，首尾像素作为搜索起点和终点，不要求图度数为1。Dijkstra搜索的正交步权为1像素，斜步为√2像素。路径描述量为：


$$L_{path}=\sum_{e\in P}w_e\tag{9}$$


少于十个骨架像素的分量及不可用路径记为缺失。此通道测量所选连通分量内的一条优势路径。

另一通道依次进行5×5像素椭圆闭运算、骨架化，以及15次同时删除端点。八邻域内至少有三个邻居的像素形成交点候选：


$$J=S\odot\mathbf1[K_8*S\geq3]\tag{10}$$


其中K8统计八邻居。交点候选像素的八连通组定义为交点区域。按相同加权路径规则重建根轴后，移除根轴与交点像素。剩余至少含十个像素的分量定义保留根段数：


$$N_{seg}=\left|\{C_k:|C_k|\geq10\}\right|\tag{11}$$


每个保留分量中距有序根轴最近的像素定义为根段基点。根段向量终点取分量数组中第一个距基点8–15像素的像素；若不存在，则取数组末像素。根轴向量向下游延伸12个路径步，或延伸至端点。角度计算为：


$$\theta=\arccos\left(\frac{v_p\cdot v_l}{\|v_p\|\|v_l\|}\right)\frac{180^\circ}{\pi}\tag{12}$$


将归一化点积限制在[−1,1]，大于90°的角折叠为对应锐角，并对有效值取平均。四个参照描述量为优势路径长度、保留根段数、交点区域数及平均局部锐角。比较中保持此定义不变。参数敏感性覆盖36种组合：闭运算宽度3、5、7像素；端点删除5、10、15、20次；最小分量大小5、10、20像素。

### 2.5. 有序根轴与整根候选学习

#### 2.5.1. 根轴续接与曲线测量

已有入射方向选择器为每个固定D3输入提供有序优势根轴候选。其早期开发所用根轴偏好，来自对机器生成路径的选择。重建流程保留已选根轴，并考察固定较宽松前景支持掩膜中的六条已保存替代路径。续接须与根轴末端max(64像素,12个末端半径)范围按序重合，存在前景连接，初始方向一致，尾段长度足够，且不重复旧像素。仅追加满足条件的尾段，原根轴逐点保留。

按约一像素弧长间隔重采样有序路径，使用七样本窗口、二阶Savitzky–Golay滤波平滑，随后恢复首尾坐标。曲线长度为相邻平滑位置间欧氏距离之和：


$$L_{curve}=\sum_{k=1}^{K-1}\|\widetilde{x}_{k+1}-\widetilde{x}_k\|_2\tag{13}$$


其中x̃k为第k个有序重采样位置，K为位置总数。另外保留八邻域栅格弧长。此流程按路径顺序而非图像最低位置确定端点，因此计入所选路径中向上返回的部分。算法1概括该过程。

![ALG1](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/ALG1.png)

**算法1.** 有序根轴续接与曲线测量。将满足支持条件的尾段追加到原根轴，保留已有像素。最终有序曲线提供像素空间长度估计。

#### 2.5.2. 完整路径候选及特征

将固定D3掩膜转换为骨架图，其中边表示端点或接触区域之间的根段。通过比较二值掩膜及其水平镜像的SHA-256，确定统一处理朝向；图像特征提取前将路径还原至原图坐标。四种非学习式接触配对解释采用不同的固定方向与宽度连续性组合。汇集其完整轨迹与单独图边，并去除重复边集。保留的优势根轴作为受保护候选加入；至少一半像素位于受保护根轴2像素以内的图边分配给该根轴，含任一此类边的其他候选予以排除。50张图像共产生19,914条路径候选。

质量学习之前，利用标注计算完整候选池的参照上限。既有开发协议规定，75%双向支持下的一对一匹配至少达到645条参照才继续拟合；该候选池在全部50图上得到1158个匹配。此计算使用全部参照标签，且不执行边归属和根数约束；它是开发可行性筛选及该候选池的上限，不是可部署预测。

每个候选包含18项几何及12项图像特征。几何特征概括长度、位移、空间范围、半径分布、曲率、边覆盖与接触间隙。图像特征概括路径灰度与Lab值、路径与两侧的局部对比，以及沿有序曲线的变化。完整特征定义与候选生成参数随数值包提供。图4展示原图、图候选与联合选择路径之间的关系。

![FIG4](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG4.png)

**图4.** 固定D3掩膜上10-1-9(3)的根路径重建，该图属于第4折，含33个参照多边形。(a)全部229个保存的完整路径候选；(b)主要方法选出的49个对象，颜色仅区分对象，其中11条参照达到75%双向支持匹配；(c)受保护根轴（蓝）及其余选中路径（灰）。分配给受保护根轴的图边不得被其他路径占用。显示使用保存的图像与路径坐标，不重画根的几何形状。

#### 2.5.3. 轮廓衍生目标与联合选择

候选质量由独立多边形身份监督。对每个候选–参照对，预测支持Pij为候选弧长落在多边形及其两像素邻域内的比例；参照覆盖Rij为多边形计算中心线落在候选邻域内的弧长比例。后者距离容差为max(2像素,参照曲线中位根半径+1像素)。完整根目标为max_i min(Pij,Rij)。碎段替代目标为最佳双向支持参照i*对应的Pi*j(1−Ri*j)，混合替代目标为各参照候选支持中最大的两个值的较小者。

采用多输出Extra-Trees回归器，以几何特征或全部30项特征预测上述三个目标[26]。每个回归器含128棵树，最大深度12、最小叶节点样本数6、特征比例0.75，随机状态20261019。每幅训练图像中候选权重之和为1。另一个图像级Extra-Trees回归器根据掩膜及图特征预测根数，包含128棵树、最大深度6、最小叶节点样本数3，采用绝对误差划分及随机状态20261019；计数预测值小于1时取1。五个图像折轮转中，每折分别拟合几何质量、图像加几何质量及计数森林，总计15次拟合。每次拟合均排除对应测试折的全部十张图像。

对候选j，解码器结合预测完整性q̂j、碎段f̂j、混合m̂j与归一化图边覆盖aj。二元变量zj表示是否选中，n̂为图像级预测根数。目标为：


$$\max_{z}\ \sum_jz_j(2\hat q_j-1-\hat f_j-\hat m_j+a_j)-\left|\sum_jz_j-\hat n\right|,\quad\sum_{j:e\in E_j}z_j\leq1,\ z_0=1\tag{14}$$


约束要求每条图边至多出现在一个选中候选中，并将受保护根轴变量固定为1。覆盖aj为候选图边总长度除以max(1,ℓ̄)，其中ℓ̄为图边总长除以max(1,n̂)。两个非负松弛变量表示根数绝对偏差。SciPy混合整数优化的时间上限为八秒，相对间隙容差10⁻⁶[27]。解码不接收测试图像参照身份与根数。

主要方法采用图像加几何质量与根数惩罚。比较方法包括此前固定配对输出、使用同一学习计数先验的非学习式几何候选评分、几何质量学习，以及去除显式根数偏差惩罚的图像加几何学习。去除该惩罚后，覆盖归一化仍使用预测根数。非学习几何评分取q̂j = ℓj/(ℓj + s)，其中ℓj为候选路径长度，s = max(10像素, 该图候选长度中位数)，并令f̂j = m̂j = 0。各指定比较内采用相同固定输入。算法2展示训练与选择顺序。

![ALG2](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/ALG2.png)

**算法2.** 候选池筛选后的轮廓监督整根候选质量与全局选择。训练图像提供目标值及计数监督。在此分折拟合和解码流程中，留出图像只提供特征及固定路径候选，其参照身份与根数在解码后参与评价；此前的参照上限筛选使用了整个开发图集。

### 2.6. 评价与统计分析

#### 2.6.1. 分割与几何一致性

根据前景真阳性TP、假阳性FP和假阴性FN计算精确率、召回率、Dice及交并比：


$$\operatorname{Precision}=\frac{TP}{TP+FP}\tag{15}$$


$$\operatorname{Recall}=\frac{TP}{TP+FN}\tag{16}$$


$$F_1=\frac{2TP}{2TP+FP+FN}\tag{17}$$


$$\operatorname{IoU}=\frac{TP}{TP+FP+FN}\tag{18}$$


每个种子先对50张留出图像的完整图像分数取平均，再计算三个种子的均值及样本标准差。硬clDice采用二值细化掩膜。细根恢复定义为预测前景对标注骨架像素的精确覆盖，按标注宽度≤3、>3–6、>6–12及>12像素分层。宽度为至背景局部距离的两倍；图像中不存在的层单独记录。

预测与标注掩膜进入同一参照提取器。描述量一致性采用平均绝对误差（MAE）、均方根误差、偏差、非零参照的平均绝对百分比误差（MAPE），以及Lin一致性相关系数（CCC）概括[28]。图像均值差值图（附加文件1）将三个种子预测先取平均，描述性界限为偏差±1.96倍样本标准差[29]。

各区间流程对应其统计单位。分割差值独立重抽三个种子索引及50个图像索引10,000次，并在二者笛卡尔积上保持配置配对，种子为20260920。描述量差值先在图像内平均种子，再在五个固定折内分别重抽图像，10,000次，种子20260925。部署Dice重抽50个留出图像分数10,000次，种子20260926。报告的95%描述性区间取对应重抽估计值的第2.5及97.5百分位数。这些区间描述已记录图集与预测，不作假设检验显著性声明。

#### 2.6.2. 整根重建

通过参照与输出间的一对一匹配评价整根路径。当预测支持与参照覆盖均至少为t时，路径对满足阈值t。匈牙利分配对每个满足阈值的配对赋予max(参照数, 输出数)+1的奖励，并累加全部已分配配对的min(Pij, Rij)；分配后剔除低于阈值的配对。因此，首先最大化满足阈值的配对数，再以整个分配的相似度之和作为二级准则。在t=0.50、0.75及0.90评价，主要比较使用t=0.75。匹配比例分别以全部3290条参照根及输出对象数为分母。

每幅图像根数误差为输出对象数与多边形数之差的绝对值。碎段参照诊断识别由至少两条部分输出支持的参照根，条件为候选支持≥0.75、参照覆盖≥0.10。混合输出诊断识别同时受到至少两个参照明显支持的输出，条件为候选支持≥0.20、参照覆盖≥0.10。碎段比例以参照数为分母，混合比例以输出数为分母。高支持不完整输出指其最佳参照（min(Pij, Rij)最大）满足Pij≥0.75但Rij<0.75的输出。这些是轮廓诊断，并非独立解剖判定。

重建比较保留全部50张图像。重建学习器排除被评价图像所在折，但上游D3掩膜与已选根轴仍为固定开发输入，候选池可行性也已使用全部参照标签筛选。因此，该分析评价有条件的内部重建表现，并非重新嵌套检验整套图像至根流程。评价后的配对区间在固定折内重抽图像10,000次，种子20261019。保留逐图结果、消融比较与各折变化。

#### 2.6.3. 实现

RootScope Desktop提供原始分割与描述量流程的参照实现。其五个种子42 ONNX模型在相应留出图像上使用验证选择阈值评价。数值核验比较研究预测与部署输出。CPU计时采用已有150次图像运行，即50张图像各重复三次；硬件为AMD Ryzen 7 7840H，八个物理核，Windows 11，ONNX Runtime CPUExecutionProvider，八个算子内线程。计时包括解码、CLAHE、切块推理融合、阈值化及参照提取器；不包括模型加载、预热、参照评价与写盘。阶段分位数及总耗时分位数分别计算。

重建原型与打包参照应用分开维护。分析由已保存候选、解码及逐图评价记录重现。附加文件2包含源表、脚本、模型版本标识及文件哈希。附加文件1提供完整敏感性与开发对照，包括后续末端回退实验。

## 3. 结果

### 3.1. 受控分割对照

60次受控模型拟合全部完成，产生600份留出图像预测。固定骨架损失Attention U-Net在四配置中具有最高平均Dice、IoU、硬clDice和≤3像素中心线召回（表1）。其平均Dice为58.43%±0.46个百分点；像素损失Attention U-Net及U-Net分别为55.90%±0.61和55.04%±1.08。两阶段配置为57.54%±2.81，种子间变异较大。

**表1.** 共同受控协议下的留出分割表现。数值为三个种子图像均值的均值±样本标准差；各种子均值均使用相同50张源图像。

| 配置 | Dice（%） | IoU（%） | 硬clDice（%） | ≤3像素中心线召回（%） |
|---|---|---|---|---|
| U-Net，像素损失 | 55.04 ± 1.08 | 38.21 ± 0.96 | 62.07 ± 4.55 | 58.83 ± 1.37 |
| Attention U-Net，像素损失 | 55.90 ± 0.61 | 38.99 ± 0.61 | 64.63 ± 1.52 | 59.03 ± 0.41 |
| Attention U-Net，固定骨架损失 | 58.43 ± 0.46 | 41.49 ± 0.43 | 67.43 ± 0.67 | 65.64 ± 1.17 |
| Attention U-Net，两阶段损失 | 57.54 ± 2.81 | 40.65 ± 2.74 | 65.86 ± 3.28 | 63.48 ± 8.80 |

相对U-Net，仅增加注意力门使配对平均Dice变化0.86个百分点，描述性区间为−0.68至2.85个百分点。相对像素损失Attention U-Net，固定骨架损失使配对平均Dice提高2.53个百分点，区间为1.37–3.40个百分点，五折均值均为正。两阶段相对固定权重的差值为−0.89个百分点，区间−3.48至1.58，仅一折均值为正。图5展示各个种子的完整图像均值及配对差值。

![FIG5](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG5.png)

**图5.** 受控分割结果。(a)折外宏平均Dice，符号表示训练种子：圆42、方3407、三角2026，并显示三个种子值的均值±标准差。(b)逐图配对Dice差值，单位为百分点：Attention为像素损失Attention U-Net减U-Net，+fixed clDice为固定骨架损失减像素损失，+two-stage为两阶段减固定骨架损失。点为三种子图像均值，竖线覆盖种子值范围，黑横线为50图均值。(c)折与种子组合的均值以空心符号表示，短横线为折均值，长横线为总体均值，标签给出均值为正的折数。Fixed clDice对应式(7)的固定骨架损失配置。

固定骨架损失在≤3、>3–6、>6–12及>12像素根宽层的精确中心线召回依次为65.64%、72.44%、74.89%及75.77%。对应原图与错误图裁剪展示细根恢复差异（图6）。15次两阶段运行中14次选择不晚于第25轮的检查点，一次选择第27轮。因此，大多数入选两阶段预测来自系数切换之前。

![FIG6](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG6.png)

**图6.** 按根宽分层的中心线恢复。(a)精确中心线召回按参照宽度分层；宽度为参照骨架像素处距离变换值的两倍。线为三个种子各自50图均值的平均，色带为种子间±1个标准差。(b、c)在至少含750个≤3像素参照骨架点的图像中，选取整图该层召回最低与最高的种子42固定骨架损失预测；百分比为整图值。384×384像素裁区分别使漏检(b)或恢复(c)的细根参照像素最多，用于展示对比情形，并不代表典型表现。左列为灰度原图，按第1至99百分位拉伸显示；中列为参照细中心线；右列为预测错误。比例尺为100像素。

### 3.2. 向参照描述量的误差传递

四个参照描述量在每个种子中均有50对可评价图像。固定骨架损失的优势路径MAE为1210.52像素，平均偏差为−1137.20像素（表2）。保留根段及候选交点数量平均分别比标注掩膜输出高95.47与109.85。局部锐角MAE为6.84°，表示图像级计算角度汇总的一致性。

**表2.** 同一参照提取器下固定骨架损失预测与标注掩膜的一致性。MAE与偏差保留描述量单位；CCC与MAPE汇总三个种子分别计算的结果。角度为计算局部角描述量。

| 描述量 | 单位 | MAE ± SD | 平均偏差 | CCC ± SD | MAPE（%） |
|---|---|---|---|---|---|
| 优势路径长度 | 像素 | 1210.52 ± 56.26 | −1137.20 | 0.321 ± 0.047 | 41.2 |
| 保留根段数 | 个 | 97.43 ± 3.98 | +95.47 | 0.575 ± 0.023 | 100.5 |
| 候选交点区域数 | 个 | 111.71 ± 5.32 | +109.85 | 0.430 ± 0.034 | 147.8 |
| 平均局部锐角 | 度 | 6.84 ± 0.42 | −3.99 | 0.261 ± 0.097 | 12.1 |

分割重叠与描述量误差的配置排序不同（表3）。相对像素损失Attention U-Net，固定骨架损失使保留根段及交点区域MAE分别增加35.03与50.05。三个种子及五个折均值中均观察到增加，对应描述性区间为25.89–44.75个根段及37.75–63.03个交点区域。因此，配对描述量比较揭示了Dice本身未显示的取舍。

对应路径长度MAE差值为−32.93像素，描述性区间−117.53至48.27；探索性局部角MAE差值为−1.18°，区间−1.83至−0.55°。因此，各描述量的变化方向及一致性不同：两个计数误差增加，角度汇总相对同一计算算子的绝对误差降低，而路径误差区间同时包含两个方向的变化。

**表3.** 受控分割配置的描述量MAE。各单元为三个种子各自50图MAE的均值±样本标准差。参照与预测均使用相同且保持不变的参照算子。

| 配置 | 路径（像素） | 根段（个） | 交点（个） | 角度（度） |
|---|---|---|---|---|
| U-Net，像素损失 | 1258.8 ± 243.2 | 68.2 ± 26.7 | 62.1 ± 28.0 | 8.0 ± 1.5 |
| Attention U-Net，像素损失 | 1243.5 ± 152.1 | 62.4 ± 16.1 | 61.7 ± 23.8 | 8.0 ± 0.9 |
| Attention U-Net，固定骨架损失 | 1210.5 ± 56.3 | 97.4 ± 4.0 | 111.7 ± 5.3 | 6.8 ± 0.4 |
| Attention U-Net，两阶段损失 | 1200.2 ± 130.6 | 107.4 ± 12.1 | 126.1 ± 19.1 | 7.3 ± 0.3 |

提取器设置也改变测量结果。在36组参数中，平均标注保留根段数范围为111.96–165.74，默认值为144.72；根段MAE范围为45.00–217.21。附加文件1提供四个描述量的完整一致性与敏感性结果。构造的完整掩膜示例复现了已知45°分枝输出54.16°及返回路径被量短的现象，说明提取规则可独立于分割引入误差。

### 3.3. 整根候选学习与选择

在不执行边冲突和根数约束时，标注引导的候选池参照上限在75%双向支持下匹配1158/3290条参照根，占35.20%。主要重建方法保留3342个输出对象，固定配对为6440个（表4）。一对一匹配由586增至682，占参照根的17.81%与20.73%。参与匹配的输出比例由9.10%升至20.41%。图像级根数MAE由63.00降至21.36。

**表4.** 相同固定D3输入上的整根重建。匹配与碎段比例使用3290条参照根为分母；混合输出比例使用各方法输出数为分母。主要方法为采用全局选择与根数偏差惩罚的图像加几何整根学习。

| 方法 | 输出对象 | 匹配参照，n（%） | 根数MAE | 碎段参照，n（%） | 混合输出，n（%） |
|---|---|---|---|---|---|
| 固定配对 | 6440 | 586 (17.81) | 63.00 | 258 (7.84) | 674 (10.47) |
| 几何评分及计数先验 | 3453 | 518 (15.74) | 22.74 | 185 (5.62) | 666 (19.29) |
| 整根几何学习 | 3334 | 675 (20.52) | 21.20 | 152 (4.62) | 572 (17.16) |
| 图像学习，无计数惩罚 | 1571 | 365 (11.09) | 35.22 | 55 (1.67) | 424 (26.99) |
| 整根图像学习（主要） | 3342 | 682 (20.73) | 21.36 | 154 (4.68) | 562 (16.82) |

被切碎的参照根由258减至154，占参照根的7.84%及4.68%。混合输出由674减至562，但由于保留对象减少，其比例由10.47%增至16.82%。高支持不完整输出由1690减至872，在两种方法中的输出占比均约26%。因此，计数与完整路径匹配改善，并不意味着保留对象中每类错误的发生频率同步下降。

45幅图像根数误差改善，五幅恶化。35幅图像路径匹配增加，五幅不变，十幅减少。五折的根数MAE均下降。第1–4折匹配增加（129至171、79至111、111至123、100至117）；第0折根数MAE由63.70降至17.30，但匹配由167降至160。逐图平均配对变化为匹配+1.92（描述性区间0.96–2.94），绝对根数误差−41.64（−53.92至−30.54）。图7展示全部逐图变化，包括退步。

![FIG7](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG7.png)

**图7.** 相同固定D3输入上的整根重建。A为固定配对，B为含学习计数先验的非学习几何评分，C为整根几何学习，D为无根数偏差惩罚的图像加几何学习，E为带该惩罚的图像加几何学习（主要方法）。(a)75%双向支持的一对一匹配参照数；(b)根数MAE；(c)碎段参照占3290条参照的比例；(d)混合输出占各方法输出的比例，分母A至E依次为6440、3453、3334、1571、3342；(e、f)全部50图按文件名字母顺序排列的E减A逐图差值，点形区分图像折，(e)中的负值表示退步。

消融区分了根数与选路改善。使用同一计数先验的非学习几何评分将根数MAE降至22.74，但仅有518个匹配，低于固定配对。学习几何质量得到675个匹配、MAE 21.20；加入图像特征后为682与21.36。去除显式根数偏差惩罚仅保留1571个对象、365个匹配。因此，主要改善来自整候选质量与联合选择的结合，本次比较中图像特征的增量贡献较小。

### 3.4. 返回根轴与图像级示例

有序续接恢复了10-1-9(3)中一段可见的向上返回路径，使曲线估计由4077.01增至4982.83像素。235-25-7(1)中原本完整的U形根轴保留5028.08像素长度。全部整根选择方法随后均逐点保留这些根轴，使曲线续接与其余根对象的选择分开处理。

235-25-9(1)中的续接仍不完整：路线在底部附近的一条短分枝停止，一段可见返回未被追踪，其曲线估计保持3134.07像素。图8使用已保存路径对比补回的返回、完整U形和未解决情形。全部50图上的固定末端回退对照增加了路径弧长，但引入局部绕路，因此未采用；完整结果见附加文件1。

![FIG8](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/download/paper-revision-20261020/FIG8.png)

**图8.** 已保存路径中的返回根示例。上排展示输入有序根轴，下排在同一照片上展示保留或续接后的根轴。各列分别为10-1-9(3)中补回的返回、235-25-7(1)中保留的U形及235-25-9(1)中未解决的返回。蓝色表示保留的输入路径，橙色表示追加尾段，圆点与方块分别标示路径起点和终点。各面板共享原图尺度，比例尺500像素。曲线估计以像素计；示例展示重建行为，不估计独立成功率。

### 3.5. 参照实现

原始种子42 ONNX流程在50张对应留出图像上的图像级宏平均Dice为58.10%，描述性图像bootstrap区间56.63%–59.52%。与PyTorch研究流程相比，完整图像Dice及IoU的最大差分别为0–1尺度上的0.00046及0.00043。150次既有CPU运行的单图总耗时中位数52.26秒，四分位范围49.05–57.41秒，P90为66.55秒。附加文件1给出各阶段分位数。上述测量描述参照部署流程，重建原型另行评价。

## 4. 讨论

### 4.1. 同时评价像素与重建根对象

受控对照显示，固定骨架损失改善掩膜重叠及细标注中心线的精确恢复，但在共同几何算子下增加了根段与交点计数误差。这些结果关注不同图像表示。像素重叠概括局部分类，图测量则依赖前景连接方式。因此，短缺口或额外桥接可改变对象结构，却不一定主导Dice。这与多数据集的分割及根表型排序比较形成互补[18]，支持按模型实际用途评价下游量[20]。

两阶段对照也说明模型选择行为的重要性。大多数入选检查点早于权重切换，后阶段同时改变相对权重与总损失尺度。因此，现有对照不能确立后阶段的独立收益。固定骨架损失在本图集分割终点上表现较好，但选择其作为部署参照不代表它对全部几何描述量均最优。

### 4.2. 整根监督与数量–结构取舍

保留独立多边形身份，使完整候选而非仅局部接触的性质能够被评价和学习。学习目标惩罚不完整支持及跨根区域的歧义，全局选择阻止多个选中候选同时拥有同一图边。匹配与计数误差的改善表明，在固定输入下这种联合表示具有作用。几何评分对照提供了重要信息：根数MAE接近学习结果，但匹配低于固定配对。因此，输出数量合理不足以证明独立路径正确。

候选可用性也限制了重建。即便使用标注在完整候选池上匹配，在执行解码器边冲突或根数惩罚之前，75%双向支持下也仅匹配1158条参照。在这一固定池内学习更好的选择器，无法恢复池中未被充分表示的路径。因此，剩余差距同时涉及候选构建与候选排序。

混合输出的绝对数量与相对比例提供互补信息。主要方法去除了许多多余对象，但保留输出中混合路径占比更高。同样，不完整对象总数下降并未消除其较高输出比例。因此，应同时报告对象数、参照覆盖和结构性诊断。几何与图像加几何学习的差异较小，也说明不应将全部改善归因于颜色信息。

### 4.3. 曲线、接触与根轴保留

将根表示为有序曲线，可将长度与图像高度分开。路线向上返回仍会累积弧长，保留此前坐标能够避免后续候选选择切断完整U形。返回根示例展示了这一实际作用。但保护也会保留输入路线中未解决的部分，235-25-9(1)的不完整根轴即为例证。

局部失败不能仅以曲率解释。在拥挤接触处，更平顺或更粗的续接可能进入相邻根，较宽松前景也可能保留错误连接。末端回退实验说明，允许撤销较早决策可补回额外弧长，却不能据此确定续接身份。更完整的构型需要根序、端点和附着位置之间的证据，如RootNav 2.0及标准化结构描述所体现的内容[12, 30]。RootNav、SmartRoot和RhizoVision Explorer的重建或测量定义也不同，比较时须对齐目标输出[10, 12, 31, 32]。

### 4.4. 实际应用

实际筛查时，结果支持在数值之外保留明确路线及缺失信息。接触或微弱前景使重建不确定时，应结合叠加图解释根长与对象数。软件版本和可复现数值记录便于检查；更广泛的表型应用仍需针对目标生物量验证[9, 20, 33–37]。两项配对分析为像素、对象及曲线输出在作为生物测量使用之前的检验提供了明确基础。

### 4.5. 局限

本研究采用同一大豆基因型的50张开发图像，未核实植株与采集批次分组。重建分析复用此图集、单种子D3输入及固定根轴，且候选池可行性在分折学习前使用全部参照标签筛选。因此，它提供有条件的内部证据，而非新样本验证。多边形中心线及结构性诊断属于计算参照，标注不独立确立解剖根序、附着位置或发生角。缺少物理标定时，长度为二维像素空间估计。上述条件限制生物学及跨条件解释，但支持所述采集场景内的可复现比较。

## 5. 结论

固定骨架损失改善了受控大豆萌发袋对照中的分割重叠，但相应变化未一致降低参照描述量误差。在固定开发输入上，整根多边形监督、完整路径质量学习与全局选择使匹配由586增至682，图像级根数MAE由63.00降至21.36。有序曲线测量保留了已选返回根轴，而混合输出比例及未解决接触界定了剩余重建错误。像素、对象与曲线的联合评价，为考察根系图像分析的不同阶段提供了可行框架。

## 缩略语

AG，注意力门；BCE，二元交叉熵；BN，批归一化；CCC，一致性相关系数；CLAHE，对比度受限自适应直方图均衡化；CPU，中央处理器；CSV，逗号分隔值；IoU，交并比；IQR，四分位范围；MAE，平均绝对误差；MAPE，平均绝对百分比误差；ONNX，开放神经网络交换格式；OOF，折外；P90，第90百分位数；pp，百分点；RMSE，均方根误差；SD，标准差；SHA-256，256位安全散列算法。

## 声明

### 数据与材料可用性

支持本文结论的数据集存放于带固定版本的GitHub研究资料集[38]，包括50张原始TIFF照片、LabelMe标注及合并参照掩膜，以及冻结的分割与重建记录、研究代码和选定模型。文件清单记录图像分折、多边形对象数量及SHA-256。大型原图、模型及源码归档作为独立仓库资源提供；附加文件2、3提供无需下载照片或模型权重即可运行的数值与合成几何重放。

原始RootScope Desktop实现及其五个ONNX模型保留历史软件版本；重建原型单独标识，不替代打包应用。源码采用Python，参照桌面包在Windows 11上评价，分析依赖与复现命令随相应代码快照提供。软件沿用仓库的MIT许可；原图和标注保留提供者版权，已授权公开访问，复用条款载于配套数据许可文件。

[Repository](https://github.com/goldriderhszz-hash/root-phenotyping-attunet) · [Historical desktop software](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/rootscope-desktop) · [Historical source archive](https://archive.softwareheritage.org/swh:1:rev:c9103a920e84e69ee6c2280a9cc75b4830f5a286/)

### 利益冲突

作者声明不存在利益冲突。

### 经费支持

本工作获得中华人民共和国农业农村部资助的国家重点研发计划项目（2025YFD2300401）、黑龙江省揭榜挂帅项目“智能耕作决策平台关键技术研究与应用”（2021ZXJ05A03）、黑龙江省科技发展项目（2024ZX01A07），以及黑龙江省重点研发计划创新基地项目（JD2023GJ01-01）支持。

### 作者贡献

Z.W.提出方法、开发算法并撰写正文。J.T.、H.Z.和F.L.开展萌发袋培养实验，采集高分辨率根系图像并整理数据集。T.S.、Z.Z.、Y.L.和Q.C.协助数据整理、人工标注和图件准备。R.Z.负责总体指导、经费获取及研究设计。所有作者审阅稿件。

### 伦理批准与参与同意

不适用。

### 发表同意

不适用。

### 致谢

不适用。

### AI辅助使用

OpenAI Codex和Notion AI用于辅助稿件起草、数值核验与图件准备。科学内容与最终批准由作者负责。工作流程参考Scientific Agent Skills[39]。

## 参考文献

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

38. Soybean rhizobag image dataset, study code and frozen reconstruction records. Version paper-revision-20261020. root-phenotyping-attunet, GitHub. 2026. https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/paper-revision-20261020. Accessed 20 Oct 2026.

39. Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv. 2026;2609.00065. https://doi.org/10.48550/arXiv.2609.00065
