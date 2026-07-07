import os
import random
import shutil
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import skeletonize

warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础配置与路径设置
# ==========================================
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

ORIGINAL_DATA_DIR = r"D:\萌发袋照片建模"
WORKSPACE_DIR = r"D:\Root_Workspace_Sliding_lastest"

MODEL_SAVE_DIR = os.path.join(WORKSPACE_DIR, "models")
GT_10_DIR = os.path.join(WORKSPACE_DIR, "GT_10")
PRED_10_DIR = os.path.join(WORKSPACE_DIR, "Pred_10")

for d in [MODEL_SAVE_DIR, GT_10_DIR, PRED_10_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
TRAIN_STRIDE = 128

def robust_imread(path):
    if not os.path.exists(path): return None
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except: return None

# ==========================================
# 2. 网络架构：Attention U-Net
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.d1 = DoubleConv(1, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)
        self.d4 = DoubleConv(128, 256)
        self.b = DoubleConv(256, 512)
        
        self.ag4 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.ag3 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.ag2 = AttentionGate(F_g=128, F_l=64,  F_int=32)
        self.ag1 = AttentionGate(F_g=64,  F_l=32,  F_int=16)

        self.u1 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u3 = DoubleConv(128 + 64, 64)
        self.u4 = DoubleConv(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        b = self.b(self.pool(x4))
        
        up4 = self.up(b)
        u1 = self.u1(torch.cat([up4, self.ag4(g=up4, x=x4)], dim=1))
        
        up3 = self.up(u1)
        u2 = self.u2(torch.cat([up3, self.ag3(g=up3, x=x3)], dim=1))
        
        up2 = self.up(u2)
        u3 = self.u3(torch.cat([up2, self.ag2(g=up2, x=x2)], dim=1))
        
        up1 = self.up(u3)
        u4 = self.u4(torch.cat([up1, self.ag1(g=up1, x=x1)], dim=1))
        
        return torch.sigmoid(self.out(u4))

# ==========================================
# 3. 损失函数：可微 clDice 拓扑损失
# ==========================================
def dice_loss(pred, target):
    smooth = 1.0
    inter = (pred * target).sum()
    return 1 - (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=(1,1), padding=(1,0))
    p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=(3,3), stride=(1,1), padding=(1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iters=12):
    img1 = img
    skel = torch.zeros_like(img)
    for _ in range(iters):
        eroded = soft_erode(img1)
        opened = soft_open(eroded)
        skel = torch.max(skel, eroded - opened)
        img1 = eroded
    return skel

def cldice_loss(pred, target):
    smooth = 1e-5
    skel_pred = soft_skeletonize(pred)
    skel_target = soft_skeletonize(target)
    tprec = (skel_pred * target).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_target * pred).sum() / (skel_target.sum() + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + smooth)

# 核心改进：引入 epoch 参数，实现训练中后期损失函数对假阳性噪声的动态压制
def combined_topology_loss(pred, target, epoch):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    cldice = cldice_loss(pred, target)
    
    if epoch <= 25:
        # 前25轮：聚焦低对比度侧根的连通性建立 (Recall优先)
        return 0.2 * bce + 0.8 * dice + 0.3 * cldice
    else:
        # 后25轮：加大刚性BCE权重至0.4，强力清除褶皱、水珠边缘等背景假阳性噪声 (Precision优先)
        return 0.4 * bce + 0.8 * dice + 0.3 * cldice

# ==========================================
# 4. 数据集：过滤海量黑背景 + 根系多维数据增强
# ==========================================
class PatchDataset(Dataset):
    def __init__(self, files):
        self.samples = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        
        print("正在从巨图中智能筛选高质量原分辨率切片...")
        for name in files:
            img_path = os.path.join(ORIGINAL_DATA_DIR, f"{name}.tif")
            mask_path = None
            for suffix in [f"{name}_mask.png", f"{name}_mask.tif", f"{name}.png"]:
                p = os.path.join(ORIGINAL_DATA_DIR, suffix)
                if os.path.exists(p): mask_path = p; break
            if not mask_path: continue
                
            img = robust_imread(img_path)
            mask = robust_imread(mask_path)
            if img is None or mask is None: continue
                
            img = clahe.apply(img)
            mask = (mask > 127).astype(np.float32)
            h, w = img.shape
            
            for y in range(0, h - PATCH_SIZE + 1, TRAIN_STRIDE):
                for x in range(0, w - PATCH_SIZE + 1, TRAIN_STRIDE):
                    m_p = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    pixel_sum = m_p.sum()
                    
                    if pixel_sum > 150:
                        i_p = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
                        self.samples.append((i_p.copy(), m_p.copy()))
                    elif pixel_sum > 30:
                        if random.random() > 0.5:
                            i_p = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
                            self.samples.append((i_p.copy(), m_p.copy()))
                            
        print(f"智能切片提取完毕，共捕获高质量图块 {len(self.samples)} 张！")
                        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
            
        if random.random() > 0.5: img, mask = np.fliplr(img), np.fliplr(mask)
        if random.random() > 0.5: img, mask = np.flipud(img), np.flipud(mask)
        
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.clip(np.power(img, gamma), 0.0, 1.0)
            
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
            
        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(mask).unsqueeze(0).float()

# ==========================================
# 5. 推理模块：2D 高斯加权滑窗推理
# ==========================================
def get_gaussian_window(patch_size=256, sigma=64):
    x = np.arange(patch_size)
    mean = patch_size // 2
    gauss_1d = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    gauss_2d = (gauss_2d - gauss_2d.min()) / (gauss_2d.max() - gauss_2d.min() + 1e-8)
    return (gauss_2d * 0.9 + 0.1).astype(np.float32)

def gaussian_sliding_window_predict(model, img, patch_size=256, overlap_ratio=0.5):
    h, w = img.shape
    stride = int(patch_size * (1 - overlap_ratio))
    
    pred_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    gw = get_gaussian_window(patch_size)
    
    y_coords = list(range(0, h - patch_size + 1, stride))
    if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
    x_coords = list(range(0, w - patch_size + 1, stride))
    if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)
    
    for y in y_coords:
        for x in x_coords:
            patch = img[y:y+patch_size, x:x+patch_size]
            patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred_patch = model(patch_tensor).squeeze().cpu().numpy()
            
            pred_map[y:y+patch_size, x:x+patch_size] += pred_patch * gw
            weight_map[y:y+patch_size, x:x+patch_size] += gw
            
    return pred_map / (weight_map + 1e-8)

# ==========================================
# 6. 主程序与多维定量评估
# ==========================================
if __name__ == "__main__":
    all_files = [f.replace('.tif', '') for f in os.listdir(ORIGINAL_DATA_DIR) if f.endswith('.tif') and 'mask' not in f.lower()]
    all_files = sorted(list(set(all_files)))
    random.shuffle(all_files)
    
    train_files = all_files[:40]
    test_files = all_files[40:]

    for name in test_files:
        src_mask = None
        for suffix in [f"{name}_mask.png", f"{name}_mask.tif", f"{name}.png"]:
            p = os.path.join(ORIGINAL_DATA_DIR, suffix)
            if os.path.exists(p): src_mask = p; break
        if src_mask: shutil.copy(src_mask, os.path.join(GT_10_DIR, f"{name}_mask.png"))

    # --- 1. 训练准备 (扩展至 50 轮) ---
    train_dataset = PatchDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model = AttentionUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 50  # 🌟 核心改进：提升训练轮数至 50 轮
    
    # 余弦退火 T_max 同步扩展至 50，让低学习率的收敛雕刻过程更持久、更精细
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    print(f"\n 开始 50 轮深度精细化训练 (当前设备: {DEVICE})...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_topology_loss(pred, y, epoch=epoch) # 传入当前轮数进行损失权重切换
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f} | LR: {current_lr:.6f}")
        
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "precision_boost_attention_unet.pth"))
    
    # --- 2. 高斯重叠滑窗推理 (引入防噪声动态阈值) ---
    print(f"\n启动高斯权重滑窗推理...")
    model.eval()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    for name in test_files:
        src_tif = os.path.join(ORIGINAL_DATA_DIR, f"{name}.tif")
        if not os.path.exists(src_tif): continue
            
        img_orig = robust_imread(src_tif)
        h, w = img_orig.shape
        pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
        img_padded = np.pad(img_orig, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        img_prep = clahe.apply(img_padded).astype(np.float32) / 255.0
        pred_prob = gaussian_sliding_window_predict(model, img_prep, patch_size=PATCH_SIZE, overlap_ratio=0.5)
        
        pred_prob = pred_prob[:h, :w]
        # 核心改进：将二值化阈值由 0.5 轻微收紧至 0.53，直接滤除边缘虚化噪声和反光毛刺
        pred_mask = (pred_prob > 0.53).astype(np.uint8) * 255
        
        cv2.imencode('.tif', pred_mask)[1].tofile(os.path.join(PRED_10_DIR, f"{name}.tif"))


