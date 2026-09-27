import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb  # [Added] Import wandb

# ==========================================
# 1. Dataset Definition
# ==========================================
class TactileDataset(Dataset):
    def __init__(self, data_root, transform=None, split='train', val_split=0.1):
        """
        Args:
            data_root: generated dataset root (e.g., ./training_dataset)
            transform: pytorch transforms for input image
            split: 'train' or 'val'
        """
        self.data_root = data_root
        self.transform = transform
        
        # 递归寻找所有的 color 图片作为索引
        # 目录结构: data_root/object_name/xxx_color.png
        self.samples = []
        
        # 使用 glob 查找所有子文件夹中的 _color.png
        search_pattern = os.path.join(data_root, "**", "*_color.png")
        color_files = glob.glob(search_pattern, recursive=True)
        
        # 配对其他文件
        valid_samples = []
        for c_file in color_files:
            # c_file: .../mesh_0001_color.png
            base_path = c_file.replace("_color.png", "")
            
            depth_file = base_path + "_depth.npy"
            normal_file = base_path + "_normal.npy"
            mask_file = base_path + "_mask.png"
            
            if os.path.exists(depth_file) and os.path.exists(normal_file) and os.path.exists(mask_file):
                valid_samples.append({
                    'color': c_file,
                    'depth': depth_file,
                    'normal': normal_file,
                    'mask': mask_file
                })
        
        # 划分训练集和验证集
        np.random.seed(42)
        np.random.shuffle(valid_samples)
        num_val = int(len(valid_samples) * val_split)
        
        if split == 'train':
            self.samples = valid_samples[num_val:]
        else:
            self.samples = valid_samples[:num_val]
            
        print(f"[{split}] Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load Input Image (H, W, 3) -> (3, H, W)
        img = cv2.imread(item['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
        
        # 2. Load Targets
        # Depth: (H, W) -> (1, H, W)
        depth = np.load(item['depth']).astype(np.float32)
        depth = np.expand_dims(depth, axis=0) 
        
        # Normal: (H, W, 3) -> (3, H, W)
        normal = np.load(item['normal']).astype(np.float32)
        # 归一化法线到 [-1, 1] 并不是必须的，因为 Taxim 输出已经是单位向量，
        # 但为了训练稳定，确保它是单位向量。
        # 如果保存的是向量，通常已经在 [-1, 1] 之间。
        normal = np.transpose(normal, (2, 0, 1))
        
        # Mask: (H, W) -> (1, H, W)
        mask = cv2.imread(item['mask'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32) # Binarize
        mask = np.expand_dims(mask, axis=0)
        
        # Transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return {
            'image': img, 
            'depth': torch.from_numpy(depth),
            'normal': torch.from_numpy(normal),
            'mask': torch.from_numpy(mask)
        }

# ==========================================
# 2. Network Architecture (Multi-Task U-Net)
# ==========================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MultiTaskUNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(MultiTaskUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # --- Shared Encoder ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # --- Decoder for Depth ---
        self.up1_d = Up(1024, 512 // factor, bilinear)
        self.up2_d = Up(512, 256 // factor, bilinear)
        self.up3_d = Up(256, 128 // factor, bilinear)
        self.up4_d = Up(128, 64, bilinear)
        self.out_d = OutConv(64, 1) # 1 channel depth

        # --- Decoder for Normal ---
        self.up1_n = Up(1024, 512 // factor, bilinear)
        self.up2_n = Up(512, 256 // factor, bilinear)
        self.up3_n = Up(256, 128 // factor, bilinear)
        self.up4_n = Up(128, 64, bilinear)
        self.out_n = OutConv(64, 3) # 3 channel normal

        # --- Decoder for Mask ---
        self.up1_m = Up(1024, 512 // factor, bilinear)
        self.up2_m = Up(512, 256 // factor, bilinear)
        self.up3_m = Up(256, 128 // factor, bilinear)
        self.up4_m = Up(128, 64, bilinear)
        self.out_m = OutConv(64, 1) # 1 channel mask

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Depth Decoder
        d = self.up1_d(x5, x4)
        d = self.up2_d(d, x3)
        d = self.up3_d(d, x2)
        d = self.up4_d(d, x1)
        depth_pred = self.out_d(d)

        # Normal Decoder
        n = self.up1_n(x5, x4)
        n = self.up2_n(n, x3)
        n = self.up3_n(n, x2)
        n = self.up4_n(n, x1)
        normal_pred = self.out_n(n)
        # Normalize normal vectors
        normal_pred = F.normalize(normal_pred, dim=1)

        # Mask Decoder
        m = self.up1_m(x5, x4)
        m = self.up2_m(m, x3)
        m = self.up3_m(m, x2)
        m = self.up4_m(m, x1)
        mask_pred = self.out_m(m) # Logits

        return depth_pred, normal_pred, mask_pred

# ==========================================
# 3. Loss Functions
# ==========================================
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        # Cosine Similarity: 1 - cos(theta)
        # pred and target should be (B, 3, H, W)
        loss = 1 - F.cosine_similarity(pred, target, dim=1).mean()
        return loss

# ==========================================
# 4. Helper for WandB Visualization
# ==========================================
def log_visualizations(model, val_loader, device, background_img_torch, num_samples=4):
    """
    Pick first few samples from validation loader and log visualizations to WandB.
    """
    model.eval()
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return

    imgs = batch['image'].to(device)
    depth_gt = batch['depth'].to(device)
    normal_gt = batch['normal'].to(device)
    mask_gt = batch['mask'].to(device)

    with torch.no_grad():
        d_pred, n_pred, m_pred = model(imgs)

    # Prepare for logging
    vis_data = []
    
    # Process up to num_samples
    count = min(num_samples, imgs.shape[0])
    
    if args.background_path is not None:
        background_img = cv2.imread(args.background_path)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    
    for i in range(count):
        # --- 1. Input Image ---
        # Denormalize: (tensor * std) + mean
        # Assuming Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = std * img_np + mean
        img_vis = np.clip(img_vis, 0, 1)
        
        if background_img_torch is not None:
            imgs = imgs - background_img_torch

        # --- 2. Depth Maps ---
        d_gt_vis = depth_gt[i, 0].cpu().numpy()
        d_pred_vis = d_pred[i, 0].cpu().numpy()
        
        # Normalize depth for visualization (Min-Max) to see details better
        def norm_depth(d):
            d_min, d_max = d.min(), d.max()
            if d_max - d_min > 1e-6:
                return (d - d_min) / (d_max - d_min)
            return d
        
        # --- 3. Normal Maps ---
        # Map [-1, 1] to [0, 1] for visualization
        n_gt_vis = (normal_gt[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
        n_pred_vis = (n_pred[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
        n_gt_vis = np.clip(n_gt_vis, 0, 1)
        n_pred_vis = np.clip(n_pred_vis, 0, 1)

        # --- 4. Mask ---
        m_gt_vis = mask_gt[i, 0].cpu().numpy()
        m_pred_vis = torch.sigmoid(m_pred[i, 0]).cpu().numpy()
        
        # Create WandB Images
        vis_data.append(wandb.Image(img_vis, caption=f"Input {i}"))
        vis_data.append(wandb.Image(norm_depth(d_gt_vis), caption=f"GT Depth {i}"))
        vis_data.append(wandb.Image(norm_depth(d_pred_vis), caption=f"Pred Depth {i}"))
        vis_data.append(wandb.Image(n_gt_vis, caption=f"GT Normal {i}"))
        vis_data.append(wandb.Image(n_pred_vis, caption=f"Pred Normal {i}"))
        vis_data.append(wandb.Image(m_gt_vis, caption=f"GT Mask {i}"))
        vis_data.append(wandb.Image(m_pred_vis, caption=f"Pred Mask {i}"))

    wandb.log({"val/examples": vis_data})


# ==========================================
# 5. Training Script
# ==========================================
def train_model(args):
    # [Added] Init WandB
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Loader
    train_set = TactileDataset(args.data_root, transform=train_transform, split='train')
    val_set = TactileDataset(args.data_root, transform=val_transform, split='val')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = MultiTaskUNet(n_channels=3).to(device)

    # Optimizer & Losses
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion_depth = nn.L1Loss() 
    criterion_normal = CosineLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    # Create Checkpoint Dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Starting training...")
    if args.background_path is not None:
        background_img = cv2.imread(args.background_path)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img_torch = torch.from_numpy(background_img).unsqueeze(0).permute(0, 3, 1, 2).to(device) / 255.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(loop):
            imgs = batch['image'].to(device)
            depth_gt = batch['depth'].to(device)
            normal_gt = batch['normal'].to(device)
            mask_gt = batch['mask'].to(device)
            if args.background_path is not None:
                imgs = imgs - background_img_torch
            optimizer.zero_grad()
            
            # Forward
            depth_pred, normal_pred, mask_pred = model(imgs)
            
            # Calculate Losses
            mask_bool = (mask_gt > 0.5)
            
            # Mask Loss (Learn to distinguish contact)
            loss_m = criterion_mask(mask_pred, mask_gt)
            
            # Depth Loss (Focus on contact area)
            if mask_bool.sum() > 0:
                loss_d = criterion_depth(depth_pred[mask_bool], depth_gt[mask_bool])
                loss_n = criterion_normal(normal_pred, normal_gt)
            else:
                loss_d = torch.tensor(0.0, device=device, requires_grad=True)
                loss_n = torch.tensor(0.0, device=device, requires_grad=True)

            # Weighted Sum
            total_loss = args.w_depth * loss_d + args.w_normal * loss_n + args.w_mask * loss_m
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), d=loss_d.item(), n=loss_n.item(), m=loss_m.item())
            
            # [Added] Log batch metrics
            wandb.log({
                "train/loss": total_loss.item(),
                "train/loss_depth": loss_d.item() * args.w_depth,
                "train/loss_normal": loss_n.item() * args.w_normal,
                "train/loss_mask": loss_m.item() * args.w_mask,
                "epoch": epoch
            })

        

        scheduler.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                imgs = batch['image'].to(device)
                depth_gt = batch['depth'].to(device)
                normal_gt = batch['normal'].to(device)
                mask_gt = batch['mask'].to(device)
                
                depth_pred, normal_pred, mask_pred = model(imgs)
                
                loss_m = criterion_mask(mask_pred, mask_gt)
                mask_bool = (mask_gt > 0.5)
                
                if mask_bool.sum() > 0:
                    loss_d = criterion_depth(depth_pred[mask_bool], depth_gt[mask_bool])
                    loss_n = criterion_normal(normal_pred, normal_gt)
                else:
                    loss_d = 0
                    loss_n = 0
                
                total_loss = args.w_depth * loss_d + args.w_normal * loss_n + args.w_mask * loss_m
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        # [Added] Log Validation metrics & Visualizations
        wandb.log({"val/loss": avg_val_loss, "epoch": epoch})
        log_visualizations(model, val_loader, device, background_img_torch=None, num_samples=8)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        
        # Save Regular Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth"))
            
    # [Added] Finish wandb
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/training_dataset", help="Dataset root directory")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32) # 320x240 is kinda large, reduce batch size if OOM
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # [Modify] Adjusted loss weights for L1 Loss balance
    parser.add_argument("--w_depth", type=float, default=0.1, help="Weight for depth loss")
    parser.add_argument("--w_normal", type=float, default=1.0, help="Weight for normal loss")
    parser.add_argument("--w_mask", type=float, default=5.0, help="Weight for mask loss")
    parser.add_argument("--background_path", type=str, default=None, help="Path to background image")
    
    # [Added] WandB args
    parser.add_argument("--project_name", type=str, default="tactile-multitask", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist. Please run generate_dataset.py first.")
    else:
        train_model(args)