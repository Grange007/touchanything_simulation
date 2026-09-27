import os
import glob
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ==========================================
# 1. 模型结构 (保持不变)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class MultiTaskUNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(MultiTaskUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1_d = Up(1024, 512 // factor, bilinear)
        self.up2_d = Up(512, 256 // factor, bilinear)
        self.up3_d = Up(256, 128 // factor, bilinear)
        self.up4_d = Up(128, 64, bilinear)
        self.out_d = OutConv(64, 1)
        self.up1_n = Up(1024, 512 // factor, bilinear)
        self.up2_n = Up(512, 256 // factor, bilinear)
        self.up3_n = Up(256, 128 // factor, bilinear)
        self.up4_n = Up(128, 64, bilinear)
        self.out_n = OutConv(64, 3)
        self.up1_m = Up(1024, 512 // factor, bilinear)
        self.up2_m = Up(512, 256 // factor, bilinear)
        self.up3_m = Up(256, 128 // factor, bilinear)
        self.up4_m = Up(128, 64, bilinear)
        self.out_m = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d = self.up1_d(x5, x4); d = self.up2_d(d, x3); d = self.up3_d(d, x2); d = self.up4_d(d, x1); depth_pred = self.out_d(d)
        n = self.up1_n(x5, x4); n = self.up2_n(n, x3); n = self.up3_n(n, x2); n = self.up4_n(n, x1); normal_pred = self.out_n(n); normal_pred = F.normalize(normal_pred, dim=1)
        m = self.up1_m(x5, x4); m = self.up2_m(m, x3); m = self.up3_m(m, x2); m = self.up4_m(m, x1); mask_pred = self.out_m(m)
        return depth_pred, normal_pred, mask_pred

# ==========================================
# 2. 推理并保存 Raw Data
# ==========================================
def reconstruct_and_save_raw(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    model = MultiTaskUNet(n_channels=3).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 准备路径
    # 假设 args.data_root 包含 'alignment' 文件夹和图片文件夹 (args.mesh_name)
    align_path = os.path.join(args.data_root, args.mesh_name, "alignment", "contact_frames.npy")
    input_img_dir = os.path.join(args.data_root, args.mesh_name, "raw")
    
    # 结果输出目录
    results_dir = os.path.join(args.data_root, args.mesh_name, "reconstruction") # e.g., ./intermediate_results/002_master_chef_can
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Alignment file not found: {align_path}")
        
    # 复制位姿文件到结果目录，方便下一步直接读取
    # 或者下一步直接去读源目录也行，这里为了自包含，我们假设只输出 Raw 预测数据
    
    img_files = sorted(glob.glob(os.path.join(input_img_dir, "*_color.png")))
    if not img_files:
        raise FileNotFoundError(f"No tactile color images in {input_img_dir}")
    contact_frames = np.load(align_path, allow_pickle=True)
    
    print(f"Found {len(img_files)} images.")
    print(f"Saving raw predictions to {results_dir} ...")

    # Visualization setup
    all_pcds = []
    camera_frames = []
    imgh, imgw = 240, 320
    ppmm = 0.0634
    x = (np.arange(imgw) - imgw / 2 + 0.5) * ppmm / 1000.0
    y = (np.arange(imgh) - imgh / 2 + 0.5) * ppmm / 1000.0
    xv, yv = np.meshgrid(x, y)

    with torch.no_grad():
        for i, img_path in enumerate(img_files):
            try:
                frame_idx = int(os.path.basename(img_path).split('_')[-2])
            except: continue
            
            if frame_idx >= len(contact_frames): continue

            # --- 推理 ---
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            d_pred, n_pred, m_pred = model(input_tensor)
            
            # --- 解析数据 ---
            # 1. 深度
            depth_m =  d_pred.squeeze().cpu().numpy() * ppmm / 1000.0
            depth_m = -depth_m + np.max(depth_m)  # 转换为距离传感器的深度值
            
            # 2. Mask (Logits -> Sigmoid -> 0/1)
            mask_logits = m_pred.squeeze().cpu().numpy()
            mask_prob = 1 / (1 + np.exp(-mask_logits))
            mask_uint8 = (mask_prob > 0.5).astype(np.uint8) * 255
            
            # 3. 法线 (Model Predicted, 仅供参考，下一步可能会重新计算)
            # 格式: (3, H, W) -> (H, W, 3)
            normal_pred = n_pred.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            # --- 保存 Raw Data ---
            file_id = f"{frame_idx:06d}"
            
            os.makedirs(os.path.join(results_dir), exist_ok=True)
            # 保存转换后的深度 (转换为米，方便 SDF 使用)
            np.save(os.path.join(results_dir, f"{file_id}_pred_depth.npy"), depth_m)
            
            # 保存预测法线 (作为备用)
            np.save(os.path.join(results_dir, f"{file_id}_pred_normal.npy"), normal_pred)
            
            # 保存 Mask
            cv2.imwrite(os.path.join(results_dir, f"{file_id}_pred_mask.png"), mask_uint8)
            
            # 保存原始 RGB (方便下一步打包)
            cv2.imwrite(os.path.join(results_dir, f"{file_id}_rgb.png"), img_bgr)
            
            if args.visualize:
                valid_mask = mask_uint8 > 127
                if np.sum(valid_mask) > 0:
                    z_vals = depth_m[valid_mask]
                    x_vals = xv[valid_mask]
                    y_vals = yv[valid_mask]
                    
                    points_sensor = np.stack([x_vals, y_vals, z_vals], axis=-1)
                    
                    pose = contact_frames[frame_idx]
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    
                    points_world = (R @ points_sensor.T).T + t
                    
                    # Downsample
                    if len(points_world) > 500:
                        idx = np.random.choice(len(points_world), 500, replace=False)
                        points_world = points_world[idx]
                        
                    all_pcds.append(points_world)
                    
                    camera_frames.append(pose)

            if i % 50 == 0:
                print(f"Processed {i}/{len(img_files)}")

    print("Inference complete.")
    
    if args.visualize and len(all_pcds) > 0:
        print("Visualizing...")
        # o3d.visualization.draw_geometries(all_pcds + camera_frames)
        all_pcds = np.concatenate(all_pcds, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pcds)
        o3d.io.write_point_cloud(os.path.join(results_dir, f"{args.mesh_name}_all_points.ply"), pcd, write_ascii=False, compressed=False, print_progress=False)
        # o3d.io.write_triangle_mesh(os.path.join(args.output_dir, f"{args.mesh_name}_camera_frames.ply"), camera_frames, write_ascii=False, compressed=False, print_progress=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2a: Inference (Save Raw Predictions)")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir of Step 1 output (contains alignment/)")
    parser.add_argument("--mesh_name", type=str, required=True, help="Object Name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Visualize results with Open3D")
    
    args = parser.parse_args()
    reconstruct_and_save_raw(args)
