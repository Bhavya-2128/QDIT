"""
Retinal Fundus Image Preprocessing & Dataset Management for Diabetic Retinopathy Detection
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- Ben Graham's Gaussian filter normalization for retina scans (Section 4.2)
- PyTorch Dataset and DataLoader for APTOS 2019 / Kaggle DR 5-stage classification
- Built-in Synthetic Fundus Image Generator for offline experimentation and testing
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def apply_graham_gaussian_filter(
    image: Image.Image,
    sigma: int = 10,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0,
    auto_crop: bool = True,
    enhance_contrast: bool = True
) -> Image.Image:
    """
    Robust multi-format fundus preprocessing pipeline supporting RGB, Grayscale,
    RGBA, and mixed retina image distributions (Section 4.2 of paper):
    1. Mode normalization (safely standardizes L/RGBA/CMYK -> 3-channel RGB).
    2. Auto-cropping outer black boundary borders around the retinal FOV.
    3. Adaptive contrast enhancement (equalizes illumination across grayscale & RGB).
    4. Ben Graham's Gaussian filter subtraction: I_filtered = alpha*I + beta*Blur(I) + gamma.
    5. Circular boundary mask preserving clean retinal disk.
    """
    # 1. Standardize image mode to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image, dtype=np.uint8)

    # 2. Auto-crop black FOV border margin
    if auto_crop and img_np.ndim == 3:
        gray = np.mean(img_np, axis=-1)
        mask = gray > 10  # Detect non-background pixels
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            h, w = gray.shape
            rmin = max(0, rmin - 2)
            rmax = min(h, rmax + 3)
            cmin = max(0, cmin - 2)
            cmax = min(w, cmax + 3)
            if (rmax - rmin > 30) and (cmax - cmin > 30):
                img_np = img_np[rmin:rmax, cmin:cmax]

    img_pil = Image.fromarray(img_np)

    # 3. Contrast normalization for mixed grayscale / low-contrast fundus scans
    if enhance_contrast:
        img_pil = ImageOps.autocontrast(img_pil, cutoff=1)

    # 4. Ben Graham's Gaussian Filter
    img_f = np.array(img_pil, dtype=np.float32)
    blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    blurred_f = np.array(blurred_pil, dtype=np.float32)

    # Blend to highlight microaneurysms, hemorrhages, and exudates
    filtered_f = alpha * img_f + beta * blurred_f + gamma
    filtered_np = np.clip(filtered_f, 0, 255).astype(np.uint8)

    # 5. Circular mask to eliminate outer boundary noise
    h, w, _ = filtered_np.shape
    center_x, center_y = w // 2, h // 2
    radius = int(min(center_x, center_y) * 0.96)
    y, x = np.ogrid[:h, :w]
    circ_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    filtered_np[~circ_mask] = 0

    filtered_pil = Image.fromarray(filtered_np)
    # Ensure final dimensions strictly match input size
    if filtered_pil.size != image.size:
        filtered_pil = filtered_pil.resize(image.size, Image.BILINEAR)

    return filtered_pil


def get_fundus_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    apply_graham: bool = True
) -> transforms.Compose:
    """
    Constructs torchvision transform pipeline including resize, robust Graham filter,
    data augmentations, and ImageNet tensor normalization.
    """
    transform_list = []

    # 1. Resize and crop
    transform_list.append(transforms.Resize(image_size))
    if is_training:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=20))
        transform_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))

    # 2. Gaussian Graham filter with contrast normalization & auto-cropping
    if apply_graham:
        transform_list.append(transforms.Lambda(apply_graham_gaussian_filter))

    # 3. Tensor conversion and standard ImageNet normalization
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    return transforms.Compose(transform_list)


def resolve_dataset_path(dataset_source: Optional[str] = None) -> str:
    """
    Resolves the dataset directory path whether running on Kaggle (/kaggle/input/...),
    via kagglehub slug (e.g., 'bhavyasanghavi2348/data-qdit'), or from a local folder.
    """
    # 1. Direct check if path exists locally
    if dataset_source and os.path.exists(dataset_source):
        return dataset_source

    # 2. Check standard Kaggle attached dataset paths
    kaggle_paths = [
        "/kaggle/input/data-qdit",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input/diabetic-retinopathy-detection"
    ]
    for kp in kaggle_paths:
        if os.path.exists(kp):
            print(f"📦 Detected Kaggle attached dataset at: {kp}")
            return kp

    # 3. Check if dataset_source is a competition or dataset slug via kagglehub
    if dataset_source and not os.path.exists(dataset_source):
        try:
            import kagglehub
            clean_slug = dataset_source.strip()
            # If it is a competition name (like 'aptos2019-blindness-detection')
            if "/" not in clean_slug or "aptos2019" in clean_slug:
                try:
                    print(f"📥 Accessing Kaggle competition '{clean_slug}' via kagglehub...")
                    comp_path = kagglehub.competition_download(clean_slug)
                    print(f"✅ Kagglehub competition available at: {comp_path}")
                    return comp_path
                except Exception as comp_err:
                    print(f"⚠️  Competition access requires accepting rules: {comp_err}")
                    print("🔄 Falling back to public open mirror dataset on Kaggle...")
                    try:
                        mirror_path = kagglehub.dataset_download("bhavyasanghavi2348/data-qdit")
                        print(f"✅ Public open mirror available at: {mirror_path}")
                        return mirror_path
                    except Exception as m_err:
                        print(f"ℹ️  Mirror 1 failed: {m_err}. Trying secondary mirror...")
                        try:
                            mirror_path2 = kagglehub.dataset_download("sovitrath/diabetic-retinopathy-224x224-gaussian-filtered")
                            print(f"✅ Secondary mirror available at: {mirror_path2}")
                            return mirror_path2
                        except Exception:
                            pass
            
            print(f"📥 Accessing Kaggle dataset '{clean_slug}' via kagglehub...")
            downloaded_path = kagglehub.dataset_download(clean_slug)
            print(f"✅ Kagglehub dataset available at: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            print(f"⚠️  Could not access via kagglehub ({e}).")

    # 4. Fallback: Check if any dataset exists inside /kaggle/input
    if os.path.exists("/kaggle/input"):
        subdirs = [os.path.join("/kaggle/input", d) for d in os.listdir("/kaggle/input") if os.path.isdir(os.path.join("/kaggle/input", d))]
        if subdirs:
            print(f"📦 Auto-selected Kaggle input directory: {subdirs[0]}")
            return subdirs[0]

    return dataset_source or "./data/synthetic_fundus"


class DiabeticRetinopathyDataset(Dataset):
    """
    Dataset class for loading fundus images from folder structures:
      dataset_dir/
        0/ (No DR)
        1/ (Mild DR)
        2/ (Moderate DR)
        3/ (Severe DR)
        4/ (Proliferative DR)
    or from CSV files (e.g. train.csv with 'id_code' and 'diagnosis' / 'level').
    Supports automatic recursive exploration for Kaggle dataset structures.
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: Optional[str] = None,
        image_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        resolved_root = resolve_dataset_path(root_dir)
        self.root_dir = Path(resolved_root) if resolved_root else None
        self.transform = transform or get_fundus_transforms(image_size=image_size, is_training=False)
        self.samples: List[Tuple[Path, int]] = []

        if not self.root_dir or not self.root_dir.exists():
            return

        # Check for CSV files inside root_dir if not explicitly passed
        if not csv_file:
            for candidate_csv_name in ["train.csv", "train_labels.csv", "labels.csv", "train_dr.csv"]:
                p = self.root_dir / candidate_csv_name
                if p.exists():
                    csv_file = str(p)
                    break
            if not csv_file:
                for potential_csv in self.root_dir.glob("*.csv"):
                    if "sample" not in potential_csv.name.lower() and "test" not in potential_csv.name.lower():
                        csv_file = str(potential_csv)
                        break

        if csv_file and os.path.exists(csv_file):
            import pandas as pd
            df = pd.read_csv(csv_file)

            # Determine image column and target column
            img_col = next((c for c in ['id_code', 'image', 'image_id', 'filename', 'Image_name'] if c in df.columns), df.columns[0])
            label_col = next((c for c in ['diagnosis', 'level', 'stage', 'label', 'Retinopathy grade'] if c in df.columns), df.columns[-1])

            # Determine image directory
            candidate_dirs = [
                self.root_dir / "train_images",
                self.root_dir / "train",
                self.root_dir / "images",
                self.root_dir / "gaussian_filtered_images" / "gaussian_filtered_images",
                self.root_dir / "gaussian_filtered_images",
                self.root_dir,
            ]
            actual_img_dir = Path(image_dir) if image_dir else None
            if not actual_img_dir:
                for cd in candidate_dirs:
                    if cd.exists() and cd.is_dir():
                        if len(df) > 0:
                            first_name = str(df.iloc[0][img_col])
                            if (cd / first_name).exists() or (cd / f"{first_name}.png").exists() or (cd / f"{first_name}.jpg").exists():
                                actual_img_dir = cd
                                break
                        if any(cd.glob("*.png")) or any(cd.glob("*.jpg")):
                            actual_img_dir = cd
                            break
                if not actual_img_dir:
                    actual_img_dir = self.root_dir

            for _, row in df.iterrows():
                base_name = str(row[img_col])
                img_path = actual_img_dir / base_name
                if not img_path.exists():
                    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                        test_p = actual_img_dir / f"{base_name}{ext}"
                        if test_p.exists():
                            img_path = test_p
                            break
                if img_path.exists():
                    try:
                        label = int(row[label_col])
                        self.samples.append((img_path, label))
                    except Exception:
                        pass

        # If no samples from CSV, search stage subfolders (0, 1, 2, 3, 4)
        if len(self.samples) == 0:
            for stage in range(5):
                stage_matches = list(self.root_dir.glob(f"**/{stage}/*.*"))
                for file_path in stage_matches:
                    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                        self.samples.append((file_path, stage))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def generate_synthetic_fundus_image(
    stage: int = 0,
    size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Generates a realistic synthetic retinal fundus image for testing and demonstration.
    Simulates:
      - Orange-red retinal fundus background with radial shading
      - Bright circular optic disc
      - Dark fovea / macula region
      - Branching retinal blood vessels
      - Stage-dependent lesions (0: clean, 1: microaneurysms, 2: hard exudates, 3: cotton wool spots, 4: neovascularization & hemorrhages)
    """
    w, h = size
    img = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Retina background disk
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * 0.9)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(180, 60, 20))

    # Inner warm glow
    r_inner = int(r * 0.7)
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(210, 80, 25))

    # 2. Optic Disc (bright yellowish disc on the left/right side)
    od_x = cx - int(r * 0.45)
    od_y = cy - int(r * 0.1)
    od_r = int(r * 0.18)
    draw.ellipse([od_x - od_r, od_y - od_r, od_x + od_r, od_y + od_r], fill=(255, 230, 150))

    # 3. Macula / Fovea (dark central spot)
    mac_x = cx + int(r * 0.25)
    mac_y = cy
    mac_r = int(r * 0.12)
    draw.ellipse([mac_x - mac_r, mac_y - mac_r, mac_x + mac_r, mac_y + mac_r], fill=(140, 40, 15))

    # 4. Branching Blood Vessels
    math_pi = 3.14159
    for _ in range(8):
        vessel_points = [(od_x, od_y)]
        curr_x, curr_y = od_x, od_y
        angle = random.uniform(-math_pi, math_pi)
        length = random.randint(3, 6)
        for _ in range(length):
            angle += random.uniform(-0.4, 0.4)
            step = random.randint(15, 30)
            curr_x += int(step * np.cos(angle))
            curr_y += int(step * np.sin(angle))
            vessel_points.append((curr_x, curr_y))
        draw.line(vessel_points, fill=(110, 20, 10), width=random.randint(2, 4))

    # 5. Add stage-specific pathological lesions
    if stage >= 1:
        # Stage 1: Microaneurysms (small red dots)
        num_ma = random.randint(5, 12) * stage
        for _ in range(num_ma):
            lx = random.randint(cx - int(r * 0.6), cx + int(r * 0.6))
            ly = random.randint(cy - int(r * 0.6), cy + int(r * 0.6))
            draw.ellipse([lx - 2, ly - 2, lx + 2, ly + 2], fill=(160, 10, 10))

    if stage >= 2:
        # Stage 2: Hard Exudates (bright yellowish lipid deposits)
        num_ex = random.randint(6, 15) * (stage - 1)
        for _ in range(num_ex):
            lx = random.randint(cx - int(r * 0.5), cx + int(r * 0.5))
            ly = random.randint(cy - int(r * 0.5), cy + int(r * 0.5))
            draw.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=(255, 250, 180))

    if stage >= 3:
        # Stage 3: Cotton Wool Spots (fluffy whitish ischemic lesions) & large blot hemorrhages
        num_cws = random.randint(4, 8)
        for _ in range(num_cws):
            lx = random.randint(cx - int(r * 0.5), cx + int(r * 0.5))
            ly = random.randint(cy - int(r * 0.5), cy + int(r * 0.5))
            draw.ellipse([lx - 7, ly - 5, lx + 7, ly + 5], fill=(230, 225, 210))

    if stage >= 4:
        # Stage 4: Proliferative DR (extensive neovascularization and vitreous/preretinal hemorrhages)
        for _ in range(5):
            hx = random.randint(cx - int(r * 0.5), cx + int(r * 0.5))
            hy = random.randint(cy - int(r * 0.5), cy + int(r * 0.5))
            draw.ellipse([hx - 14, hy - 10, hx + 14, hy + 10], fill=(120, 5, 5))

    # Add soft blur to simulate camera optics
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def create_synthetic_fundus_dataset(
    output_dir: str,
    samples_per_stage: int = 20,
    image_size: Tuple[int, int] = (224, 224)
) -> str:
    """
    Creates a sample dataset of synthetic retinal fundus images organized by DR stages (0 to 4).
    """
    base_dir = Path(output_dir)
    for stage in range(5):
        stage_dir = base_dir / str(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_stage):
            img = generate_synthetic_fundus_image(stage=stage, size=image_size)
            img_path = stage_dir / f"fundus_stage{stage}_sample{i:03d}.png"
            img.save(img_path)

    return str(base_dir)


class TransformedSubset(Dataset):
    """
    Applies custom dataset transformation to indexed subsets without dataset duplication.
    """
    def __init__(self, full_ds, indices, tf):
        self.full_ds = full_ds
        self.indices = indices
        self.tf = tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        img_path, label = self.full_ds.samples[orig_idx]
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)
        return image, label


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (224, 224),
    apply_graham: bool = True,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and validation DataLoaders with stratified/random split.
    """
    full_dataset = DiabeticRetinopathyDataset(
        root_dir=dataset_dir,
        transform=None,
        image_size=image_size
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No fundus images found in directory: {dataset_dir}")

    total_len = len(full_dataset)
    labels = [s[1] for s in full_dataset.samples]

    # Use Stratified Split so all 5 classes are represented proportionally in validation
    try:
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            list(range(total_len)),
            train_size=train_split,
            stratify=labels,
            random_state=seed
        )
    except Exception:
        train_len = int(total_len * train_split)
        val_len = total_len - train_len
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = torch.utils.data.random_split(
            range(total_len), [train_len, val_len], generator=generator
        )

    train_transform = get_fundus_transforms(image_size=image_size, is_training=True, apply_graham=apply_graham)
    val_transform = get_fundus_transforms(image_size=image_size, is_training=False, apply_graham=apply_graham)

    train_ds = TransformedSubset(full_dataset, train_indices, train_transform)
    val_ds = TransformedSubset(full_dataset, val_indices, val_transform)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
