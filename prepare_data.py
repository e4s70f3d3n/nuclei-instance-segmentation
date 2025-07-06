import random
import os
import glob
import tifffile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from typing import List, Optional, Tuple, Dict
from skimage import exposure

class NuInsSegDatasetV2(Dataset):
    def __init__(
        self,
        root_dir: str,
        tissue_types: Optional[List[str]] = None,
        num_points: int = 256,
        apply_augmentation: bool = True,
        stain_normalize: bool = True,
        use_albumentations: bool = False,
        subset_fraction: Optional[float] = None,
        subset_seed: int = 42
    ):
        """ 
        Args:
            root_dir: Root directory containing tissue type folders
            tissue_types: Specific tissue types to include (None for all)
            num_points: Number of prompt points per image
            apply_augmentation: Whether to apply data augmentation
            stain_normalize: Whether to apply stain normalization
            use_albumentations: Whether to use albumentations for augmentations
            subset_fraction: Fraction of dataset to use (for debugging)
            subset_seed: Random seed for subset selection
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        self.stain_normalize = stain_normalize
        self.use_albumentations = use_albumentations
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        
        # Initialize augmentation transforms
        self._init_transforms()
        
        # Discover and validate samples
        self.samples = self._discover_samples(tissue_types)
        
        # Apply subset if requested
        if self.subset_fraction is not None:
            self._apply_subset()
        
        # Normalization parameters
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _apply_subset(self) -> None:
        """Create a random subset of the dataset for debugging."""
        if self.subset_fraction <= 0 or self.subset_fraction > 1:
            raise ValueError("subset_fraction must be between 0 and 1")
        
        # Calculate number of samples to keep
        num_samples = len(self.samples)
        subset_size = max(1, int(num_samples * self.subset_fraction))
        
        # Set random seed for reproducibility
        rng = random.Random(self.subset_seed)
        
        # Create subset
        self.samples = rng.sample(self.samples, subset_size)
        
        # Print subset information
        print(f"Created subset with {subset_size}/{num_samples} samples "
              f"({self.subset_fraction*100:.1f}%)")

    def _init_transforms(self) -> None:
        """Initialize different transformation pipelines."""
        if self.use_albumentations:
            # Albumentations augmentation pipeline
            self.augmentation = A.Compose([
                A.OneOf([
                    A.Rotate(limit=30, p=0.5),
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.2),
                ], p=0.8),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=0.5),
                A.CLAHE(p=0.2),
            ])
            
            # Separate transform for converting to tensor
            self.to_tensor = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            # Traditional augmentation options
            self.aug_types = ['rotation', 'flip', 'color', 'stain_norm']

    def _discover_samples(self, tissue_types: Optional[List[str]]) -> List[Tuple]:
        """Discover and validate all available samples."""
        if tissue_types is None:
            tissue_types = [
                d for d in os.listdir(self.root_dir) 
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        
        samples = []
        
        for tissue in tissue_types:
            tissue_dir = os.path.join(self.root_dir, tissue)
            image_dir = os.path.join(tissue_dir, "tissue images")
            label_dir = os.path.join(tissue_dir, "label masks")
            
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                continue
                
            # Find matching image-label pairs
            for img_path in glob.glob(os.path.join(image_dir, "*.png")):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, f"{base_name}.tif")
                
                if os.path.exists(label_path):
                    # Standard format: (img_path, label_path, is_augmented, aug_type)
                    # For original samples (non-augmented)
                    samples.append((img_path, label_path, False, None))
                    
                    # Add augmented versions if enabled
                    if self.apply_augmentation:
                        if self.use_albumentations:
                            # For albumentations
                            samples.append((img_path, label_path, True, "albumentations"))
                        else:
                            # For traditional augmentations
                            for aug_type in self.aug_types:
                                samples.append((img_path, label_path, True, aug_type))
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        img_path = sample_info[0]
        label_path = sample_info[1]
        
        image = self._load_image(img_path)
        instance_mask = tifffile.imread(label_path)
        
        binary_mask = (instance_mask > 0).astype(np.float32)
        panoptic_mask = np.stack([instance_mask, binary_mask], axis=0)
        
        prompt_points, prompt_labels = self._generate_prompts(instance_mask)
        
        if len(sample_info) > 2 and sample_info[2]: 
            if self.use_albumentations:
                augmented = self.augmentation(
                    image=image,
                    masks=[instance_mask, binary_mask]
                )
                image = augmented['image']
                instance_mask, binary_mask = augmented['masks']

                if any(isinstance(t, A.geometric.transforms.GeometricTransform) 
                       for t in self.augmentation.transforms):
                    prompt_points, prompt_labels = self._generate_prompts(instance_mask)
            else:
                aug_type = sample_info[3]
                image, instance_mask, binary_mask, prompt_points = self._apply_manual_augmentation(
                    image, instance_mask, binary_mask, prompt_points, prompt_labels, aug_type
                )
        
        if self.use_albumentations:
            tensorized = self.to_tensor(image=image)
            image_tensor = tensorized['image']
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = self.normalize(image_tensor)
        
        sample = {
            'image': image_tensor,
            'panoptic_mask': torch.from_numpy(panoptic_mask).float(),
            'binary_mask': torch.from_numpy(binary_mask).float(),
            'instance_mask': torch.from_numpy(instance_mask).float(),
            'prompt_points': torch.tensor(prompt_points, dtype=torch.float32),
            'prompt_labels': torch.tensor(prompt_labels, dtype=torch.float32),
            'image_path': img_path,
            'mask_path': label_path
        }
        
        return sample

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image with different approach."""
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        
        if self.stain_normalize:
            img = self._normalize_stains(img)
            
        return img

    def _normalize_stains(self, img: np.ndarray) -> np.ndarray:
        """Alternative stain normalization using Macenko method."""
        od = -np.log((img.astype(np.float32) + 1) / 255)  # Fixed the parenthesis
        
        mask = (od > 0.15).any(axis=-1) & (od < 0.85).any(axis=-1)
        
        if mask.sum() > 0:
            od_vectors = od[mask].reshape(-1, 3)
            _, _, vh = np.linalg.svd(od_vectors, full_matrices=False)
            
            stains = vh[:2, :]
            
            stains = stains / np.linalg.norm(stains, axis=1, keepdims=True)
            
            proj = np.dot(od.reshape(-1, 3), stains.T)
            proj = exposure.rescale_intensity(proj, out_range=(0, 1))
            
            reconstructed = np.dot(proj, stains).reshape(img.shape)
            normalized = np.exp(-reconstructed) * 255
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            return normalized
        
        return img

    def _generate_prompts(self, instance_mask: np.ndarray) -> Tuple[List, List]:
        instance_ids = np.unique(instance_mask)[1:]  
        prompt_points = []
        prompt_labels = []
        
        for instance_id in instance_ids:
            # Find all coordinates for this instance
            coords = np.argwhere(instance_mask == instance_id)
            
            if len(coords) > 0:
                centroid = np.mean(coords, axis=0)
                prompt_points.append([centroid[1], centroid[0]])  # x, y
                prompt_labels.append(1)
        
        num_valid = len(prompt_points)
        
        if num_valid >= self.num_points:
            indices = np.random.permutation(num_valid)[:self.num_points]
            prompt_points = [prompt_points[i] for i in indices]
            prompt_labels = [prompt_labels[i] for i in indices]
        else:
            num_pad = self.num_points - num_valid
            h, w = instance_mask.shape
            
            if num_valid > 0:
                pass  # Keep existing points
            else:
                # If no valid points, create some random background points
                for _ in range(min(10, self.num_points)):
                    prompt_points.append([
                        random.randint(0, w-1),
                        random.randint(0, h-1)
                    ])
                    prompt_labels.append(0)
                num_pad = self.num_points - len(prompt_points)
            
            # Pad with invalid points
            prompt_points.extend([[0, 0]] * num_pad)
            prompt_labels.extend([-1] * num_pad)
            
        return prompt_points, prompt_labels

    def _apply_manual_augmentation(
        self,
        image: np.ndarray,
        instance_mask: np.ndarray,
        binary_mask: np.ndarray,
        prompt_points: List,
        prompt_labels: List,
        aug_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        h, w = image.shape[:2]
        
        if aug_type == 'rotation':
            angle = random.uniform(-180, 180)
            image = ndimage.rotate(image, angle, reshape=False, mode='reflect')
            instance_mask = ndimage.rotate(instance_mask, angle, reshape=False, order=0, mode='constant')
            binary_mask = ndimage.rotate(binary_mask, angle, reshape=False, order=0, mode='constant')
            
            center = np.array([w/2, h/2])
            for i, (x, y) in enumerate(prompt_points):
                if prompt_labels[i] == 1: 
                    point = np.array([x, y]) - center
                    rad = np.deg2rad(angle)
                    rot_mat = np.array([
                        [np.cos(rad), -np.sin(rad)],
                        [np.sin(rad), np.cos(rad)]
                    ])
                    rotated = np.dot(rot_mat, point) + center
                    prompt_points[i] = rotated.tolist()
        
        elif aug_type == 'flip':
            flip_code = random.choice([-1, 0, 1]) 
            
            image = cv2.flip(image, flip_code)
            instance_mask = cv2.flip(instance_mask, flip_code)
            binary_mask = cv2.flip(binary_mask, flip_code)
            
            for i, (x, y) in enumerate(prompt_points):
                if prompt_labels[i] == 1:
                    if flip_code == 1 or flip_code == -1:  
                        x = w - 1 - x
                    if flip_code == 0 or flip_code == -1: 
                        y = h - 1 - y
                    prompt_points[i] = [x, y]
        
        elif aug_type == 'color':
            image = image.astype(np.float32)
            
            gamma = random.uniform(0.7, 1.5)
            image = np.power(image / 255.0, gamma) * 255.0
            
            for c in range(3):
                shift = random.uniform(-30, 30)
                image[..., c] = np.clip(image[..., c] + shift, 0, 255)
            
            image = image.astype(np.uint8)
        
        elif aug_type == 'stain_norm':
            image = self._normalize_stains(image)
        
        return image, instance_mask, binary_mask, prompt_points


class BalancedNuInsSegDataset(NuInsSegDatasetV2):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_sample_indices()
    
    def _build_sample_indices(self) -> None:
        self.tissue_indices = {}
        
        for idx, (img_path, *_) in enumerate(self.samples):
            tissue_type = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            if tissue_type not in self.tissue_indices:
                self.tissue_indices[tissue_type] = []
            self.tissue_indices[tissue_type].append(idx)
        
        self.tissue_types = list(self.tissue_indices.keys())
        
        if self.subset_fraction is not None:
            self._apply_balanced_subset()

    def __getitem__(self, idx: int) -> Dict:
        tissue = random.choice(self.tissue_types)
        idx = random.choice(self.tissue_indices[tissue])
        return super().__getitem__(idx)

    def _apply_balanced_subset(self) -> None:
        """Create a balanced subset maintaining tissue type proportions."""
        rng = random.Random(self.subset_seed)
        new_indices = []
        
        for tissue, indices in self.tissue_indices.items():
            subset_size = max(1, int(len(indices) * self.subset_fraction))
    
            new_indices.extend(rng.sample(indices, subset_size))
        
        self.samples = [self.samples[i] for i in new_indices]
        self._build_sample_indices()
        
        print(f"Created balanced subset with {len(new_indices)} samples "
              f"({self.subset_fraction*100:.1f}%)")