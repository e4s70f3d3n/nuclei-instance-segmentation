import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
from monai.metrics import DiceMetric, MeanIoU
from skimage.measure import label
from skimage.segmentation import find_boundaries

import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from skimage.segmentation import find_boundaries
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
from monai.metrics import DiceMetric, MeanIoU
from skimage.measure import label
from torchvision.utils import save_image
import os

def visualize_predictions(model, dataset, device, output_dir=None, num_samples=5):
    model.eval()
    visualizations = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Handle Subset by accessing the underlying dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices_map = dataset.indices
    else:
        base_dataset = dataset
        indices_map = list(range(len(dataset)))

    # Filter non-augmented indices only
    original_indices = [i for i, s in enumerate(base_dataset.samples) if not s[2]]
    mapped_indices = [indices_map[i] for i in original_indices if i < len(indices_map)]
    indices = np.random.choice(mapped_indices, min(num_samples, len(mapped_indices)), replace=False)

    vis_subset = Subset(base_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in vis_loader:
            images = batch['image'].to(device).float()
            gt_masks = batch['binary_mask'].to(device)
            prompt_points = batch['prompt_points'].to(device).unsqueeze(1).float()
            prompt_labels = batch['prompt_labels'].to(device).unsqueeze(1).float()
            image_path = batch['image_path'][0]
            sample_name = os.path.basename(image_path).split('.')[0]

            predicted_masks, iou_predictions = model(images, prompt_points, prompt_labels)
            if iou_predictions is not None:
                best_idx = iou_predictions.argmax(dim=2)
                predicted_mask = predicted_masks[0, 0, best_idx[0, 0]]
            else:
                predicted_mask = predicted_masks[0, 0, 0]

            if predicted_mask.shape != gt_masks.shape[-2:]:
                predicted_mask = F.interpolate(predicted_mask.unsqueeze(0).unsqueeze(0), 
                                               size=gt_masks.shape[-2:], 
                                               mode='bilinear', align_corners=False).squeeze()

            pred_binary = (predicted_mask > 0.5).float()
            gt_instance_mask = batch['instance_mask'][0].cpu().numpy()
            pred_binary_np = pred_binary.cpu().numpy()
            pred_instance_mask = label(pred_binary_np)

            visualizations.append({
                "filename": sample_name,
                "image": images[0].cpu(),
                "prediction": pred_binary.cpu(),
                "ground_truth": gt_masks[0].cpu()
            })

            # Denormalize and prepare image
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

            try:
                orig_img = np.array(Image.open(image_path).convert("RGB"))
            except:
                orig_img = img

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title('Original Image'); 
            axes[0, 0].axis('off')

            gt_mask_np = gt_masks[0].cpu().numpy()
            axes[0, 1].imshow(orig_img)
            axes[0, 1].imshow(gt_mask_np, alpha=0.5, cmap='Reds')
            axes[0, 1].set_title('Ground Truth Mask'); 
            axes[0, 1].axis('off')

            axes[0, 2].imshow(orig_img)
            axes[0, 2].imshow(pred_binary_np, alpha=0.5, cmap='Blues')
            axes[0, 2].set_title('Predicted Mask'); 
            axes[0, 2].axis('off')

            cmap = plt.cm.tab20

            gt_instance_colored = np.zeros((*gt_instance_mask.shape, 4))
            for idx, inst_id in enumerate(np.unique(gt_instance_mask)[1:]):
                gt_instance_colored[gt_instance_mask == inst_id] = cmap(idx % 20)
            axes[1, 0].imshow(orig_img)
            axes[1, 0].imshow(gt_instance_colored, alpha=0.7)
            axes[1, 0].set_title('Ground Truth Instances'); axes[1, 0].axis('off')

            pred_instance_colored = np.zeros((*pred_instance_mask.shape, 4))
            for idx, inst_id in enumerate(np.unique(pred_instance_mask)[1:]):
                pred_instance_colored[pred_instance_mask == inst_id] = cmap(idx % 20)
            axes[1, 1].imshow(orig_img)
            axes[1, 1].imshow(pred_instance_colored, alpha=0.7)
            axes[1, 1].set_title('Predicted Instances'); axes[1, 1].axis('off')

            tp_mask = np.logical_and(pred_binary_np > 0.5, gt_mask_np > 0.5)
            fp_mask = np.logical_and(pred_binary_np > 0.5, gt_mask_np <= 0.5)
            fn_mask = np.logical_and(pred_binary_np <= 0.5, gt_mask_np > 0.5)

            error_vis = np.zeros((*gt_mask_np.shape, 3))
            error_vis[tp_mask, 1] = 1.0
            error_vis[fp_mask, 0] = 1.0
            error_vis[fn_mask, 2] = 1.0
            axes[1, 2].imshow(orig_img)
            axes[1, 2].imshow(error_vis, alpha=0.5)
            axes[1, 2].set_title('Error Analysis'); axes[1, 2].axis('off')

            for pt_idx in range(prompt_points.shape[2]):
                if prompt_labels[0, 0, pt_idx] > 0:
                    x, y = prompt_points[0, 0, pt_idx].cpu().numpy()
                    for ax in axes[0, :]:
                        ax.plot(x, y, 'yo', markersize=8, alpha=0.7)

            dice_metric = DiceMetric(include_background=False, reduction="mean")
            jaccard_metric = MeanIoU()
            dice_metric.reset()
            jaccard_metric.reset()
            dice_metric(y_pred=pred_binary.unsqueeze(0).unsqueeze(0), y=gt_masks.unsqueeze(0))
            jaccard_metric(y_pred=pred_binary.unsqueeze(0).unsqueeze(0), y=gt_masks.unsqueeze(0))
            dice_score = dice_metric.aggregate().item()
            iou_score = jaccard_metric.aggregate().item()

            pred_instances = len(np.unique(pred_instance_mask)) - 1
            gt_instances = len(np.unique(gt_instance_mask)) - 1

            plt.suptitle(f"{sample_name} | Dice: {dice_score:.4f}, IoU: {iou_score:.4f} | "
                         f"GT: {gt_instances}, Pred: {pred_instances}", fontsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            if output_dir:
                save_path = os.path.join(output_dir, f"{sample_name}_vis.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization to {save_path}")
            else:
                plt.close(fig)

    if output_dir:
        print(f"All visualizations saved to: {output_dir}")
    return visualizations

def save_visualizations_to_drive(visuals, output_dir, fold, epoch=None):
    os.makedirs(output_dir, exist_ok=True)
    for i, v in enumerate(visuals):
        prefix = f"fold{fold}_sample{i}"
        if epoch is not None:
            prefix = f"fold{fold}_epoch{epoch}_sample{i}"

        # Save image, prediction, ground truth
        save_image(v['image'], os.path.join(output_dir, f"{prefix}_image.png"))
        save_image(v['prediction'].unsqueeze(0), os.path.join(output_dir, f"{prefix}_pred.png"))
        save_image(v['ground_truth'].unsqueeze(0), os.path.join(output_dir, f"{prefix}_gt.png"))

def display_visualizations_inline(visuals, fold=None):
    for vis in visuals:
        image = vis['image']
        prediction = vis['prediction']
        ground_truth = vis['ground_truth']
        filename = vis['filename']

        plt.figure(figsize=(12, 4))
        for i, (title, img) in enumerate(zip(
            ['Image', 'Prediction', 'Ground Truth'],
            [image, prediction, ground_truth]
        )):
            plt.subplot(1, 3, i + 1)

            if isinstance(img, torch.Tensor):
                img = img.cpu()

            if img.ndim == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)  # CHW → HWC
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)       # 1CH → HW

            img = img.numpy()
            plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
            plt.title(title)
            plt.axis('off')

        suptitle = f"Visualization - {filename}"
        if fold is not None:
            suptitle += f" | Fold {fold}"
        plt.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        plt.show()
