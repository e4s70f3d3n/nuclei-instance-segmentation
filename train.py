# train.py
import os
import numpy as np
import torch
import sys
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import logging
from scipy.ndimage import label
from monai.metrics import DiceMetric, MeanIoU, PanopticQualityMetric
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import wandb
from IPython.display import Image, display

from utils.loss_functions import DiceLoss, clDiceLoss, FocalLoss, MCELoss
from utils.visualize import visualize_predictions, display_visualizations_inline, save_visualizations_to_drive
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
from prepare_data import NuInsSegDatasetV2
from lora_sam import AdaptiveLoRA_EfficientSAM

def get_loss(cfg):
    if cfg.loss_type == 'dice':
        return DiceLoss()
    elif cfg.loss_type == 'cldice':
        return clDiceLoss()
    elif cfg.loss_type == 'focal':
        return FocalLoss()
    elif cfg.loss_type == 'mce':
        return MCELoss()
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss_type}")

def train_one_epoch(network, data_loader, loss_function, optim, lr_scheduler, device, config, use_augmentation=True):
    """Train the model for one epoch with configurable augmentation and dynamic loss handling"""
    network.train()
    total_loss = 0.0
    processed_batches = 0

    progress_bar = tqdm(enumerate(data_loader), desc="Training batches", total=len(data_loader))
    
    for batch_idx, sample in progress_bar:
        input_images = sample['image'].to(device).float()
        target_segmentations = sample['binary_mask'].to(device).float()
        instance_masks = sample['instance_mask'].to(device)

        # Ensure correct dimensions
        if input_images.dim() == 3:
            input_images = input_images.unsqueeze(0)
        if target_segmentations.dim() == 2:
            target_segmentations = target_segmentations.unsqueeze(0)

        # Prepare prompt data
        point_prompts = sample['prompt_points'].to(device).float()
        prompt_targets = sample['prompt_labels'].to(device).float()

        if len(point_prompts.shape) == 2:
            point_prompts = point_prompts.unsqueeze(0)
        point_prompts = point_prompts.unsqueeze(1)

        if len(prompt_targets.shape) == 1:
            prompt_targets = prompt_targets.unsqueeze(0)
        prompt_targets = prompt_targets.unsqueeze(1)

        try:
            # Forward pass
            output_masks, iou_scores = network(input_images, point_prompts, prompt_targets)

            # Select best mask based on IoU predictions if available
            if iou_scores is not None:
                best_indices = iou_scores.argmax(dim=2)
                batch_size, num_queries = best_indices.shape
                selected_masks = torch.zeros(batch_size, num_queries, 
                                            output_masks.shape[3], 
                                            output_masks.shape[4], 
                                            device=device)

                for b in range(batch_size):
                    for q in range(num_queries):
                        selected_masks[b, q] = output_masks[b, q, best_indices[b, q]]

                if num_queries == 1:
                    selected_masks = selected_masks.squeeze(1)
            else:
                selected_masks = output_masks[:, 0, 0]

            # Resize if needed
            if selected_masks.shape[-2:] != target_segmentations.shape[-2:]:
                selected_masks = F.interpolate(
                    selected_masks.unsqueeze(1),
                    size=target_segmentations.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            # Apply activation if needed
            if config.loss_type in ["dice", "cldice"]:
                selected_masks = torch.sigmoid(selected_masks)

            # Compute loss
            if config.loss_type in ["dice", "cldice"]:
                batch_loss = loss_function(selected_masks, target_segmentations)
            elif config.loss_type in ["focal", "mce"]:
                batch_loss = loss_function(selected_masks, target_segmentations.long())
            else:
                batch_loss = loss_function(selected_masks, target_segmentations, instance_masks)

            # Backward pass
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
            lr_scheduler.step()

            total_loss += batch_loss.item()
            processed_batches += 1

            # Update progress bar with current loss
            progress_bar.set_postfix({'batch_loss': batch_loss.item()})

        except Exception as error:
            print(f"\nError processing batch {batch_idx}: {str(error)}")
            import traceback
            traceback.print_exc()
            continue

    average_loss = total_loss / max(1, processed_batches)
    print(f"\nEpoch completed - Average loss: {average_loss:.4f}")
    return average_loss


@torch.no_grad()
def evaluate_model(network, validation_loader, loss_fn, device, configuration):
    """Assess model performance on validation dataset with comprehensive metrics"""
    network.eval()
    total_loss = 0.0
    processed_batches = 0

    # Initialize metrics
    dice_calculator = DiceMetric(include_background=False, reduction="mean")
    iou_calculator = MeanIoU()
    panoptic_scorer = PanopticQualityMetric(num_classes=1)

    dice_results = []
    iou_results = []
    panoptic_scores = []

    progress_bar = tqdm(enumerate(validation_loader), 
                       desc="Validating batches", 
                       total=len(validation_loader))

    with torch.no_grad():
        for batch_idx, sample in progress_bar:
            true_panoptic = sample['panoptic_mask'].to(device)
            input_images = sample['image'].to(device).float()
            true_masks = sample['binary_mask'].to(device).float()
            instance_labels = sample['instance_mask'].to(device)

            # Prepare prompt data
            point_prompts = sample['prompt_points'].to(device).float()
            prompt_targets = sample['prompt_labels'].to(device).float()

            if len(point_prompts.shape) == 2:
                point_prompts = point_prompts.unsqueeze(0)
            point_prompts = point_prompts.unsqueeze(1)

            if len(prompt_targets.shape) == 1:
                prompt_targets = prompt_targets.unsqueeze(0)
            prompt_targets = prompt_targets.unsqueeze(1)

            try:
                # Model prediction
                predicted_segmentations, iou_estimates = network(input_images, point_prompts, prompt_targets)

                # Select best mask based on IoU predictions if available
                if iou_estimates is not None:
                    optimal_indices = iou_estimates.argmax(dim=2)
                    batch_size, num_queries = optimal_indices.shape
                    selected_segmentations = torch.zeros(
                        batch_size, num_queries, 
                        predicted_segmentations.shape[3], 
                        predicted_segmentations.shape[4], 
                        device=device
                    )

                    for b_idx in range(batch_size):
                        for q_idx in range(num_queries):
                            selected_segmentations[b_idx, q_idx] = \
                                predicted_segmentations[b_idx, q_idx, optimal_indices[b_idx, q_idx]]

                    if num_queries == 1:
                        selected_segmentations = selected_segmentations.squeeze(1)
                else:
                    selected_segmentations = predicted_segmentations[:, 0, 0]

                # Resize predictions if needed
                if selected_segmentations.shape[-2:] != true_masks.shape[-2:]:
                    selected_segmentations = F.interpolate(
                        selected_segmentations.unsqueeze(1),
                        size=true_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                # Apply activation if needed
                if configuration.loss_type in ["dice", "cldice"]:
                    selected_segmentations = torch.sigmoid(selected_segmentations)

                # Calculate loss
                if configuration.loss_type in ["dice", "cldice"]:
                    batch_loss = loss_fn(selected_segmentations, true_masks)
                elif configuration.loss_type in ["focal", "mce"]:
                    batch_loss = loss_fn(selected_segmentations, true_masks.long())
                else:
                    batch_loss = loss_fn(selected_segmentations, true_masks, instance_labels)

                # Convert to binary predictions
                binary_predictions = (selected_segmentations > 0.5).float()
                current_batch_size = binary_predictions.shape[0]
                height, width = binary_predictions.shape[-2:]
                predicted_panoptic = torch.zeros((current_batch_size, 2, height, width), device=device)

                # Process each sample in batch
                for sample_idx in range(current_batch_size):
                    sample_prediction = binary_predictions[sample_idx].cpu().numpy()
                    labeled_prediction, _ = label(sample_prediction)
                    instance_prediction = torch.from_numpy(labeled_prediction).float().to(device)
                    predicted_panoptic[sample_idx, 0] = instance_prediction
                    predicted_panoptic[sample_idx, 1] = (instance_prediction > 0).float()

                # Calculate panoptic quality
                panoptic_scorer.reset()
                panoptic_scorer(y_pred=predicted_panoptic, y=true_panoptic)
                current_pq = panoptic_scorer.aggregate().mean().item()
                panoptic_scores.append(current_pq)

                # Prepare for metric calculation
                predictions_onehot = binary_predictions.unsqueeze(0)
                groundtruth_onehot = true_masks.unsqueeze(0)

                # Calculate Dice score
                dice_calculator.reset()
                dice_calculator(y_pred=predictions_onehot.float(), y=groundtruth_onehot.float())
                current_dice = dice_calculator.aggregate().mean().item()
                dice_results.append(current_dice)

                # Calculate IoU score
                iou_calculator.reset()
                iou_calculator(y_pred=predictions_onehot.float(), y=groundtruth_onehot.float())
                current_iou = iou_calculator.aggregate().mean().item()
                iou_results.append(current_iou)

                total_loss += batch_loss.item()
                processed_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'batch_loss': batch_loss.item(),
                    'dice': current_dice,
                    'iou': current_iou
                })

            except Exception as error:
                print(f"\nError validating batch {batch_idx}: {str(error)}")
                import traceback
                traceback.print_exc()
                continue

    # Calculate final metrics
    average_loss = total_loss / max(1, processed_batches)
    mean_dice = np.mean(dice_results) if dice_results else 0.0
    mean_iou = np.mean(iou_results) if iou_results else 0.0
    mean_pq = np.mean(panoptic_scores) if panoptic_scores else 0.0

    print(f"\nValidation completed - "
          f"Loss: {average_loss:.4f}, "
          f"Dice: {mean_dice:.4f}, "
          f"IoU: {mean_iou:.4f}, "
          f"PQ: {mean_pq:.4f}")

    return average_loss, mean_dice, mean_iou, mean_pq

def build_sam():
    return build_efficient_sam(
        encoder_patch_embed_dim=192, encoder_num_heads=3,
        checkpoint="/content/drive/MyDrive/medical_image_computing/histo-segmentation/EfficientSAM/weights/efficient_sam_vitt.pt"
    ).eval()
