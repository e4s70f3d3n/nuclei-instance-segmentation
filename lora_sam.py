import math
import torch
import torch.nn as nn
from typing import Optional, List, Dict

class LoRA_QKV_Projection(nn.Module):
    """Alternative implementation of LoRA for QKV projections with different architecture"""
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 4.0,
        device: Optional[str] = None
    ):
        super().__init__()
        self.original_proj = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Using Conv1d instead of Linear for the LoRA projections
        self.lora_q = nn.Sequential(
            nn.Conv1d(self.in_features, rank, 1, bias=False),
            nn.Conv1d(rank, self.in_features, 1, bias=False)
        ).to(device)
        
        self.lora_v = nn.Sequential(
            nn.Conv1d(self.in_features, rank, 1, bias=False),
            nn.Conv1d(rank, self.in_features, 1, bias=False)
        ).to(device)
        
        # Initialize parameters
        self._init_lora_weights()
        
    def _init_lora_weights(self):
        # Different initialization strategy
        nn.init.normal_(self.lora_q[0].weight, std=1/math.sqrt(self.rank))
        nn.init.zeros_(self.lora_q[1].weight)
        nn.init.normal_(self.lora_v[0].weight, std=1/math.sqrt(self.rank))
        nn.init.zeros_(self.lora_v[1].weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original projection
        qkv = self.original_proj(x)  # B,N,3*C
        
        # Get dimensions
        B, N, _ = qkv.shape
        C = self.in_features
        
        # Reshape for conv1d (B,C,N)
        x_reshaped = x.transpose(1, 2)
        
        # Compute LoRA contributions
        lora_q = self.lora_q(x_reshaped).transpose(1, 2)  # B,N,C
        lora_v = self.lora_v(x_reshaped).transpose(1, 2)  # B,N,C
        
        # Split and add LoRA contributions
        qkv = qkv.view(B, N, 3, C)
        qkv[:, :, 0] += self.scaling * lora_q  # Add to query
        qkv[:, :, 2] += self.scaling * lora_v  # Add to value
        
        return qkv.view(B, N, 3 * C)

class AdaptiveLoRA_EfficientSAM(nn.Module):
    """Alternative LoRA implementation for EfficientSAM with different features:
    
    - Uses Conv1d instead of Linear for LoRA projections
    - Implements layer-wise learning rates
    - Adds optional dropout for LoRA paths
    - Implements gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        config,
        sam_model,
        rank: int = 4,
        alpha: float = 4.0,
        lora_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.sam = sam_model
        self.image_encoder = self.sam.image_encoder
        self.device = config.device
        self.use_checkpoint = use_checkpoint
        
        # Freeze original model
        for param in self.sam.parameters():
            param.requires_grad_(False)
            
        # Default to all layers if not specified
        if lora_layers is None:
            lora_layers = list(range(len(self.image_encoder.blocks)))
            
        self.lora_layers = lora_layers
        self.rank = rank
        self.alpha = alpha
        
        # Store original layers and create LoRA versions
        self.original_layers = nn.ModuleDict()
        self.lora_modules = nn.ModuleList()
        
        for layer_idx in lora_layers:
            block = self.image_encoder.blocks[layer_idx]
            orig_qkv = block.attn.qkv
            
            # Store original layer
            self.original_layers[f'layer_{layer_idx}_qkv'] = orig_qkv
            
            # Create LoRA version
            lora_qkv = LoRA_QKV_Projection(
                orig_qkv,
                rank=rank,
                alpha=alpha,
                device=self.device
            )
            
            # Replace original with LoRA version
            block.attn.qkv = lora_qkv
            self.lora_modules.append(lora_qkv)
            
        # Dropout for LoRA paths
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, batched_images, batched_points, batched_point_labels, scale_to_original_image_size=True):
        """Forward pass with optional gradient checkpointing"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.sam,
                batched_images,
                batched_points,
                batched_point_labels,
                scale_to_original_image_size
            )
        return self.sam(batched_images, batched_points, batched_point_labels, scale_to_original_image_size)
    
    def get_lora_params(self) -> List[Dict]:
        """Get LoRA parameters with layer-wise learning rate scaling"""
        params = []
        for layer_idx, lora_module in zip(self.lora_layers, self.lora_modules):
            # Scale learning rate by layer depth
            lr_scale = 1.0 / (layer_idx + 1)
            
            params.append({
                'params': lora_module.lora_q.parameters(),
                'lr_scale': lr_scale
            })
            params.append({
                'params': lora_module.lora_v.parameters(),
                'lr_scale': lr_scale
            })
        return params
    
    def save_lora_state(self, filename: str):
        """Save LoRA state with additional metadata"""
        state = {
            'rank': self.rank,
            'alpha': self.alpha,
            'lora_layers': self.lora_layers,
            'state_dict': {
                name: param.data for name, param in self.named_parameters()
                if 'lora_q' in name or 'lora_v' in name
            }
        }
        torch.save(state, filename)
    
    def load_lora_state(self, filename: str):
        """Load LoRA state with compatibility checks"""
        device = next(self.parameters()).device
        state = torch.load(filename, map_location=device)
        
        # Check compatibility
        if state['rank'] != self.rank:
            print(f"Warning: Loaded rank {state['rank']} doesn't match current rank {self.rank}")
        if state['alpha'] != self.alpha:
            print(f"Warning: Loaded alpha {state['alpha']} doesn't match current alpha {self.alpha}")
        
        # Load parameters
        current_state = self.state_dict()
        for name, param in state['state_dict'].items():
            if name in current_state:
                current_state[name].copy_(param)
            else:
                print(f"Warning: Parameter {name} not found in model")
    
    def get_image_embeddings(self, batched_images):
            """Get image embeddings from the EfficientSAM model"""
            return self.sam.get_image_embeddings(batched_images)
    
    def predict_masks(self, image_embeddings, batched_points, batched_point_labels, 
                     multimask_output, input_h, input_w, output_h=-1, output_w=-1):
        """Predict masks from image embeddings and prompts"""
        return self.sam.predict_masks(
            image_embeddings, 
            batched_points, 
            batched_point_labels, 
            multimask_output, 
            input_h, 
            input_w, 
            output_h, 
            output_w
        )
