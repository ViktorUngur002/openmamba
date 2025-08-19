import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba2 import VMAMBA2, Backbone_VMAMBA2

class MambaFeatureDecoder(nn.Module):
    def __init__(self, in_channels=[48, 96, 192, 384], hidden_dim=256, out_dim=128, out_size=32):
        """
        A decoder for Mamba features that preserves semantics better than direct interpolation.
        
        Args:
            in_channels: List of channel dimensions for each feature level
            hidden_dim: Internal dimension for feature processing
            out_dim: Output feature dimension
            out_size: Output spatial size (height and width)
        """
        super().__init__()
        self.out_size = out_size
        
        # Adapters to convert features to a common dimension
        self.adapters = nn.ModuleList([
            nn.Conv2d(in_channels[0], hidden_dim, kernel_size=1),
            nn.Conv2d(in_channels[1], hidden_dim, kernel_size=1),
            nn.Conv2d(in_channels[2], hidden_dim, kernel_size=1),
            nn.Conv2d(in_channels[3], hidden_dim, kernel_size=1)
        ])
        
        # Upsampling blocks - using transposed convolutions instead of interpolation
        # to learn better upsampling that preserves semantics
        self.upsample_blocks = nn.ModuleList([
            # 1x1 → 2x2
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            # 2x2 → 4x4
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            # 4x4 → 8x8
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            # 8x8 → 16x16
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            # 16x16 → 32x32
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        ])
        
        # Feature fusion blocks for each level
        self.fusion_blocks = nn.ModuleList([
            # After first upsample: combine 1x1 and 2x2
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 2, 2]),
                nn.GELU()
            ),
            # After second upsample: combine with 4x4
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 4, 4]),
                nn.GELU()
            ),
            # After third upsample: combine with 8x8
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 8, 8]),
                nn.GELU()
            ),
            # Final refinement at 16x16 (no direct input at this resolution)
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 16, 16]),
                nn.GELU()
            )
        ])
        
        # Final projection layer
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.LayerNorm([out_dim, 32, 32]),
            nn.GELU()
        )
        
        # SSM-inspired sequence modeling to maintain Mamba-like processing
        # This helps preserve the temporal/sequential nature of Mamba features
        self.seq_modeling = nn.ModuleList([
            SequenceEnhancer(hidden_dim) for _ in range(4)
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps [f1, f2, f3, f4] with shapes:
                f1: [B, 48, 8, 8]
                f2: [B, 96, 4, 4]
                f3: [B, 192, 2, 2]
                f4: [B, 384, 1, 1]
        
        Returns:
            Tensor of shape [B, out_dim, out_size, out_size]
        """
        # Extract features from different levels
        f1, f2, f3, f4 = features
        
        # Convert all features to common dimension
        f1 = self.adapters[0](f1)  # [B, hidden_dim, 8, 8]
        f2 = self.adapters[1](f2)  # [B, hidden_dim, 4, 4]
        f3 = self.adapters[2](f3)  # [B, hidden_dim, 2, 2]
        f4 = self.adapters[3](f4)  # [B, hidden_dim, 1, 1]
        
        # Start with deepest features and progressively upsample and merge
        x = f4  # [B, hidden_dim, 1, 1]
        
        # First upsample: 1x1 → 2x2 and merge with f3
        x = self.upsample_blocks[0](x)  # [B, hidden_dim, 2, 2]
        x = torch.cat([x, f3], dim=1)  # [B, hidden_dim*2, 2, 2]
        x = self.fusion_blocks[0](x)    # [B, hidden_dim, 2, 2]
        x = self.seq_modeling[0](x)     # Apply sequence enhancement
        
        # Second upsample: 2x2 → 4x4 and merge with f2
        x = self.upsample_blocks[1](x)  # [B, hidden_dim, 4, 4]
        x = torch.cat([x, f2], dim=1)  # [B, hidden_dim*2, 4, 4]
        x = self.fusion_blocks[1](x)    # [B, hidden_dim, 4, 4]
        x = self.seq_modeling[1](x)     # Apply sequence enhancement
        
        # Third upsample: 4x4 → 8x8 and merge with f1
        x = self.upsample_blocks[2](x)  # [B, hidden_dim, 8, 8]
        x = torch.cat([x, f1], dim=1)  # [B, hidden_dim*2, 8, 8]
        x = self.fusion_blocks[2](x)    # [B, hidden_dim, 8, 8]
        x = self.seq_modeling[2](x)     # Apply sequence enhancement
        
        # Fourth upsample: 8x8 → 16x16 (no direct feature to merge)
        x = self.upsample_blocks[3](x)  # [B, hidden_dim, 16, 16]
        x = self.fusion_blocks[3](x)    # [B, hidden_dim, 16, 16]
        x = self.seq_modeling[3](x)     # Apply sequence enhancement
        
        # Final upsample: 16x16 → 32x32
        x = self.upsample_blocks[4](x)  # [B, hidden_dim, 32, 32]
        
        # Project to final output dimension
        x = self.output_proj(x)  # [B, out_dim, 32, 32]
        
        return x


class SequenceEnhancer(nn.Module):
    """
    A lightweight module inspired by state-space models to preserve
    the sequential nature of Mamba features during upsampling.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Sequence modeling components
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, 2*dim)
        self.proj_out = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Normalize
        x_norm = self.norm(x_seq)
        
        # Project and split
        x_proj = self.proj_in(x_norm)
        x_1, x_2 = x_proj.chunk(2, dim=-1)
        
        # Apply gating mechanism (similar to how Mamba controls information flow)
        gate = torch.sigmoid(self.gate(x_norm))
        
        # Element-wise multiplication (simulating selective update)
        x_gated = x_1 * torch.tanh(x_2) * gate
        
        # Project back
        x_out = self.proj_out(x_gated)
        
        # Add residual connection
        x_out = x_out + x_seq
        
        # Reshape back to spatial
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


# Example usage
def test_decoder():
    # Create sample features
    batch_size = 128
    f1 = torch.randn(batch_size, 48, 8, 8)
    f2 = torch.randn(batch_size, 96, 4, 4)
    f3 = torch.randn(batch_size, 192, 2, 2)
    f4 = torch.randn(batch_size, 384, 1, 1)
    
    # Create decoder
    decoder = MambaFeatureDecoder()
    
    # Forward pass
    output = decoder([f1, f2, f3, f4])
    
    print(f"Output shape: {output.shape}")  # Should be [128, 128, 32, 32]
    
    return output

decoder = MambaFeatureDecoder()
model = Backbone_VMAMBA2(
            image_size=32,
            patch_size=2,
            in_chans=784,
            embed_dim=48,
            depths=[ 2, 2, 8, 4],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.,
            drop_rate=0,
            drop_path_rate=0.2,
            simple_downsample=False,
            simple_patch_embed=False,
            ssd_expansion=2,
            ssd_ngroups=1,
            ssd_chunk_size=256,
            linear_attn_duality = True,
            lepe=False,
            attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
            bidirection=False,
            d_state=48,
            ssd_positve_dA = True,
        )

dummy_input = torch.randn(128, 784, 32, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
decoder.to(device)

# Move the dummy input tensor to the same device as the model
dummy_input = dummy_input.to(device)

output_features = model(dummy_input)

o1, o2, o3, o4 = output_features

print(f"The shape of passing thorugh classig VisionMamba {o1.shape}")
print(f"The shape of passing thorugh classig VisionMamba {o2.shape}")
print(f"The shape of passing thorugh classig VisionMamba {o3.shape}")
print(f"The shape of passing thorugh classig VisionMamba {o4.shape}")

final_features = decoder(output_features)

print(f"The shape of the final features {final_features.shape}")