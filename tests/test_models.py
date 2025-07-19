"""Tests for model architecture."""

import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model import UNet


class TestUNet:
    """Test cases for U-Net model."""
    
    def test_unet_initialization(self):
        """Test U-Net model initialization."""
        model = UNet(in_channels=1, out_channels=1)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_unet_forward_pass(self):
        """Test U-Net forward pass."""
        model = UNet(in_channels=1, out_channels=1)
        
        # Test with batch of images
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 1, 256, 256)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_unet_different_input_sizes(self):
        """Test U-Net with different input sizes."""
        model = UNet(in_channels=1, out_channels=1)
        
        # Test multiple input sizes
        sizes = [128, 256, 512]
        
        for size in sizes:
            input_tensor = torch.randn(1, 1, size, size)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, 1, size, size)
    
    def test_unet_parameter_count(self):
        """Test U-Net parameter count is reasonable."""
        model = UNet(in_channels=1, out_channels=1)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check parameter count is reasonable (not too small, not too large)
        assert 1_000_000 < total_params < 100_000_000
        assert total_params == trainable_params  # All parameters should be trainable