"""Beltrami image denoising on the original 0–255 intensity scale."""

from .classical import beltrami_original, patch_beltrami, psnr

__all__ = ["beltrami_original", "patch_beltrami", "psnr"]
__version__ = "1.0.0"
