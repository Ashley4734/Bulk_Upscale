#!/usr/bin/env python3
"""
Configuration file for Bulk Image Upscaler
"""

from pathlib import Path

# Default image storage path
# This is the default input directory used when no -i/--input argument is provided
DEFAULT_IMAGE_PATH = "/Users/ashleyharris/Desktop/Etsy_Business_Mockups"

# Default output directory name (relative to input directory)
# When no output is specified, images will be saved to <input_dir>_upscaled
DEFAULT_OUTPUT_SUFFIX = "_upscaled"

# Default upscaling settings
DEFAULT_SCALE_FACTOR = 2.0
DEFAULT_METHOD = "lanczos"
DEFAULT_QUALITY = 95
DEFAULT_WORKERS = 4
