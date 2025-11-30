#!/usr/bin/env python3
"""
Bulk Image Upscaler
A powerful tool to upscale multiple images using AI or traditional methods.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import concurrent.futures
from typing import List, Tuple
import config
import cv2
import numpy as np

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Resampling methods (for traditional upscaling)
RESAMPLING_METHODS = {
    'lanczos': Image.Resampling.LANCZOS,
    'bicubic': Image.Resampling.BICUBIC,
    'bilinear': Image.Resampling.BILINEAR,
    'nearest': Image.Resampling.NEAREST,
}

# AI upscaling models
AI_MODELS = {
    'realesrgan-x4': 'RealESRGAN_x4plus',
    'realesrgan-x2': 'RealESRGAN_x2plus',
    'realesr-animevideo': 'realesr-animevideov3',
}

# Try to import Real-ESRGAN
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class ImageUpscaler:
    """Handles bulk image upscaling operations."""

    def __init__(self, scale_factor: float = 2.0, method: str = 'lanczos',
                 quality: int = 95, workers: int = 4, use_ai: bool = False,
                 ai_model: str = 'realesrgan-x4'):
        """
        Initialize the upscaler.

        Args:
            scale_factor: Factor by which to upscale images (e.g., 2.0 for 2x) - ignored for AI
            method: Resampling method ('lanczos', 'bicubic', 'bilinear', 'nearest')
            quality: JPEG quality for output (1-100)
            workers: Number of parallel workers for processing
            use_ai: Use AI-based upscaling (requires Real-ESRGAN)
            ai_model: AI model to use ('realesrgan-x4', 'realesrgan-x2', 'realesr-animevideo')
        """
        self.scale_factor = scale_factor
        self.method = RESAMPLING_METHODS.get(method.lower(), Image.Resampling.LANCZOS)
        self.quality = max(1, min(100, quality))
        self.workers = workers
        self.use_ai = use_ai
        self.ai_model = ai_model
        self.upsampler = None

        # Initialize AI upscaler if requested
        if self.use_ai:
            if not AI_AVAILABLE:
                raise ImportError(
                    "AI upscaling requires Real-ESRGAN. Install with:\n"
                    "pip install realesrgan basicsr facexlib gfpgan opencv-python"
                )
            self._init_ai_upscaler()

    def _init_ai_upscaler(self):
        """Initialize the Real-ESRGAN upscaler."""
        model_name = AI_MODELS.get(self.ai_model, 'RealESRGAN_x4plus')

        # Model configurations
        if 'x4' in self.ai_model:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif 'x2' in self.ai_model:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4

        # Model download paths (will auto-download)
        model_urls = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        }

        model_path = model_urls.get(model_name)

        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,  # 0 for no tiling, or set to 400-800 for GPU memory constraints
            tile_pad=10,
            pre_pad=0,
            half=False  # Set to True if using GPU with FP16 support
        )

    def upscale_image(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Upscale a single image.

        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image

        Returns:
            Tuple of (success, message)
        """
        try:
            if self.use_ai:
                return self._upscale_with_ai(input_path, output_path)
            else:
                return self._upscale_traditional(input_path, output_path)
        except Exception as e:
            return False, f"✗ Failed: {input_path.name} - {str(e)}"

    def _upscale_with_ai(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Upscale using AI (Real-ESRGAN)."""
        # Read image with OpenCV
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        original_height, original_width = img.shape[:2]

        # Enhance with Real-ESRGAN
        output, _ = self.upsampler.enhance(img, outscale=self.upsampler.scale)

        new_height, new_width = output.shape[:2]

        # Save the output
        if output_path.suffix.lower() in {'.jpg', '.jpeg'}:
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        elif output_path.suffix.lower() == '.png':
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(output_path), output)

        return True, f"✓ AI Upscaled: {input_path.name} ({original_width}x{original_height} → {new_width}x{new_height})"

    def _upscale_traditional(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Upscale using traditional interpolation methods."""
        # Open image
        with Image.open(input_path) as img:
            # Convert RGBA to RGB if saving as JPEG
            if output_path.suffix.lower() in {'.jpg', '.jpeg'} and img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background

            original_width, original_height = img.width, img.height

            # Calculate new dimensions
            new_width = int(img.width * self.scale_factor)
            new_height = int(img.height * self.scale_factor)

            # Upscale
            upscaled = img.resize((new_width, new_height), self.method)

            # Save with appropriate settings
            save_kwargs = {}
            if output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                save_kwargs['quality'] = self.quality
                save_kwargs['optimize'] = True
            elif output_path.suffix.lower() == '.png':
                save_kwargs['optimize'] = True

            upscaled.save(output_path, **save_kwargs)

        return True, f"✓ Upscaled: {input_path.name} ({original_width}x{original_height} → {new_width}x{new_height})"

    def process_directory(self, input_dir: Path, output_dir: Path,
                         recursive: bool = False) -> None:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images to upscale
            output_dir: Directory to save upscaled images
            recursive: Whether to process subdirectories
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        if recursive:
            image_files = []
            for ext in SUPPORTED_FORMATS:
                image_files.extend(input_dir.rglob(f'*{ext}'))
                image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
        else:
            image_files = []
            for ext in SUPPORTED_FORMATS:
                image_files.extend(input_dir.glob(f'*{ext}'))
                image_files.extend(input_dir.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} image(s) to process")
        if self.use_ai:
            print(f"Mode: AI Upscaling ({self.ai_model})")
            print(f"Scale: {self.upsampler.scale}x (determined by model)")
        else:
            print(f"Mode: Traditional ({[k for k, v in RESAMPLING_METHODS.items() if v == self.method][0]})")
            print(f"Scale factor: {self.scale_factor}x")
        print(f"Workers: {self.workers}")
        print("-" * 60)

        # Process images in parallel
        tasks = []
        for img_path in image_files:
            # Preserve relative directory structure if recursive
            if recursive:
                relative_path = img_path.relative_to(input_dir)
                out_path = output_dir / relative_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = output_dir / img_path.name

            tasks.append((img_path, out_path))

        # Execute with progress
        success_count = 0
        fail_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.upscale_image, inp, out): (inp, out)
                      for inp, out in tasks}

            for future in concurrent.futures.as_completed(futures):
                success, message = future.result()
                print(message)
                if success:
                    success_count += 1
                else:
                    fail_count += 1

        print("-" * 60)
        print(f"Completed: {success_count} succeeded, {fail_count} failed")
        print(f"Output directory: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Bulk Image Upscaler - Upscale multiple images at once',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale all images in current directory by 2x
  python upscale.py -i ./images -o ./output

  # Upscale by 4x with bicubic interpolation
  python upscale.py -i ./images -o ./output -s 4 -m bicubic

  # Process subdirectories recursively
  python upscale.py -i ./images -o ./output -r

  # Use 8 parallel workers for faster processing
  python upscale.py -i ./images -o ./output -w 8
        """
    )

    parser.add_argument('-i', '--input', type=str, required=False,
                       help=f'Input directory containing images (default: {config.DEFAULT_IMAGE_PATH})')
    parser.add_argument('-o', '--output', type=str, required=False,
                       help='Output directory for upscaled images (default: <input_dir>_upscaled)')
    parser.add_argument('-s', '--scale', type=float, default=2.0,
                       help='Scale factor for traditional upscaling (default: 2.0, ignored for AI)')
    parser.add_argument('-m', '--method', type=str, default='lanczos',
                       choices=['lanczos', 'bicubic', 'bilinear', 'nearest'],
                       help='Resampling method for traditional upscaling (default: lanczos)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                       help='JPEG quality 1-100 (default: 95)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Process subdirectories recursively')
    parser.add_argument('-w', '--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--ai', action='store_true',
                       help='Use AI-based upscaling (Real-ESRGAN) for better quality')
    parser.add_argument('--ai-model', type=str, default='realesrgan-x4',
                       choices=['realesrgan-x4', 'realesrgan-x2', 'realesr-animevideo'],
                       help='AI model to use (default: realesrgan-x4 for 4x upscale)')

    args = parser.parse_args()

    # Use configured default path if no input provided
    if args.input is None:
        input_path = config.DEFAULT_IMAGE_PATH
        print(f"Using configured default image path: {input_path}")
    else:
        input_path = args.input

    # Validate paths
    input_dir = Path(input_path).expanduser().resolve()

    # Use default output path if not provided
    if args.output is None:
        output_dir = Path(str(input_dir) + config.DEFAULT_OUTPUT_SUFFIX)
        print(f"Using default output directory: {output_dir}")
    else:
        output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory")
        sys.exit(1)

    if args.scale <= 0:
        print("Error: Scale factor must be positive")
        sys.exit(1)

    # Warn if input and output are the same
    if input_dir == output_dir:
        print("Warning: Input and output directories are the same. This will overwrite original images!")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)

    # Create upscaler and process
    upscaler = ImageUpscaler(
        scale_factor=args.scale,
        method=args.method,
        quality=args.quality,
        workers=args.workers,
        use_ai=args.ai,
        ai_model=args.ai_model
    )

    upscaler.process_directory(input_dir, output_dir, args.recursive)


if __name__ == '__main__':
    main()
