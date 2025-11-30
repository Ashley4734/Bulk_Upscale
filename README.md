# Bulk Image Upscaler

A fast, efficient command-line tool for upscaling multiple images at once on macOS (and other platforms).

## Features

- **AI-Powered Upscaling**: Use Real-ESRGAN for superior quality enhancement with AI
- **Bulk Processing**: Upscale hundreds of images in one command
- **Multiple Methods**: Choose between AI upscaling or traditional resampling (Lanczos, Bicubic, etc.)
- **Parallel Processing**: Use multiple CPU cores for faster processing
- **Recursive Support**: Process entire directory trees
- **Format Support**: Works with JPG, PNG, BMP, TIFF, and WebP
- **Quality Control**: Adjustable JPEG quality settings
- **Preserves Structure**: Maintains directory hierarchy when processing recursively

## Installation

### Prerequisites

- Python 3.7 or higher (macOS comes with Python 3 pre-installed)
- pip (Python package manager)

### Setup

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/Ashley4734/Bulk_Upscale.git
   cd Bulk_Upscale
   ```

2. **Create a virtual environment (recommended for macOS/modern Python):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   **For traditional upscaling only** (faster install, smaller footprint):
   ```bash
   pip install Pillow
   ```

   **For AI upscaling** (recommended for best quality):
   ```bash
   pip install -r requirements.txt
   ```

   Note: AI upscaling requires PyTorch and Real-ESRGAN, which will download ~500MB of packages. The AI model weights (~17MB for x4, ~11MB for x2) will be automatically downloaded on first use.

4. **Make the script executable (optional):**
   ```bash
   chmod +x upscale.py
   ```

**Note for macOS users:** Modern macOS systems require using a virtual environment for Python packages. If you get an "externally-managed-environment" error, make sure you've activated the virtual environment with `source venv/bin/activate` before installing packages.

## Configuration

You can configure a default image storage path by editing the `config.py` file:

```python
# Default image storage path
DEFAULT_IMAGE_PATH = "/Users/ashleyharris/Desktop/Etsy_Business_Mockups"
```

With this configured, you can run the upscaler without specifying the input directory:

```bash
# Uses the configured default path
python3 upscale.py

# Output directory defaults to <input_dir>_upscaled
# You can still override with -o flag
python3 upscale.py -o ./my_custom_output
```

You can still override the configured path with the `-i` flag:

```bash
python3 upscale.py -i ./different_folder
```

## Usage

**Important:** If you're using a virtual environment, make sure to activate it first:
```bash
source venv/bin/activate
```

### Basic Usage

```bash
# With configured default path (see Configuration section)
python3 upscale.py

# Or specify input and output directories
python3 upscale.py -i <input_directory> -o <output_directory>
```

### Examples

**AI Upscaling (Recommended for Quality):**

```bash
# AI upscale by 4x (best quality, uses Real-ESRGAN)
python3 upscale.py -i ./photos -o ./upscaled_photos --ai

# AI upscale by 2x (faster, smaller output files)
python3 upscale.py -i ./photos -o ./upscaled_photos --ai --ai-model realesrgan-x2

# AI upscale for anime/animation (specialized model)
python3 upscale.py -i ./anime -o ./upscaled_anime --ai --ai-model realesr-animevideo

# AI upscale with high JPEG quality
python3 upscale.py -i ./photos -o ./upscaled_photos --ai -q 98
```

**Traditional Upscaling (Faster, Lower Quality):**

```bash
# Upscale all images in a folder by 2x (default)
python3 upscale.py -i ./photos -o ./upscaled_photos

# Upscale by 4x with bicubic interpolation
python3 upscale.py -i ./photos -o ./upscaled_photos -s 4 -m bicubic

# Process subdirectories recursively
python3 upscale.py -i ./photos -o ./upscaled_photos -r

# Use 8 parallel workers for faster processing
python3 upscale.py -i ./photos -o ./upscaled_photos -w 8

# Upscale with custom JPEG quality
python3 upscale.py -i ./photos -o ./upscaled_photos -q 98
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input directory containing images | Configured in config.py |
| `-o, --output` | Output directory for upscaled images | `<input_dir>_upscaled` |
| `--ai` | Use AI-based upscaling (Real-ESRGAN) | False |
| `--ai-model` | AI model: `realesrgan-x4`, `realesrgan-x2`, `realesr-animevideo` | realesrgan-x4 |
| `-s, --scale` | Scale factor for traditional upscaling (ignored for AI) | 2.0 |
| `-m, --method` | Resampling method: `lanczos`, `bicubic`, `bilinear`, `nearest` | lanczos |
| `-q, --quality` | JPEG quality (1-100) | 95 |
| `-r, --recursive` | Process subdirectories recursively | False |
| `-w, --workers` | Number of parallel workers | 4 |

### AI vs Traditional Upscaling

**AI Upscaling (--ai flag):**
- Uses deep learning (Real-ESRGAN) to enhance image quality
- Actually adds realistic details and sharpness
- Best for: Photos, textures, and images where quality matters
- Slower but produces significantly better results
- Fixed scale factors (2x or 4x depending on model)
- Requires PyTorch and additional dependencies

**Traditional Upscaling (default):**
- Uses mathematical interpolation (Lanczos, Bicubic, etc.)
- Simply resizes images without adding detail
- Best for: Quick resizing, preparing images for specific dimensions
- Much faster, lower resource usage
- Flexible scale factors (any value)
- Minimal dependencies (just Pillow)

### Resampling Methods (Traditional Mode Only)

- **Lanczos** (recommended): Highest quality interpolation, best for photos
- **Bicubic**: Good quality, faster than Lanczos
- **Bilinear**: Moderate quality, faster processing
- **Nearest**: Lowest quality, fastest, good for pixel art

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Tips

1. **For best quality**: Use `--ai` flag with Real-ESRGAN for actual quality enhancement
2. **For maximum quality preservation**: Use AI upscaling with `-q 98` or `-q 100`
3. **For speed**: Use traditional upscaling with Bicubic method and 4-8 workers
4. **For large batches**: Enable recursive mode and let it run in the background
5. **GPU acceleration**: If you have an NVIDIA GPU, PyTorch will automatically use it for AI upscaling (much faster)
6. **Preserve originals**: Always use a different output directory to keep your original images safe
7. **First-time AI usage**: The model weights will be downloaded automatically (~17MB), which may take a minute

## Performance

**Traditional Upscaling:**
- Very fast, CPU-based
- ~50-100 photos (1920x1080) per minute at 2x scale
- ~20-40 photos per minute at 4x scale
- Speed scales with number of workers (use `-w` flag)

**AI Upscaling:**
- Slower but much higher quality
- CPU: ~2-5 images per minute (1920x1080)
- GPU (NVIDIA): ~10-30 images per minute (1920x1080)
- Processing time increases with image resolution
- First run downloads model weights (~17MB)

## Troubleshooting

**"No images found"**: Make sure your input directory contains supported image formats

**"Permission denied"**: Run with appropriate permissions or choose a different output directory

**Memory errors**: Reduce the number of workers or process smaller batches

**RGBA/RGB warnings**: The tool automatically converts RGBA images to RGB when saving as JPEG

**AI upscaling import errors**: Install AI dependencies with `pip install -r requirements.txt`

**Slow AI upscaling**: This is normal on CPU. For faster processing, use an NVIDIA GPU or switch to traditional upscaling

**PyTorch installation issues on Mac**: For Apple Silicon Macs, PyTorch will use CPU. For best AI performance, use a system with an NVIDIA GPU or use traditional upscaling for speed

## Examples of Real-World Use

```bash
# AI upscale vacation photos for printing (best quality)
python3 upscale.py -i ~/Pictures/Vacation2024 -o ~/Pictures/Vacation2024_AI --ai -q 98

# AI upscale low-resolution product photos for e-commerce
python3 upscale.py -i ~/Products -o ~/Products_HD --ai --ai-model realesrgan-x4

# Quick batch resize screenshots (traditional method for speed)
python3 upscale.py -i ~/Screenshots -o ~/Screenshots_HD -s 2 -m bicubic

# Process entire photo library recursively with AI enhancement
python3 upscale.py -i ~/Photos -o ~/Photos_Enhanced -r --ai

# Upscale anime artwork with specialized model
python3 upscale.py -i ~/Anime -o ~/Anime_4K --ai --ai-model realesr-animevideo
```

## License

This project is open source and available for personal and commercial use.

## Contributing

Feel free to submit issues and enhancement requests!
