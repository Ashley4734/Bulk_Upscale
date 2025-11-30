# Bulk Image Upscaler

A fast, efficient command-line tool for upscaling multiple images at once on macOS (and other platforms).

## Features

- **Bulk Processing**: Upscale hundreds of images in one command
- **Multiple Algorithms**: Choose from Lanczos, Bicubic, Bilinear, or Nearest Neighbor resampling
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

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

   Or install Pillow directly:
   ```bash
   pip3 install Pillow
   ```

3. **Make the script executable (optional):**
   ```bash
   chmod +x upscale.py
   ```

## Usage

### Basic Usage

```bash
python3 upscale.py -i <input_directory> -o <output_directory>
```

### Examples

**Upscale all images in a folder by 2x (default):**
```bash
python3 upscale.py -i ./photos -o ./upscaled_photos
```

**Upscale by 4x with bicubic interpolation:**
```bash
python3 upscale.py -i ./photos -o ./upscaled_photos -s 4 -m bicubic
```

**Process subdirectories recursively:**
```bash
python3 upscale.py -i ./photos -o ./upscaled_photos -r
```

**Use 8 parallel workers for faster processing:**
```bash
python3 upscale.py -i ./photos -o ./upscaled_photos -w 8
```

**Upscale with custom JPEG quality:**
```bash
python3 upscale.py -i ./photos -o ./upscaled_photos -q 98
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input directory containing images | Required |
| `-o, --output` | Output directory for upscaled images | Required |
| `-s, --scale` | Scale factor (e.g., 2.0 for 2x, 4.0 for 4x) | 2.0 |
| `-m, --method` | Resampling method: `lanczos`, `bicubic`, `bilinear`, `nearest` | lanczos |
| `-q, --quality` | JPEG quality (1-100) | 95 |
| `-r, --recursive` | Process subdirectories recursively | False |
| `-w, --workers` | Number of parallel workers | 4 |

### Resampling Methods

- **Lanczos** (recommended): Highest quality, best for photos and detailed images
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

1. **For best quality**: Use Lanczos resampling with high JPEG quality (95-100)
2. **For speed**: Use Bicubic with 4-8 workers depending on your CPU
3. **For large batches**: Enable recursive mode and let it run in the background
4. **Preserve originals**: Always use a different output directory to keep your original images safe

## Performance

Processing speed depends on:
- Image size and quantity
- Scale factor (4x takes longer than 2x)
- Resampling method (Lanczos is slower but higher quality)
- Number of workers (more workers = faster, up to your CPU core count)

Example: On a modern Mac, you can typically process:
- ~50-100 photos (1920x1080) per minute at 2x scale
- ~20-40 photos per minute at 4x scale

## Troubleshooting

**"No images found"**: Make sure your input directory contains supported image formats

**"Permission denied"**: Run with appropriate permissions or choose a different output directory

**Memory errors**: Reduce the number of workers or process smaller batches

**RGBA/RGB warnings**: The tool automatically converts RGBA images to RGB when saving as JPEG

## Examples of Real-World Use

```bash
# Upscale vacation photos for printing
python3 upscale.py -i ~/Pictures/Vacation2024 -o ~/Pictures/Vacation2024_4K -s 2 -q 98

# Batch upscale screenshots
python3 upscale.py -i ~/Screenshots -o ~/Screenshots_HD -s 2 -m bicubic

# Process entire photo library recursively
python3 upscale.py -i ~/Photos -o ~/Photos_Upscaled -r -w 8 -s 2
```

## License

This project is open source and available for personal and commercial use.

## Contributing

Feel free to submit issues and enhancement requests!
