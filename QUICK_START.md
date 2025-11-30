# Quick Start Guide

Get started with Bulk Image Upscaler in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Step 2: Prepare Your Images

Put all images you want to upscale in a folder, for example:
```
/Users/yourname/Pictures/my_images/
```

## Step 3: Run the Upscaler

```bash
python3 upscale.py -i /Users/yourname/Pictures/my_images -o /Users/yourname/Pictures/upscaled
```

That's it! Your upscaled images will be in the output folder.

## Quick Examples

### 2x upscale (default)
```bash
python3 upscale.py -i ./input -o ./output
```

### 4x upscale
```bash
python3 upscale.py -i ./input -o ./output -s 4
```

### Process all subfolders
```bash
python3 upscale.py -i ./input -o ./output -r
```

### Faster processing (8 cores)
```bash
python3 upscale.py -i ./input -o ./output -w 8
```

## Using the Shell Wrapper

You can also use the shell script wrapper:

```bash
./upscale.sh -i ./input -o ./output -s 4
```

## Need Help?

```bash
python3 upscale.py --help
```

See README.md for full documentation.
