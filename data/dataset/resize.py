import os
from PIL import Image
from PIL import Image, ImageOps

# Source and target root paths
src_root = './generator_defect_classification'
dst_root = './generator_defect_classification_resized'
target_size = (64, 64)

# Traverse subdirectories
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.lower().endswith('.png'):
            # Source and destination paths
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_path, src_root)
            dst_path = os.path.join(dst_root, relative_path)

            # Create target directory if needed
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Open, resize, and save image
            try:
                img = Image.open(src_path).convert('RGB')
                # img = img.resize(target_size, Image.ANTIALIAS)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(dst_path)
                print(f"Saved: {dst_path}")
            except Exception as e:
                print(f"Failed on {src_path}: {e}")