!cp "/content/drive/MyDrive/dataset/images1024x1024.zip" "/content/images.zip"
# Force kill any stuck Drive process
!pkill -f "drive"

# Remove the old mount folder (just local, not your files!)
!rm -rf /content/drive

# Now try to mount again
from google.colab import drive
drive.mount('/content/drive')


# resize
import os
from PIL import Image
from tqdm import tqdm

# Source directories (original 512x512 images)
input_color_dir = "/content/drive/MyDrive/ColorizationBatches/batch_1_color"
input_bw_dir = "/content/drive/MyDrive/ColorizationBatches/batch_1_bw"

# Target directories (resized 64x64 images)
output_color_dir = "/content/drive/MyDrive/ColorizationBatches/batch_1_color_64"
output_bw_dir = "/content/drive/MyDrive/ColorizationBatches/batch_1_bw_64"

# Create output directories if they don't exist
os.makedirs(output_color_dir, exist_ok=True)
os.makedirs(output_bw_dir, exist_ok=True)

# Resize function
def resize_images(input_dir, output_dir, size=(64, 64)):
    filenames = sorted(os.listdir(input_dir))
    for filename in tqdm(filenames, desc=f"Resizing in {os.path.basename(input_dir)}"):
        try:
            img = Image.open(os.path.join(input_dir, filename)).resize(size, Image.BICUBIC)
            img.save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Do the resizing
resize_images(input_color_dir, output_color_dir)
resize_images(input_bw_dir, output_bw_dir)
