import zipfile
from PIL import Image
import os
import numpy as np
from skimage.color import rgb2lab
from tqdm import tqdm
import shutil

# CONFIG
BATCH_SIZE = 5000
RESIZE_TO = (512, 512)
log_file_path = "/content/drive/MyDrive/ColorizationBatches/processed_log.txt"

# Determine batch number dynamically from log
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as f:
        total_processed = len(f.readlines())
else:
    total_processed = 0

BATCH_NUMBER = total_processed // BATCH_SIZE + 1

# PATHS
working_zip = "/content/images.zip"
temp_dir = "/content/batch_work/temp_batch"
output_drive = "/content/drive/MyDrive/ColorizationBatches"
os.makedirs("/content/batch_work", exist_ok=True)
os.makedirs(output_drive, exist_ok=True)

# LOAD PROCESSED LOG
processed = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as f:
        processed = set(x.strip() for x in f.readlines())

# READ FROM ZIP
with zipfile.ZipFile(working_zip, 'r') as zip_file:
    all_images = [f for f in zip_file.namelist() if f.endswith(('.jpg', '.png'))]
    remaining = [f for f in all_images if os.path.basename(f) not in processed]
    current_batch = remaining[:BATCH_SIZE]

    if not current_batch:
        print("✅ No unprocessed images left.")
    else:
        # Create temp folders
        color_dir = os.path.join(temp_dir, "color")
        bw_dir = os.path.join(temp_dir, "bw")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(bw_dir, exist_ok=True)

        processed_now = []
        for name in tqdm(current_batch, desc=f"Batch {BATCH_NUMBER}"):
            try:
                with zip_file.open(name) as file:
                    img = Image.open(file).convert("RGB").resize(RESIZE_TO)
            except:
                continue

            base_name = os.path.basename(name)
            processed_now.append(base_name)

            # Save resized color image
            img.save(os.path.join(color_dir, base_name))

            # Convert to Lab and extract L channel for grayscale
            img_np = np.array(img) / 255.0
            lab = rgb2lab(img_np).astype("float32")
            L = lab[:, :, 0] / 100.0
            L_img = (L * 255).astype("uint8")
            bw_img = Image.fromarray(L_img)

            # Save grayscale image
            bw_img.save(os.path.join(bw_dir, base_name))

# SAVE FOLDERS TO DRIVE
shutil.move(color_dir, os.path.join(output_drive, f"batch_{BATCH_NUMBER}_color"))
shutil.move(bw_dir, os.path.join(output_drive, f"batch_{BATCH_NUMBER}_bw"))
shutil.rmtree(temp_dir)

# REMOVE PROCESSED FILES FROM ZIP
with zipfile.ZipFile(working_zip, 'r') as zin:
    with zipfile.ZipFile("/content/images_remaining.zip", 'w') as zout:
        for item in zin.infolist():
            if os.path.basename(item.filename) not in processed_now:
                zout.writestr(item, zin.read(item.filename))
os.remove(working_zip)
os.rename("/content/images_remaining.zip", working_zip)

# UPDATE LOG
with open(log_file_path, 'a') as log:
    for name in processed_now:
        log.write(name + '\n')

print(f"✅ Batch {BATCH_NUMBER} complete. Resized to {RESIZE_TO}, saved to Drive, removed from ZIP, and logged.")
