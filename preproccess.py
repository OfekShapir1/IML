import zipfile
from PIL import Image
import os
import numpy as np
from skimage.color import rgb2lab
from tqdm import tqdm
import shutil

# CONFIGURATION
BATCH_SIZE = 5000
batch_counter = 1

# Paths
zip_path = "/content/images.zip"  # Original ZIP you copied from Drive
working_zip_path = "/content/batch_work/images_working.zip"
temp_dir = "/content/batch_work/temp_batch"
output_drive_path = "/content/drive/MyDrive/ColorizationBatches"

# Setup safe workspace
os.makedirs("/content/batch_work", exist_ok=True)
os.makedirs(output_drive_path, exist_ok=True)

# Make a copy of the zip to work on
if not os.path.exists(working_zip_path):
    shutil.copy(zip_path, working_zip_path)

while True:
    print(f"\n🔁 Starting Batch {batch_counter}")
    color_dir = os.path.join(temp_dir, "color")
    bw_dir = os.path.join(temp_dir, "bw")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(bw_dir, exist_ok=True)

    # Open working zip
    with zipfile.ZipFile(working_zip_path, 'r') as archive:
        all_images = [f for f in archive.namelist() if f.endswith(('.jpg', '.png'))]
        if not all_images:
            print("✅ All images processed and ZIP is empty!")
            break

        current_batch = all_images[:BATCH_SIZE]

        for name in tqdm(current_batch, desc=f"Processing Batch {batch_counter}"):
            try:
                with archive.open(name) as file:
                    img = Image.open(file).convert("RGB")
            except:
                continue

            base_name = os.path.basename(name)
            img.save(os.path.join(color_dir, base_name))

            img_np = np.array(img) / 255.0
            lab = rgb2lab(img_np).astype("float32")
            L = lab[:, :, 0] / 100.0
            L_img = (L * 255).astype("uint8")
            bw_img = Image.fromarray(L_img)
            bw_img.save(os.path.join(bw_dir, base_name))

    # Zip the batch
    zip_output_path = shutil.make_archive(f"/content/batch_work/colorization_batch_{batch_counter}", 'zip', temp_dir)
    shutil.move(zip_output_path, os.path.join(output_drive_path, f"batch_{batch_counter}.zip"))
    shutil.rmtree(temp_dir)

    # Remove processed images from the ZIP
    remaining_files = [f for f in all_images if f not in current_batch]
    with zipfile.ZipFile(working_zip_path, 'r') as zin:
        with zipfile.ZipFile("/content/batch_work/temp_remaining.zip", 'w') as zout:
            for item in zin.infolist():
                if item.filename in remaining_files:
                    zout.writestr(item, zin.read(item.filename))

    os.remove(working_zip_path)
    os.rename("/content/batch_work/temp_remaining.zip", working_zip_path)

    print(f"✅ Batch {batch_counter} complete and saved to Drive.")
    batch_counter += 1

print("🎉 All done! All images processed and removed from original ZIP.")
