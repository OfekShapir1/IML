from tqdm import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt

# choose locations.
COLOR_DIR = "/content/drive/MyDrive/ColorizationBatches/batch_1_color"
BW_DIR = "/content/drive/MyDrive/ColorizationBatches/batch_1_bw"
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_PATH = "/content/drive/MyDrive/models/generator_finetuned.pth"
DISC_PATH = "/content/drive/MyDrive/models/discriminator.pth"

# organize the transfer to lab.
class ColorizationDataset(Dataset):
    def __init__(self, colored_images_dir, bw_images_dir, image_size, num_of_images):
        self.colored_images_dir = colored_images_dir
        self.bw_images_dir = bw_images_dir
        self.filenames = sorted(os.listdir(colored_images_dir))
        self.num_of_images = num_of_images
        self.size = image_size

    def __len__(self):
        return len(self.num_of_images)

    def __getitem__(self, idx):
        """
        this function normalize the LAB and transfer the RGB to LAB
        :param idx:
        :return:
        """
        index = self.num_of_images[idx]
        filename = self.filenames[index]
        color_img = Image.open(os.path.join(self.colored_images_dir, filename)).convert("RGB")
        bw_img = Image.open(os.path.join(self.bw_images_dir, filename)).convert("RGB")
        color_np = np.array(color_img) / 255.0
        lab = rgb2lab(color_np).astype("float32")
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:] / 128.0
        L_tensor = torch.tensor(L).unsqueeze(0) # py torch need it to be 3d so it adds channels 1
        ab_tensor = torch.tensor(ab).permute(2, 0, 1) # py torch want ( channels, height , width) so its rearrange.
        return L_tensor, ab_tensor

# ===== Generator (U-Net) =====
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_c, out_c):
        """
        our conv layer in_c - how many channels we get in_out - what me make it/
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True), # make it non linear
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_c, out_c):
        """
        the decoder, gets the small resolution image with the channels and decode it slowly.
        :return:
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_c, out_c)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        d4 = self.up4(b) + e4
        d3 = self.up3(d4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        return self.tanh(self.final(d1))

# ===== Discriminator =====
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ===== Losses =====
adv_criterion = nn.BCELoss()
recon_criterion = nn.L1Loss()

def combine_ab(L, ab):
    L = L * 100
    ab = ab * 128
    lab = torch.cat([L, ab], dim=1)
    return lab

# ===== Train/Test Split =====
all_filenames = sorted(os.listdir(COLOR_DIR))
train_part = list(range(400))
test_part = list(range(400, 500))

train_dataset = ColorizationDataset(COLOR_DIR, BW_DIR, IMAGE_SIZE, train_part)
test_dataset = ColorizationDataset(COLOR_DIR, BW_DIR, IMAGE_SIZE, test_part)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== Model Setup =====
G = UNetGenerator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=5, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=5, gamma=0.5)

# ===== Training Loop =====
for epoch in range(EPOCHS):
    G.train(); D.train()
    total_G_loss, total_D_loss = 0.0, 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for L, real_ab in loop:
        L, real_ab = L.to(DEVICE), real_ab.to(DEVICE)
        fake_ab = G(L)

        real_lab = combine_ab(L, real_ab)
        fake_lab = combine_ab(L, fake_ab.detach())

        D_real = D(real_lab)
        D_fake = D(fake_lab)

        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        D_loss = adv_criterion(D_real, real_labels) + adv_criterion(D_fake, fake_labels)
        opt_D.zero_grad(); D_loss.backward(); opt_D.step()

        D_fake_for_G = D(fake_lab)
        adv_loss = adv_criterion(D_fake_for_G, real_labels)
        l1_loss = recon_criterion(fake_ab, real_ab)
        G_loss = l1_loss + 0.003 * adv_loss

        opt_G.zero_grad(); G_loss.backward(); opt_G.step()

        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
        loop.set_postfix(G_loss=G_loss.item(), D_loss=D_loss.item())

    scheduler_G.step()
    scheduler_D.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | G Loss: {total_G_loss / len(train_loader):.4f} | D Loss: {total_D_loss / len(train_loader):.4f}")

    # Evaluate on test set
    G.eval()
    with torch.no_grad():
        for L, _ in test_loader:
            L = L.to(DEVICE)
            pred = G(L)
            lab = torch.cat([L*100, pred*128], dim=1)
            lab_np = lab[0].cpu().numpy().transpose(1, 2, 0)
            rgb = lab2rgb(lab_np)
            os.makedirs("/content/drive/MyDrive/generated_samples", exist_ok=True)
            plt.imsave(f"/content/drive/MyDrive/generated_samples/sample_epoch_{epoch+1}.png", rgb)
            break

# ===== Save Final Weights =====
os.makedirs("/content/drive/MyDrive/models", exist_ok=True)
torch.save(G.state_dict(), GEN_PATH)
torch.save(D.state_dict(), DISC_PATH)

# ===== Final Inference Sample =====
G.eval()
with torch.no_grad():
    for L, real_ab in test_loader:
        L = L.to(DEVICE)
        pred_ab = G(L).cpu()
        for i in range(min(3, L.size(0))):
            L_img = L[i][0].cpu().numpy() * 100
            ab_img = pred_ab[i].permute(1, 2, 0).numpy() * 128
            lab = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
            lab[:, :, 0] = L_img
            lab[:, :, 1:] = ab_img
            rgb = lab2rgb(lab)
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
        break