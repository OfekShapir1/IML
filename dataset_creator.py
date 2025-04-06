import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ColorizationDataset(Dataset):
    def __init__(self, grayscale_folder, color_folder, transform=None):
        """
        Args:
            grayscale_folder (string): Directory with grayscale images.
            color_folder (string): Directory with corresponding color images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.grayscale_folder = grayscale_folder
        self.color_folder = color_folder
        self.grayscale_images = os.listdir(grayscale_folder)
        self.color_images = os.listdir(color_folder)
        self.transform = transform

    def __len__(self):
        # The length of the dataset is the number of grayscale images
        return len(self.grayscale_images)

    def __getitem__(self, idx):
        # Get the grayscale and color images by index
        grayscale_image_path = os.path.join(self.grayscale_folder, self.grayscale_images[idx])
        color_image_path = os.path.join(self.color_folder, self.color_images[idx])

        # Open images
        grayscale_image = Image.open(grayscale_image_path).convert('RGB')  # Grayscale converted to RGB for processing
        color_image = Image.open(color_image_path).convert('RGB')

        # Apply transformations (optional)
        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            color_image = self.transform(color_image)

        return grayscale_image, color_image


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])


from torch.utils.data import DataLoader

# Set paths for your grayscale and color images
grayscale_folder = "path_to_grayscale_images"
color_folder = "path_to_color_images"

# Create dataset instance
dataset = ColorizationDataset(grayscale_folder, color_folder, transform=transform)

# Create DataLoader instance
batch_size = 16  # Define batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# Load pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Remove the fully connected layers to use ResNet-18 as an encoder
encoder = nn.Sequential(*list(resnet18.children())[:-2])  # Remove FC layers and avgpool

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Upsample
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Upsample
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Upsample
        self.relu3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)      # Output channels: RGB

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        return x


class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        self.encoder = encoder  # ResNet-18 as feature extractor
        self.decoder = Decoder()  # Decoder for colorization

    def forward(self, x):
        features = self.encoder(x)  # Extract features using ResNet-18
        colorized_image = self.decoder(features)  # Decode features to colorized image
        return colorized_image

criterion = nn.MSELoss()

# Set up the model, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ColorizationModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop (simplified)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data in dataloader:  # Assuming you have a DataLoader with grayscale/color pairs
        grayscale_images, color_images = data
        grayscale_images = grayscale_images.to(device)
        color_images = color_images.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(grayscale_images)

        # Compute loss
        loss = criterion(outputs, color_images)
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

# Testing loop (simplified)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for grayscale_image, color_image in dataloader:
        grayscale_image = grayscale_image.to(device)
        
        # Generate the colorized image
        generated_colorized_image = model(grayscale_image)

        # Convert the tensor back to a PIL image for visualization
        generated_colorized_image = generated_colorized_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        generated_colorized_image = (generated_colorized_image + 1) / 2  # Rescale to [0, 1] range for display

        plt.imshow(generated_colorized_image)
        plt.show()
        



