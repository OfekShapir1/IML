import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
2. Apply Transformations
PyTorch’s transforms module allows you to apply a series of transformations to images, such as resizing, normalization, or tensor conversion. Typically, you'd want to resize the images to a consistent size and convert them into tensors.

Here's an example of how to define a transform that resizes images and converts them to PyTorch tensors:

python
Copy
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])
This transform:

Resizes the images to 256x256 pixels.

Converts the images to tensors, which PyTorch can work with.

Normalizes the pixel values to the range [-1, 1].

3. Instantiate the Dataset and DataLoader
Now that we have the ColorizationDataset class and the transform, you can instantiate the dataset and wrap it in a DataLoader to efficiently load data in batches.

python
Copy
from torch.utils.data import DataLoader

# Set paths for your grayscale and color images
grayscale_folder = "path_to_grayscale_images"
color_folder = "path_to_color_images"

# Create dataset instance
dataset = ColorizationDataset(grayscale_folder, color_folder, transform=transform)

# Create DataLoader instance
batch_size = 16  # Define batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
