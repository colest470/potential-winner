import os
import glob
import time
import numpy as np
from PIL import Image
import kagglehub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from juliacall import Main as jl

jl.include("./src/edge.jl")

class EdgeDetectionDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx]

        jl_img = jl.detectEdges(path)

        img_np = np.array(jl_img)

        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        img_np = (img_np * 255).astype(np.uint8)

        img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)

        return img, label

class Classifier(nn.Module):
    def __init__(self, input_pixels):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_pixels, 1000),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1000, 900),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(900, 800),
            nn.ReLU(),

            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.network(x)

def main():
    base_path = "./kaggle/input/handwritten-digits-0-9"

    extensions = ("*.jpg", "*.jpeg", "*.png")

    file_paths = []

    for label in range(10):
        path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
        folder = os.path.join(base_path, str(label))

        for ext in extensions:
            for filename in glob.glob(os.path.join(folder, ext)):
                file_paths.append((filename, label))

    if len(file_paths) == 0:
        print("Found no dataset downloaded!")

        base_path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")

        print("Path to dataset files:", base_path)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = EdgeDetectionDataset(file_paths, transform)

    print("Dataset size:", len(dataset))

    sample_img, sample_label = dataset[0]

    print("Image shape:", sample_img.shape)
    print("Label:", sample_label)

    input_pixels = sample_img.numel()

    print("Input pixels:", input_pixels)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(input_pixels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    print("Training started...")

    start = time.time()

    for epoch in range(10):

        model.train()

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/10 "
                    f"Batch {batch_idx} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(dataloader)

        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    end = time.time()

    print(f"\nTraining completed in {end-start:.2f} seconds")

if __name__ == "__main__":
    main()