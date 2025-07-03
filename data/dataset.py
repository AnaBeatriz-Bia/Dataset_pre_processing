import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


class HumanBoatDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Listar todas as imagens
        self.image_files = list(self.images_dir.glob('*.jpg'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Carregar anotações
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        print(f"Label path {label_path}")
        boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        # Parse: class x_center y_center width height
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            boxes.append([class_id, x_center, y_center, width, height])
        
        # Aplicar transformações na imagem
        if self.transform:
            image = self.transform(image)
        
        # Converter boxes para tensor
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, boxes


def collate_fn(batch):
    """
    Função para tratar batches com número diferente de objetos por imagem
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets


def get_transforms(img_size=640):

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(dataset_dir, batch_size=8, img_size=640, num_workers=4):
    """
    Cria os DataLoaders para treino e validação
    """
    transform = get_transforms(img_size)
    
    # Datasets
    train_dataset = HumanBoatDataset(
        images_dir=f"{dataset_dir}/train/images",
        labels_dir=f"{dataset_dir}/train/labels",
        img_size=img_size,
        transform=transform
    )
    
    val_dataset = HumanBoatDataset(
        images_dir=f"{dataset_dir}/val/images",
        labels_dir=f"{dataset_dir}/val/labels", 
        img_size=img_size,
        transform=transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader