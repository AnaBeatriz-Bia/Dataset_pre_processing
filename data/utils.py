import torch
import os
from pathlib import Path


def get_device():
    """
    Retorna o dispositivo disponível (GPU ou CPU)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_dataset_structure(dataset_dir):
    """
    Verifica se a estrutura do dataset está correta
    """
    dataset_path = Path(dataset_dir)
    
    required_dirs = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels'
    ]
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {full_path}")
            
    print("✓ Estrutura do dataset verificada com sucesso!")
    
    # Contar arquivos
    for split in ['train', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        img_count = len(list(images_dir.glob('*.jpg')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"✓ {split.capitalize()}: {img_count} imagens, {label_count} labels")


def print_model_info(model):
    """
    Imprime informações sobre o modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total de parâmetros: {total_params:,}")
    print(f"✓ Parâmetros treináveis: {trainable_params:,}")
    print(f"✓ Tamanho do modelo: {total_params * 4 / 1024 / 1024:.2f} MB")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Salva um checkpoint completo do treinamento
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint salvo em: {path}")


def load_checkpoint(model, optimizer, path):
    """
    Carrega um checkpoint do treinamento
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint carregado: Epoch {epoch}, Loss: {loss:.4f}")
    return epoch, loss