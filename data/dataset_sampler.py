import os
import shutil
import random
from pathlib import Path
from typing import Optional


def create_test_dataset(
    source_dataset_dir: str,
    output_dataset_dir: str,
    train_samples: int = 80,
    val_samples: int = 20,
    seed: Optional[int] = 42
) -> None:
    """
    Cria um dataset pequeno para testes a partir do dataset original
    
    Args:
        source_dataset_dir: Diretório do dataset original
        output_dataset_dir: Diretório onde será criado o dataset de teste
        train_samples: Número de amostras para treino (padrão: 80)
        val_samples: Número de amostras para validação (padrão: 20)
        seed: Seed para reprodutibilidade (padrão: 42)
    """
    
    if seed is not None:
        random.seed(seed)
    
    source_path = Path(source_dataset_dir)
    output_path = Path(output_dataset_dir)
    
    # Verificar se o dataset original existe
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset original não encontrado: {source_dataset_dir}")
    
    # Criar estrutura de diretórios do dataset de teste
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            (output_path / split / folder).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Criando dataset de teste em: {output_dataset_dir}")
    print(f"📊 Amostras de treino: {train_samples}")
    print(f"📊 Amostras de validação: {val_samples}")
    print("-" * 50)
    
    # Processar cada split (train/val)
    for split in ['train', 'val']:
        samples_needed = train_samples if split == 'train' else val_samples
        
        source_images_dir = source_path / split / 'images'
        source_labels_dir = source_path / split / 'labels'
        
        if not source_images_dir.exists():
            print(f"⚠️ Aviso: {source_images_dir} não encontrado, pulando...")
            continue
        
        # Listar todas as imagens disponíveis
        available_images = list(source_images_dir.glob('*.jpg'))
        
        if len(available_images) == 0:
            print(f"⚠️ Aviso: Nenhuma imagem encontrada em {source_images_dir}")
            continue
        
        # Selecionar amostras aleatórias
        if len(available_images) < samples_needed:
            print(f"⚠️ Apenas {len(available_images)} imagens disponíveis em {split}, usando todas")
            selected_images = available_images
        else:
            selected_images = random.sample(available_images, samples_needed)
        
        # Copiar imagens e labels selecionados
        copied_count = 0
        for img_path in selected_images:
            # Copiar imagem
            dest_img_path = output_path / split / 'images' / img_path.name
            shutil.copy2(img_path, dest_img_path)
            
            # Copiar label correspondente (se existir)
            label_path = source_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label_path = output_path / split / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dest_label_path)
            
            copied_count += 1
        
        print(f"✅ {split.capitalize()}: {copied_count} amostras copiadas")
    
    print("-" * 50)
    print(f"✅ Dataset de teste criado com sucesso!")
    print(f"📍 Localização: {output_dataset_dir}")
    
    # Mostrar estatísticas finais
    show_dataset_stats(output_dataset_dir)


def show_dataset_stats(dataset_dir: str) -> None:
    """
    Mostra estatísticas do dataset
    """
    dataset_path = Path(dataset_dir)
    
    print("\n📊 ESTATÍSTICAS DO DATASET:")
    print("-" * 30)
    
    total_images = 0
    total_labels = 0
    
    for split in ['train', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            img_count = len(list(images_dir.glob('*.jpg')))
            label_count = len(list(labels_dir.glob('*.txt')))
            
            print(f"{split.capitalize():>5}: {img_count:>3} imagens, {label_count:>3} labels")
            
            total_images += img_count
            total_labels += label_count
    
    print("-" * 30)
    print(f"{'Total':>5}: {total_images:>3} imagens, {total_labels:>3} labels")


def quick_test_setup(
    source_dataset_dir: str = "dataset_final",
    test_dataset_dir: str = "dataset_test",
    samples: int = 100
) -> str:
    """
    Setup rápido para criar um dataset de teste pequeno
    
    Args:
        source_dataset_dir: Dataset original
        test_dataset_dir: Nome do dataset de teste
        samples: Total de amostras (será dividido 80/20 entre train/val)
    
    Returns:
        Caminho do dataset de teste criado
    """
    
    train_samples = int(samples * 0.8)  # 80% para treino
    val_samples = samples - train_samples  # 20% para validação
    
    create_test_dataset(
        source_dataset_dir=source_dataset_dir,
        output_dataset_dir=test_dataset_dir,
        train_samples=train_samples,
        val_samples=val_samples
    )
    
    return test_dataset_dir


def cleanup_test_dataset(test_dataset_dir: str) -> None:
    """
    Remove o dataset de teste
    """
    test_path = Path(test_dataset_dir)
    if test_path.exists():
        shutil.rmtree(test_path)
        print(f"🗑️ Dataset de teste removido: {test_dataset_dir}")
    else:
        print(f"⚠️ Dataset de teste não encontrado: {test_dataset_dir}")


# Função para usar diretamente no notebook
def create_quick_test_dataset(samples: int = 100) -> str:
    """
    Função simplificada para criar rapidamente um dataset de teste
    
    Args:
        samples: Número total de amostras desejadas
    
    Returns:
        Nome do diretório do dataset de teste criado
    """
    print("🚀 CRIANDO DATASET DE TESTE RÁPIDO")
    print("=" * 40)
    
    test_dir = quick_test_setup(samples=samples)
    
    print("\n💡 COMO USAR:")
    print("1. Importe: from config import *")
    print(f"2. Mude: DATASET_DIR = '{test_dir}'")
    print("3. Execute seu treinamento normalmente!")
    print("\n⚡ Treinamento será muito mais rápido para testes!")
    
    return test_dir


if __name__ == "__main__":
    # Exemplo de uso direto
    test_dataset = create_quick_test_dataset(samples=100)
    print(f"\nDataset de teste criado: {test_dataset}")