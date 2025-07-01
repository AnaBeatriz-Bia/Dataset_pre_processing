# main.py
"""
Script principal para executar o treinamento do modelo YOLO
"""

import torch
from config import *
from dataset import create_dataloaders
from model import create_model
from trainer import train_model
from utils import get_device, check_dataset_structure, print_model_info


def main():
    """
    Função principal para executar o treinamento
    """
    print("=" * 60)
    print("INICIANDO TREINAMENTO DO MODELO YOLO")
    print("=" * 60)
    
    # Verificar estrutura do dataset
    print("\n1. Verificando estrutura do dataset...")
    check_dataset_structure(DATASET_DIR)
    
    # Configurar dispositivo
    print("\n2. Configurando dispositivo...")
    device = get_device()
    print(f"✓ Usando dispositivo: {device}")
    
    # Criar DataLoaders
    print("\n3. Carregando datasets...")
    train_loader, val_loader = create_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"✓ DataLoaders criados com sucesso!")
    
    # Criar modelo
    print("\n4. Criando modelo...")
    model = create_model(num_classes=NUM_CLASSES, device=device)
    print_model_info(model)
    
    # Iniciar treinamento
    print("\n5. Iniciando treinamento...")
    print("=" * 60)
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=MODEL_SAVE_PATH
    )
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print(f"Modelo salvo em: {MODEL_SAVE_PATH}")
    print("=" * 60)
    
    return trained_model


if __name__ == "__main__":
    model = main()