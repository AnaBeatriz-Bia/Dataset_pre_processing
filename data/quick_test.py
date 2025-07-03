# quick_test.py
"""
Script para executar um teste rápido do treinamento com poucas imagens
"""

import torch
from dataset_sampler import create_quick_test_dataset, cleanup_test_dataset
from config import *
from dataset import create_dataloaders
from model import create_model
from trainer import train_model
from utils import get_device, print_model_info


def run_quick_test(
    samples: int = 100,
    epochs: int = 5,
    batch_size: int = 4,
    cleanup_after: bool = True
):
    """
    Executa um teste rápido do pipeline de treinamento
    
    Args:
        samples: Número total de amostras para teste
        epochs: Número de épocas para treinar
        batch_size: Tamanho do batch
        cleanup_after: Se deve remover dataset de teste após o treino
    """
    
    print("🚀 INICIANDO TESTE RÁPIDO DO PIPELINE")
    print("=" * 50)
    
    test_dataset_dir = None
    
    try:
        # 1. Criar dataset de teste
        print("\n1️⃣ Criando dataset de teste...")
        test_dataset_dir = create_quick_test_dataset(samples=samples)
        
        # 2. Configurar dispositivo
        print("\n2️⃣ Configurando dispositivo...")
        device = get_device()
        print(f"✅ Usando: {device}")
        
        # 3. Criar DataLoaders
        print("\n3️⃣ Criando DataLoaders...")
        train_loader, val_loader = create_dataloaders(
            dataset_dir=test_dataset_dir,
            batch_size=batch_size,
            img_size=IMG_SIZE,
            num_workers=min(NUM_WORKERS, 2)  # Menos workers para teste
        )
        
        print(f"✅ Train: {len(train_loader.dataset)} imagens")
        print(f"✅ Val: {len(val_loader.dataset)} imagens")
        
        # 4. Criar modelo
        print("\n4️⃣ Criando modelo...")
        model = create_model(num_classes=NUM_CLASSES, device=device)
        print_model_info(model)
        
        # 5. Testar forward pass
        print("\n5️⃣ Testando forward pass...")
        sample_batch = next(iter(train_loader))
        images, targets = sample_batch
        
        with torch.no_grad():
            outputs = model(images.to(device))
            print(f"✅ Input shape: {images.shape}")
            print(f"✅ Output shape: {outputs.shape}")
        
        # 6. Executar treinamento de teste
        print("\n6️⃣ Executando treinamento de teste...")
        print(f"⏱️ Épocas: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print("-" * 30)
        
        test_model_path = 'test_model.pth'
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            learning_rate=LEARNING_RATE,
            device=device,
            save_path=test_model_path
        )
        
        # 7. Testar inferência
        print("\n7️⃣ Testando inferência...")
        trained_model.eval()
        with torch.no_grad():
            test_output = trained_model(images[:1].to(device))
            print(f"✅ Inferência OK: {test_output.shape}")
        
        print("\n" + "=" * 50)
        print("🎉 TESTE RÁPIDO CONCLUÍDO COM SUCESSO!")
        print("=" * 50)
        print("✅ Pipeline funcionando corretamente")
        print("✅ Pronto para treinamento completo")
        print(f"✅ Modelo de teste salvo: {test_model_path}")
        
        return trained_model, test_dataset_dir
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TESTE: {e}")
        print("🔍 Verifique os logs acima para detalhes")
        raise e
        
    finally:
        # Limpeza opcional
        if cleanup_after and test_dataset_dir:
            print(f"\n🧹 Limpando dataset de teste...")
            cleanup_test_dataset(test_dataset_dir)


def run_ultra_quick_test():
    """
    Teste ultra rápido com configurações mínimas
    """
    print("⚡ TESTE ULTRA RÁPIDO (50 samples, 2 epochs)")
    return run_quick_test(
        samples=50,
        epochs=2,
        batch_size=2,
        cleanup_after=True
    )


def run_medium_test():
    """
    Teste médio para validação mais robusta
    """
    print("🔄 TESTE MÉDIO (200 samples, 10 epochs)")
    return run_quick_test(
        samples=200,
        epochs=10,
        batch_size=8,
        cleanup_after=False  # Manter dataset para análise
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Teste rápido do pipeline YOLO")
    parser.add_argument("--samples", type=int, default=100, help="Número de amostras")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--no-cleanup", action="store_true", help="Não remover dataset de teste")
    parser.add_argument("--ultra", action="store_true", help="Teste ultra rápido")
    parser.add_argument("--medium", action="store_true", help="Teste médio")
    
    args = parser.parse_args()
    
    if args.ultra:
        run_ultra_quick_test()
    elif args.medium:
        run_medium_test()
    else:
        run_quick_test(
            samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cleanup_after=not args.no_cleanup
        )