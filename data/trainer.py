# trainer.py
"""
Funções para treinamento e validação do modelo
"""

import torch
import torch.optim as optim
from model import YOLOLoss


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Treina por uma época
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validação
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def setup_optimizer_and_scheduler(model, learning_rate=0.001, step_size=30, gamma=0.1):
    """
    Configura optimizer e scheduler
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                device='cpu', save_path='best_yolo_model.pth'):
    """
    Função principal de treinamento
    """
    # Setup
    criterion = YOLOLoss()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, learning_rate)
    
    best_val_loss = float('inf')
    
    print(f"Device: {device}")
    print(f"Dataset train: {len(train_loader.dataset)} imagens")
    print(f"Dataset val: {len(val_loader.dataset)} imagens")
    
    # Loop de treinamento
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Treinar
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Modelo salvo!")
    
    return model