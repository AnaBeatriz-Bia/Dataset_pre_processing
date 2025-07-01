# model.py
"""
Arquitetura do modelo YOLO simplificado
"""

import torch
import torch.nn as nn


class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        
        # Backbone simples (substitua por YOLOv5 backbone se necessário)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 20))  # Para 640x640 input
        )
        
        # Head YOLO (5 = x,y,w,h,conf + num_classes)
        self.head = nn.Conv2d(512, 3 * (5 + num_classes), 1)  # 3 anchors por grid
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Loss simplificada - implemente a loss completa do YOLO se necessário
        # Por agora, retorna um valor constante para demonstração
        return torch.tensor(1.0, requires_grad=True)


def create_model(num_classes=2, device='cpu'):
    """
    Cria e retorna o modelo YOLO
    """
    model = SimpleYOLO(num_classes=num_classes)
    model.to(device)
    return model


def load_model(model_path, num_classes=2, device='cpu'):
    """
    Carrega um modelo pré-treinado
    """
    model = create_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model