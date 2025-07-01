# config.py
"""
Configurações centralizadas do projeto YOLO
"""

# Configurações do Dataset
DATASET_DIR = "dataset_final"
NUM_CLASSES = 2  # 0: banhistas, 1: barcos

# Configurações de Treinamento
LEARNING_RATE = 0.001
BATCH_SIZE = 8
IMG_SIZE = 640
NUM_EPOCHS = 50

# Configurações do DataLoader
NUM_WORKERS = 4

# Configurações do Scheduler
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

# Configurações de Checkpoint
MODEL_SAVE_PATH = 'best_yolo_model.pth'