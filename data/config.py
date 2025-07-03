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

# Configurações para Teste Rápido
TEST_MODE = False  # Mude para True para usar dataset pequeno
TEST_DATASET_DIR = "dataset_test"
TEST_SAMPLES = 100  # Total de amostras para teste (80 train + 20 val)
TEST_EPOCHS = 5     # Poucas épocas para teste rápido

def get_dataset_config():
    """Retorna configurações do dataset baseado no modo"""
    if TEST_MODE:
        return {
            'dataset_dir': TEST_DATASET_DIR,
            'num_epochs': TEST_EPOCHS,
            'batch_size': min(BATCH_SIZE, 4)  # Batch menor para teste
        }
    else:
        return {
            'dataset_dir': DATASET_DIR,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE
        }