import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data paths
TRAIN_PATH = os.path.join(DATA_DIR, "train")
VALIDATION_TEST_PATH = os.path.join(DATA_DIR, "validation_test")
CK_TEST_CSV = os.path.join(DATA_DIR, "test", "ckextended.csv")

# Model save path
MODEL_FILE_NAME = os.path.join(MODELS_DIR, "emotion_model.keras")

# Plot save path
PLOT_FILE_NAME = os.path.join(REPORTS_DIR, "performance_metrics.png")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 5
INPUT_SHAPE = (48, 48, 1)
