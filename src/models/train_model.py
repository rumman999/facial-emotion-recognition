import os
import sys

# Ensure src module can be imported when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keras.callbacks import EarlyStopping

from config import (
    TRAIN_PATH, TEST_PATH, MODEL_FILE_NAME, PLOT_FILE_NAME,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, INPUT_SHAPE
)
from data.make_dataset import load_data
from features.build_features import preprocess_features
from models.build_model import build_emotion_model
from visualization.visualize import plot_training_history

def main():
    print("=== 1. Data Loading ===")
    X_train_raw, y_train_raw = load_data(TRAIN_PATH)
    X_test_raw, y_test_raw = load_data(TEST_PATH)
    
    print("\n=== 2. Feature Preprocessing ===")
    X_train, y_train, X_test, y_test = preprocess_features(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, num_classes=NUM_CLASSES
    )
    
    print("\n=== 3. Model Building ===")
    model = build_emotion_model(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )
    print(model.summary())
    
    print("\n=== 4. Model Training ===")
    stopEarly = EarlyStopping(monitor='val_accuracy', patience=5)
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[stopEarly]
    )
    
    print("\n=== 5. Visualization ===")
    plot_training_history(history, PLOT_FILE_NAME)
    
    print("\n=== 6. Saving Model ===")
    os.makedirs(os.path.dirname(MODEL_FILE_NAME), exist_ok=True)
    model.save(MODEL_FILE_NAME)
    print(f"Model successfully saved to {MODEL_FILE_NAME}")

if __name__ == "__main__":
    main()
