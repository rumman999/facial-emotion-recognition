import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_features(X_train, y_train, X_test, y_test, num_classes=5):
    """
    Normalizes images, reshapes them for the CNN, and converts labels to categorical arrays.
    """
    # Normalize images
    X_train_np = np.array(X_train, 'float32') / 255.0
    X_test_np = np.array(X_test, 'float32') / 255.0
    
    # Convert labels to int32
    y_train_np = np.array(y_train, 'int32')
    y_test_np = np.array(y_test, 'int32')
    
    # Reshape features to (batch_size, 48, 48, 1)
    num_train_images = X_train_np.shape[0]
    X_train_reshaped = X_train_np.reshape(num_train_images, 48, 48, 1)
    
    num_test_images = X_test_np.shape[0]
    X_test_reshaped = X_test_np.reshape(num_test_images, 48, 48, 1)
    
    # One-hot encode the labels
    y_train_cat = to_categorical(y_train_np, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_np, num_classes=num_classes)
    
    return X_train_reshaped, y_train_cat, X_test_reshaped, y_test_cat
