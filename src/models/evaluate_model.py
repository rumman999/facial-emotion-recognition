import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_FILE_NAME, CK_TEST_CSV, REPORTS_DIR

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# 1. Load your model
model = load_model(MODEL_FILE_NAME)

# Your model's labels (Alphabetical order from your training script)
my_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# 2. Define the CK+ CSV label mapping based on the dataset metadata
ck_label_map = {
    0: 'angry',
    1: 'disgust',  # Will be ignored
    2: 'fear',     # Will be ignored
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral',
    7: 'contempt'  # Will be ignored
}

def evaluate_ck_csv(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Lists to track everything for the Confusion Matrix
    y_true = []
    y_pred = []
    
    for index, row in df.iterrows():
        # Get the true label
        ck_integer = int(row['emotion'])
        true_emotion = ck_label_map.get(ck_integer, "unknown")
        
        # Skip emotions your model was not trained to predict
        if true_emotion not in my_labels:
            continue
            
        # Extract the image from the CSV text
        pixel_string = row['pixels']
        pixel_array = np.fromstring(pixel_string, dtype=int, sep=' ')
        img_reshaped = pixel_array.reshape(48, 48)
        
        # Normalize and reshape for the model
        img_normalized = np.array(img_reshaped, 'float32') / 255.0
        input_img = img_normalized.reshape(1, 48, 48, 1)
        
        # Predict
        prediction = model.predict(input_img, verbose=0)
        predicted_emotion = my_labels[np.argmax(prediction)]
        
        # Append to our tracking lists
        y_true.append(true_emotion)
        y_pred.append(predicted_emotion)
            
    if len(y_true) == 0:
        print("Error: No matching emotions found. Check your CSV column names.")
        return
        
    # --- TERMINAL OUTPUT ---
    print("\n" + "="*40)
    print("      CK+ EVALUATION RESULTS      ")
    print("="*40)
    
    # scikit-learn generates a beautiful terminal report with Precision, Recall, and F1-Score
    report = classification_report(y_true, y_pred, target_names=my_labels)
    print(report)
    print("="*40)

    # --- CONFUSION MATRIX VISUALIZATION ---
    # Generate the raw matrix numbers
    cm = confusion_matrix(y_true, y_pred, labels=my_labels)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    
    # Create a Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=my_labels, 
                yticklabels=my_labels,
                cbar=False)
    
    # Add titles and labels
    plt.title('Emotion Recognition Confusion Matrix (CK+ Dataset)', pad=20, fontsize=14)
    plt.ylabel('True Emotion', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    
    # Adjust layout and save the graphic
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Confusion Matrix saved successfully to: {save_path}")
    
    # Optionally display it on the screen right now
    plt.show()

# Usage
evaluate_ck_csv(CK_TEST_CSV)