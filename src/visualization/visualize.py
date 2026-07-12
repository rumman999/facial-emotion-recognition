import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_path):
    """
    Plots the training and validation accuracy and loss, then saves the figure.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    ax1.plot(epochs_range, acc, 'r-', label="Train")
    ax1.plot(epochs_range, val_acc, 'b-', label="Val")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Loss
    ax2.plot(epochs_range, loss, 'r-', label="Train")
    ax2.plot(epochs_range, val_loss, 'b-', label="Val")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Performance metrics plot saved to {save_path}")
    
    # plt.show()
