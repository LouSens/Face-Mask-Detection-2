# plot_history.py
import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_training_history():
    try:
        # Load saved history
        with open("training_history.pkl", "rb") as f:
            history = pickle.load(f)

        # Create figure with square aspect ratio
        plt.figure(figsize=(8, 8))  # Square figure

        # Plot Loss (left axis)
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        plt.plot(history['loss'], 'b-', label='Train Loss')
        plt.plot(history['val_loss'], 'b--', label='Val Loss')
        plt.ylabel('Loss')
        plt.title('Training and Validation Metrics')
        plt.legend()
        plt.grid(True)

        # Set y-axis limits for Loss to show variation
        loss_min = min(min(history['loss']), min(history['val_loss']))
        loss_max = max(max(history['loss']), max(history['val_loss']))
        plt.ylim(loss_min * 0.9, loss_max * 1.1)  # 10% padding

        # Plot Accuracy (right axis)
        plt.subplot(2, 1, 2)  # Second plot
        plt.plot(history['accuracy'], 'r-', label='Train Accuracy')
        plt.plot(history['val_accuracy'], 'r--', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Set y-axis limits for Accuracy to show variation
        plt.ylim(0, 1.05)  # Always show full accuracy range

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('training_results_square.png', dpi=300)
        plt.show()

    except Exception as e:
        print(f"Error: {e}\nMake sure training_history.pkl exists!")


if __name__ == "__main__":
    plot_training_history()
