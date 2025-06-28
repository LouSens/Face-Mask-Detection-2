# Real-Time Face Mask Detection

A continuation of a group project originally developed during my first-year undergraduate (third semester). This version significantly improves the model’s accuracy and generalization across various real-world conditions, making it robust against challenges like lighting variations, crowd density, and camera sensitivity.

## ✨ Key Improvements

* **High accuracy and validation accuracy** with minimal overfitting.
* **Real-time detection** using webcam input.
* Enhanced model generalization to handle various environments and lighting conditions.

---

## 📁 Project Structure

```
.
├── model.py             # Training pipeline: data loading, model building, training, saving
├── mask_detection.py    # Real-time detection with webcam
├── plotting.py          # Visualizes training history
├── training_history.pkl # Saved training metrics
├── mask_detector_final.keras # Final trained model
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

### 2. Prepare Dataset

Ensure your dataset is organized as follows:

```
dataset/
├── with_mask/
│   ├── image1.jpg
│   └── ...
└── without_mask/
    ├── image2.jpg
    └── ...
```

### 3. Train the Model

Run the training script:

```bash
python model.py
```

This will:

* Train and fine-tune a MobileNetV2 model
* Save the model as `mask_detector_final.keras`
* Save training history in `training_history.pkl`

### 4. Visualize Training Metrics

Plot the training/validation accuracy and loss:

```bash
python plotting.py
```

The result will be saved as `training_results_square.png`.

### 5. Run Real-Time Detection

Use your webcam to detect face masks in real time:

```bash
python mask_detection.py
```

Press **`q`** to exit the camera feed.

---

## 🧠 Model Architecture

* **Base**: MobileNetV2 (pre-trained on ImageNet)
* **Head**: Custom dense layers with dropout & batch normalization
* **Loss Function**: Binary Crossentropy
* **Optimization**: Adam with learning rate decay and early stopping

---

## 📊 Performance

The model achieves:

* High training & validation accuracy (comparable across both)
* Stable loss curves without overfitting
* Real-world robustness to lighting, angles, and distance

---

## 👤 Author

Developed and maintained by **\[Your Name]** — a first-year undergraduate student. This project is a personal improvement on a previously group-developed version, with major enhancements in performance and reliability.

---

## 📄 License

This project is for academic use and demonstration purposes. For commercial use, please contact the author.
