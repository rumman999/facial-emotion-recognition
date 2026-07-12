# Facial Emotion Recognition

A deep learning project to detect and classify human emotions from `48x48` facial images using a Convolutional Neural Network (CNN).

## Getting Started

If you are cloning this repository from GitHub, follow these steps to get the project running on your local machine.

### 1. Set Up the Environment
First, create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Mac/Linux

pip install -r requirements.txt
```

### 2. Download and Extract the Dataset
This project trains on a subset of the FER-2013 dataset (using 5 categories: `angry`, `happy`, `neutral`, `sad`, `surprise`).

1. Download the dataset (for example, from [Kaggle's FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)).
2. Ensure you have the folder structure correctly organized with `train` and `test` folders.
3. Place these folders inside the `data/` directory at the root of the project.

Your folder structure should look exactly like this before training:
```text
facial-emotion-recognition/
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   └── ...
│   └── test/
│       ├── angry/
│       ├── happy/
│       └── ...
```

### 3. Train the Model
Once your data is in place, you can train the CNN model from scratch:
```bash
python src/models/train_model.py
```
*(This will automatically save the best model to `models/emotion_model.keras` and output a performance plot to `reports/performance_metrics.png`)*

### 4. Test on a Single Image
To run inference on a new image (e.g., to test how the model performs on a sample photo), run the prediction script. Make sure you place your sample photos inside `data/samples/`.
```bash
python src/predict_emotion.py data/samples/my_face.jpg
```
*(If you run the script without providing an image, it will attempt to default to a sample image).*
