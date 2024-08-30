# Breast Cancer Detection using Deep Learning

This project implements a deep learning model to classify breast cancer images as either benign or malignant. The model is built using Keras and TensorFlow, and a graphical user interface (GUI) is provided using Tkinter to make it easy to load images, train the model, and test the model on new images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Download the Dataset:**
   - The dataset used for this project is available on Kaggle. You can download it [here](#dataset).
   - After downloading, place the dataset images in the `./data/` folder.
   - The dataset should be organized into two subfolders: `benign/` and `malignant/`, with each subfolder containing the respective images.

2. **Run the Script:**
   - To start the graphical interface, run the following command:
     ```bash
     python breast_cancer_detection.py
     ```

3. **Training:**
   - Click on the "Start Training" button to train the model with the provided dataset. The model will be saved as `28april.h5`.

4. **Testing:**
   - You can test the model on images from the dataset or on any random image using the provided GUI buttons.

5. **Visualization:**
   - The training and validation losses and accuracies can be visualized by clicking on the "See Loss and Accuracy plots" button.

## Dataset

The dataset used for this project is the **BreakHis 400x Breast Cancer Dataset**. It contains images categorized into two classes: benign and malignant. You can download the dataset from the following link:

- [BreakHis 400x Breast Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/scipygaurav/breakhis-400x-breast-cancer-dataset?resource=download)

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:
- Conv2D layer with 32 filters and a kernel size of 3x3, followed by ReLU activation and max-pooling.
- Another Conv2D layer with 32 filters and a kernel size of 3x3, followed by ReLU activation and max-pooling.
- Conv2D layer with 64 filters and a kernel size of 3x3, followed by ReLU activation and max-pooling.
- Dropout layers to prevent overfitting.
- Dense layers with ReLU activation.
- Output layer with a softmax activation function.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

## Training

- The model is trained for a specified number of epochs (default is 500) with a batch size of 128.
- The dataset is split into training (80%) and testing (20%) sets.
- The input images are normalized from 0-255 to 0-1.
- The labels are one-hot encoded.

## Testing

- The model can be tested on images from the test set or on any new image using the GUI.
- The predicted class and accuracy are displayed for the tested image.

## Visualization

- The losses and accuracies during training can be visualized using matplotlib.
- The GUI provides a convenient button to display these plots.

## Dependencies

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Tkinter
- Matplotlib
- Numpy
- Scikit-learn

## Acknowledgments

This project was developed to explore the application of deep learning in medical imaging, specifically for breast cancer detection. The dataset and preprocessing steps are crucial for the model's performance.

---

Feel free to modify the README file to fit any additional details specific to your project!
