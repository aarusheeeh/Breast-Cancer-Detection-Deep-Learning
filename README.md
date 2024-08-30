# Breast-Cancer-Detection-Deep-Learning

This project utilizes deep learning to classify breast cancer images into benign or malignant categories.

**Data:**

The data is organized in the `./data` directory and consists of two subfolders:

* `benign`: Contains images of benign breast tissue.
* `malignant`: Contains images of malignant breast tissue.

**Model:**

The model used is a Convolutional Neural Network (CNN) with the following architecture:

* **Convolutional Layers:** These layers extract features from the images.
* **Max Pooling Layers:** These layers reduce the dimensionality of the data.
* **Dropout Layers:** These layers prevent overfitting by randomly dropping neurons during training.
* **Dense Layers:** These layers make the final classification decision.

The model definition is implemented in the `larger_model` function within the script. 

**Training:**

The script performs the following steps during training:

1. **Loads images:** Images are loaded from the data folders and resized to a common size.
2. **Preprocessing:** Images are normalized by dividing pixel values by 255.
3. **One-Hot Encoding:** Labels (benign or malignant) are converted to one-hot encoded vectors.
4. **Train-Test Split:** The data is split into training and testing sets for model evaluation.
5. **Model Training:** The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training progress is monitored with validation accuracy and loss.

**Evaluation:**

The script evaluates the model's performance on the test set and reports the final accuracy.

**Testing:**

The script demonstrates how to test the model on:

* **An image from the test set:** This provides a controlled evaluation of the model's performance on unseen data.
* **A new image:** This allows you to test the model on any image you provide. 

**Visualization:**

The script visualizes the training and validation loss and accuracy curves to assess the model's learning behavior.

**Saving and Loading:**

The trained model is saved as `28april.h5`. You can reload the model using `model = load_model('28april.h5')`.

**Dependencies:**

The script requires the following Python libraries:

* numpy
* pydot (optional for visualization)
* graphviz (optional for visualization)
* matplotlib.pyplot
* h5py
* keras
* opencv-python (cv2)
* sklearn

**Further Exploration:**

* Experiment with different hyperparameters (e.g., number of epochs, learning rate) to improve model performance.
* Try different CNN architectures to see if they achieve better results.
* Explore techniques like data augmentation to increase the size and diversity of the training data.

**Disclaimer:**

This project is for educational purposes only and should not be used for medical diagnosis. This is a simplified implementation and may not achieve the same accuracy as real-world medical imaging systems.
