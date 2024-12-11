# Project - CNN for Image Classification

## 1. Installations
To run the project, you need the following libraries and frameworks installed:
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install them using:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## 2. Project Motivation
This project demonstrates the use of Convolutional Neural Networks (CNNs) for binary image classification: identifying whether an image contains a cat or a dog. The dataset includes:
- Training images (8000 total; 4000 cats and 4000 dogs)
- Test images (2000 total; 1000 cats and 1000 dogs)
- Single test image (4 total; 2 cats and 2 dogs)

The aim is to train a model to achieve high accuracy in distinguishing between these two categories.

## 3. File Descriptions
- `dataset/`: Contains the images for training and testing.
  - `training_set/`: Includes subfolders for cat and dog images used for training.
  - `test_set/`: Includes subfolders for cat and dog images used for testing.
  - `single_prediction/`: Contains a few individual images for testing the model manually.
- `CNN for Image Classification.ipynb`: The notebook for training the CNN model.
- `CNN Model Evaluation.ipynb`: A notebook to evaluate the trained model on the test set.
- `cnn_model.keras`: The saved trained model.
- `README.md`: This file, explaining the project structure.

## 4. Results
The main findings of the code can be found at the post on Medium available post linked below.
https://medium.com/@jwei1_24619/deep-learning-image-classification-cat-vs-dog-prediction-using-cnn-9e7910a0b450.

## 5. Licensing, Authors, Acknowledgements, etc.
This project is a learning exercise in machine learning/deep learning, convolutional neural networks and image classification.
- **Dataset Source**: Images were sourced from my teacher Farhad Abbasi Amiri(https://github.com/farhadabbasiamiri).
- **Author**: Jieyi Wei
- **Acknowledgements**: Thanks to TensorFlow documentation and online tutorials for guidance.

