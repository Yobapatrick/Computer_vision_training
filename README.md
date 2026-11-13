# Transfer Learning with ResNet50 on CIFAR-10

**Notebook:** `Transfer_Learning_CIFAR_10.ipynb`

## Overview

This notebook demonstrates **transfer learning** for image classification using **ResNet50**, a model pretrained on ImageNet, and TensorFlow/Keras.

CIFAR-10 contains 32 × 32 color images divided into ten classes. The notebook preprocesses these images for ResNet50, builds a classification model, trains it, and evaluates and visualizes its results.

## Objectives

- Load and explore the CIFAR-10 dataset.
- Preprocess images for ResNet50.
- Reuse ImageNet-pretrained weights.
- Add a classification head for CIFAR-10’s ten classes.
- Train and evaluate the model.
- Review learning curves and sample predictions.

## CIFAR-10 Classes

| Index | Class |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

## Model and Approach

The notebook follows these steps:

1. **Load the data:** Retrieve the CIFAR-10 training and test datasets.
2. **Preprocess the images:** Convert pixel values to `float32` and apply `resnet50.preprocess_input`.
3. **Resize the images:** Upsample images from 32 × 32 to 224 × 224 pixels, the input size used for ResNet50 in this notebook.
4. **Extract features:** Use ResNet50 without its ImageNet classification layer (`include_top=False`) and with its pretrained ImageNet weights (`weights='imagenet'`).
5. **Classify images:** Add pooling and dense layers, ending with a softmax output for ten classes.
6. **Train and evaluate:** Train with SGD and `sparse_categorical_crossentropy`, then calculate loss and accuracy on the evaluation data.

> **Note:** The provided code does not explicitly freeze ResNet50’s layers. As a result, the base model’s weights may be updated during training.

## Requirements

- Python 3
- TensorFlow with Keras
- NumPy
- Matplotlib
- TensorFlow Datasets (imported in the notebook)
- Pillow (PIL)

You can run the notebook in Google Colab or a Jupyter environment. A GPU is recommended to speed up training, but it is not required to load the dataset.

## How to Run

1. Open `C3_W1_Lab_2_Transfer_Learning_CIFAR_10.ipynb` in Google Colab or Jupyter.
2. In Colab, select **Runtime → Change runtime type → GPU** if you want to use a GPU for training.
3. Run the notebook cells in order, from top to bottom.
4. Allow CIFAR-10 and the ImageNet weights to download if prompted.
5. Review the metrics, plots, and predictions generated at the end of the notebook.

The notebook defines a batch size of 32 and trains for four epochs. The training cell uses a batch size of 64.

## Outputs

The notebook produces:

- Visualizations of CIFAR-10 images.
- A summary of the model architecture.
- Training and evaluation metrics.
- Loss and accuracy curves.
- Predictions for images from the evaluation dataset.

Actual performance depends on the environment, library versions, and training run, so no specific accuracy result is stated here.

## Notes and Potential Improvements

- The `display_images` function receives labels but does not explicitly indicate whether each prediction is correct.
- The validation-set visualization currently uses the title `"Training Data"`; changing it to `"Validation Data"` would make the plot clearer.
- `feature_extractor` constructs ResNet50 when called. For a more efficient and maintainable implementation, create the base model once and reuse it.
- Upsampling CIFAR-10 images does not add new image detail; it only adapts their dimensions to ResNet50’s expected input size.



## Author

**Patrick Yoba**
