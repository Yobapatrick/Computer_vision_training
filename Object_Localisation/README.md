# Image Classification and Object Localization

A TensorFlow project that trains a **Convolutional Neural Network (CNN)** to perform two tasks simultaneously:

- **Classify** the digit shown in an image, from 0 to 9.
- **Locate** the digit by predicting its bounding box.

The implementation is available in [`C3_W1_Lab_3_Object_Localization.ipynb`](./C3_W1_Lab_3_Object_Localization.ipynb).

## Overview

This project uses the **MNIST** handwritten digit dataset to generate a custom localization dataset.

Each original **28 × 28** image is placed at a random position on a black **75 × 75** canvas. The corresponding bounding box coordinates are calculated automatically.

Each training example contains:

| Component | Description |
|---|---|
| Input image | A grayscale image with shape `(75, 75, 1)` |
| Classification target | A one-hot encoded digit label |
| Localization target | Normalized coordinates `[xmin, ymin, xmax, ymax]` |

> The target bounding box surrounds the original 28 × 28 MNIST image patch, not the exact outline of the handwritten digit.

## Model Architecture

The model uses a shared feature extractor followed by two output branches.

```text
Input Image: 75 × 75 × 1
            |
Conv2D: 16 filters, ReLU
            |
AveragePooling2D
            |
Conv2D: 32 filters, ReLU
            |
AveragePooling2D
            |
Conv2D: 64 filters, ReLU
            |
AveragePooling2D
            |
Flatten
            |
Dense: 128 units, ReLU
            |
            +-- Classification: 10 units, Softmax
            |
            +-- Bounding Box Regression: 4 units, Linear
```

### Classification Output

The classification branch predicts a probability for each digit from 0 to 9.

The predicted class is selected using:

```python
predicted_labels = np.argmax(predictions[0], axis=1)
```

### Localization Output

The regression branch predicts four bounding box coordinates:

```text
[xmin, ymin, xmax, ymax]
```

- `xmin`, `ymin`: top-left corner.
- `xmax`, `ymax`: bottom-right corner.

Target coordinates are normalized to the range `[0, 1]`.

## Technologies

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow
- Jupyter Notebook or Google Colab

This project does not use Apache Spark.

## Installation

### Local Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow tensorflow-datasets numpy matplotlib pillow importlib-resources jupyter
```

> Use a Python version supported by your TensorFlow installation. Dependencies are not pinned in the provided notebook, so adjustments may be required depending on installed versions.

## Usage

### Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open `C3_W1_Lab_3_Object_Localization.ipynb` and execute the cells in order.

### Google Colab

1. Upload the notebook to Google Colab.
2. Optionally select a GPU runtime.
3. Ensure the required dependencies are installed.
4. Run the cells in order.

An internet connection may be required to download MNIST.

## Hardware Support

The notebook selects a TensorFlow distribution strategy based on the available hardware:

| Hardware | Strategy |
|---|---|
| CPU only | Default strategy |
| Single GPU | Default strategy |
| Multiple GPUs | `MirroredStrategy` |
| TPU | `TPUStrategy` |

A GPU is **not required**. CPU training is supported, although it will generally take longer.

> The TPU initialization code is environment-dependent and may require adaptation.

## Data Processing

The input pipeline uses `tf.data.Dataset` to:

1. Load MNIST images and labels.
2. Place each digit randomly on a 75 × 75 canvas.
3. Normalize pixel values to `[0, 1]`.
4. One-hot encode digit labels.
5. Generate normalized bounding box coordinates.
6. Shuffle training examples.
7. Create batches.
8. Prefetch upcoming training batches.

Random placement allows the model to see the same digit at different locations during training.

## Training Configuration

The notebook uses:

```python
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
EPOCHS = 10
steps_per_epoch = 60000 // BATCH_SIZE
```

The model is trained with the **Adam** optimizer.

| Output | Loss Function | Metric |
|---|---|---|
| Classification | Categorical cross-entropy | Accuracy |
| Bounding box | Mean squared error | Mean squared error |

Both losses contribute to the total training loss with equal weights by default.

## Evaluation

### Classification Accuracy

Measures the proportion of correctly classified digits.

### Bounding Box MSE

Measures the squared differences between predicted and target coordinates.

### Intersection over Union

**Intersection over Union (IoU)** measures the overlap between the predicted and target bounding boxes:

```text
IoU = Intersection Area / Union Area
```

For valid boxes:

- Values close to `0` indicate poor overlap.
- Values close to `1` indicate strong overlap.
- A value of `1` indicates identical boxes.

The notebook counts predictions with an IoU **greater than or equal to 0.6**.

> Localization and classification are evaluated separately. A correctly located digit can still be misclassified.

## Prediction Visualization

The notebook displays randomly selected examples with:

- **Green boxes:** ground-truth bounding boxes.
- **Red boxes:** predicted bounding boxes.
- The predicted digit below each image.
- The corresponding IoU score.

Incorrect digit predictions are highlighted in red.

## Limitations and Known Issues

- The model assumes **one digit per image** and does not support a variable number of objects.
- The MNIST test split is used for validation, leaving no independent final test set.
- Validation digit positions are regenerated during new dataset iterations.
- Validation uses a single batch of 10,000 images, which may require substantial memory.
- The linear regression output does not guarantee coordinates within `[0, 1]` or valid corner ordering.
- Bounding box drawing docstrings describe a different coordinate order from the one used by the notebook.
- Training loss plots may overlap because separate figures are not created.
- The unpacking of `model.evaluate()` results may require adaptation across Keras versions. Using `return_dict=True` is more robust.

## Results

The supplied notebook contains no saved execution outputs. Model performance must be measured after training.

For reproducible reporting, record:

- Python and TensorFlow versions.
- Hardware used.
- Number of epochs and batch size.
- Classification accuracy.
- Bounding box MSE.
- Mean IoU.
- Percentage of predictions with IoU greater than or equal to 0.6.



## References

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Datasets: MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- [Keras Functional API Guide](https://www.tensorflow.org/guide/keras/functional)


## Author

**Patrick Yoba**
