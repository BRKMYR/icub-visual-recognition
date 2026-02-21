# Caffe Python Scripts for iCubWorld28 Object Recognition

A collection of Python scripts for training, evaluating, and deploying convolutional neural network models using the [Caffe](https://caffe.berkeleyvision.org/) deep learning framework (BVLC/Berkeley). The primary focus is object recognition on the **iCubWorld28** dataset -- a robotic vision benchmark containing 28 object classes across 7 categories (plate, laundry detergent, sprayer, cup, soap, dishwashing detergent, sponge), captured over multiple days from the iCub humanoid robot's cameras.

## Historical Context

This project was developed in early 2016 (February -- July) as part of a student/research project at a time when Caffe was the dominant deep learning framework. It predates the widespread adoption of PyTorch (released October 2016) and TensorFlow's rise to maturity. The scripts use the BVLC Reference CaffeNet architecture (an AlexNet variant) and integrate with **YARP** (Yet Another Robot Platform), the middleware used by the iCub humanoid robot, as well as **GURLS** (Grand Unified Regularized Least Squares), a MATLAB-based machine learning library from IIT.

The code is written in Python 2 style (bare `print` statements, `map()` returning lists) with some `from __future__ import division` guards.

## What This Project Does

The pipeline covers the full workflow of a robotic vision experiment:

1. **Dataset preparation** -- Walk the iCubWorld28 directory tree and generate the `train.txt` / `test.txt` label files that Caffe requires, along with synset files and metadata (object attributes, affordances).
2. **Feature extraction** -- Run images through a pretrained CaffeNet to extract FC7 (fully connected layer 7) activations as 4096-dimensional feature vectors, then export them to MATLAB `.mat` files for use with the GURLS classifier.
3. **Classification and evaluation** -- Classify test images using a fine-tuned CaffeNet model and compute per-class and overall accuracy.
4. **Live classification via YARP** -- Receive cropped images from the iCub robot's vision pipeline in real time, classify them with Caffe, and forward the extracted features to MATLAB over YARP ports.
5. **Visualization** -- Visualize network internals: convolutional filters, layer activations, fully connected layer histograms, and softmax output.

## Scripts

### Data Preparation

| File | Description |
|------|-------------|
| `prepareData.py` | Generates Caffe-format `train.txt` and `test.txt` for the full 28-class iCubWorld28 dataset. Walks the dataset directory, maps each object subdirectory (e.g., `plate4`, `sponge2`) to a class number (0--27), and writes image paths with labels. Also generates `synsets.txt` and ground-truth class number files. |
| `prepareData_7.py` | Same as `prepareData.py` but collapses the 28 fine-grained classes into 7 coarse categories (one per object type), mapping all plate variants to class 0, all laundry detergent variants to class 1, etc. |
| `prepareData_All.py` | The most comprehensive data preparation script. Supports multiple labeling schemes selectable via `TRAIN_MODE`: `28_objects`, `7_objects`, `attribute_shape` (long/round/rectangular), `attribute_material` (ceramic/plastic/furry/clear/wet), and `affordance` (hold/drink/eat/clean/open/cut). Exports attributes and affordances as both text files and CSV. |
| `prepareDATASET.py` | A variant of the data preparation script adapted for a custom 10-object dataset (Banana, Apple, Orange, Smartphone, Sponge, Plate, Cup, Laptop, Laundry Detergent). Supports training modes for object class, shape attribute, material attribute, and affordance. |

### Feature Extraction (Caffe to GURLS)

| File | Description |
|------|-------------|
| `caffe_FC7_to_GURLS_train.py` | Loads the training split of iCubWorld28 (25,831 images), runs each image through CaffeNet, extracts the 4096-dimensional FC7 layer activations, and saves the resulting feature matrix and class labels as `.mat` files for the GURLS classifier in MATLAB. Supports switching between ImageNet-pretrained and iCubWorld28-fine-tuned models. |
| `caffe_FC7_to_GURLS_test.py` | Same pipeline as the training script but for the test split (24,884 images). Saves FC7 features and ground-truth labels to the GURLS test data directory. |

### Classification and Evaluation

| File | Description |
|------|-------------|
| `classifyTestData.py` | Classifies test images using a CaffeNet model fine-tuned on iCubWorld28 (28 classes). For each image, records the predicted class number to a text file, then computes overall accuracy and per-class accuracy by comparing predictions against ground-truth labels. |
| `classifyTestData_7.py` | Same as `classifyTestData.py` but uses a model trained on the 7-category variant, with accuracy vectors sized for 7 classes. Uses a different model checkpoint (`caffenet_train_iter_12120.caffemodel`). |
| `compareTRUEandTESTClassNumbers.py` | A standalone evaluation utility. Reads ground-truth and predicted class numbers from text files and computes overall accuracy plus per-class accuracy across all 28 classes. Useful for evaluating predictions generated by other scripts or external classifiers (e.g., GURLS). |

### Live Classification

| File | Description |
|------|-------------|
| `yarp2caffe_classify.py` | Real-time classification script for the iCub robot. Opens a YARP input port to receive cropped 256x256 RGB images from the robot's vision pipeline, classifies each frame using CaffeNet, prints the top-5 predictions, extracts the FC7 features, and sends them as a YARP image to a MATLAB port for online learning with GURLS. Supports ImageNet-pretrained, 28-class, and 7-class iCubWorld28 models. |

### Visualization

| File | Description |
|------|-------------|
| `visualize_classification.py` | A visualization and analysis script (adapted from Caffe's `00-classification.ipynb` example). Loads a trained model, classifies a single image, and visualizes: conv1 through conv5 filter weights and activations, pool5 output, FC6 and FC7 activation distributions (line plots and histograms), and the final softmax probability vector. Includes a `vis_square()` utility for tiling filter/activation grids. |

## Dependencies

These scripts were developed against the following stack (circa 2016):

- **Python 2.7** (uses `print` as a statement, `map()` returns lists)
- **Caffe** (BVLC) with PyCaffe bindings
- **NumPy**
- **Matplotlib** (for visualization)
- **SciPy** (`scipy.io` for `.mat` file export)
- **YARP** Python bindings (for iCub robot integration)
- **GURLS** (MATLAB library, not a Python dependency but the target for exported features)

### Required Model Files

The scripts expect CaffeNet model files and mean files in the Caffe installation directory:

- `bvlc_reference_caffenet.caffemodel` (ImageNet-pretrained)
- `caffenet_train_iter_16160.caffemodel` (fine-tuned on iCubWorld28, 28 classes)
- `caffenet_train_iter_12120.caffemodel` (fine-tuned on iCubWorld28, 7 categories)
- `ilsvrc_2012_mean.npy` / `iCubWorld28_mean.npy` (dataset mean files)
- Corresponding `deploy.prototxt` network definitions

## Dataset

The **iCubWorld28** dataset consists of images of 28 household objects (4 instances each of 7 categories) captured by the iCub robot across multiple days (`day1` through `day4`). The dataset is organized as:

```
iCubWorld28/
  train/
    day1/ day2/ day3/ day4/
      plate/ cup/ soap/ sponge/ sprayer/ laundry-detergent/ dishwashing-detergent/
        plate1/ plate2/ plate3/ plate4/
          00000001.ppm ...
  test/
    (same structure)
```

Training set: ~25,831 images. Test set: ~24,884 images.

## Note

This repository is preserved as a historical artifact. The hardcoded file paths throughout the scripts (e.g., `/home/niklas/Downloads/caffe/`) reflect the original development environment and would need to be updated to run on another machine. The code itself is functional but reflects the conventions and tooling of its era.
