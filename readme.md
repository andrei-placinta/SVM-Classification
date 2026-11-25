# Vehicle Data Classification Using Support Vector Machines

A MATLAB-based machine learning project focused on vehicle data classification using **Support Vector Machines (SVM)**.

The project implements and evaluates three SVM classifiers using Replay Attack, Fuzzing Attack and combined datasets, comparing their classification performance through standard evaluation metrics.

## Features

- Import and preprocessing of vehicle classification datasets using MATLAB.
- Processing of Replay Attack, Fuzzing Attack and Combined datasets.
- Feature and classification label extraction.
- Training and testing dataset partitioning (80/20 split).
- Training of three independent Support Vector Machine classifiers.
- RBF (Radial Basis Function) kernel implementation.
- Feature standardization during model training.
- Classification prediction and decision score generation.
- ROC curve generation for model performance comparison.
- Confusion matrix visualization for each classifier.
- Calculation of Accuracy, Precision, Recall (Sensitivity) and F1-score.
- Comparative evaluation of classification performance across datasets.

## Tech Stack

- **Programming Language:** MATLAB
- **Machine Learning:** Support Vector Machines (SVM)
- **Classification Algorithm:** RBF Kernel
- **Data Processing:** MATLAB Tables and Arrays
- **Data Partitioning:** Holdout Validation (80/20)
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score
- **Performance Visualization:** ROC Curves, Confusion Matrices
- **MATLAB Toolbox:** Statistics and Machine Learning Toolbox

## Dataset Processing

The project uses three CSV datasets containing vehicle classification data:

- **Replay Dataset:** Data associated with Replay Attack scenarios.
- **Fuzzing Dataset:** Data associated with Fuzzing Attack scenarios.
- **Combined Dataset:** A combination of Replay and Fuzzing data.

The script loads each dataset using MATLAB's `readtable()` function.

For each dataset:

1. The first 100,000 records are selected.
2. Input features are extracted from all columns except the last one.
3. The last column is used as the classification label.
4. The dataset is converted into MATLAB arrays for model training.

## Model Training

Three independent Support Vector Machine classifiers are trained, one for each dataset.

The models use:

- **Algorithm:** Support Vector Machine Classification.
- **Kernel Function:** Radial Basis Function (RBF).
- **Feature Standardization:** Enabled.
- **Classification Labels:** 0 and 1.
- **Training Set:** 80% of the dataset.
- **Testing Set:** 20% of the dataset.

The classification models are implemented using MATLAB's `fitcsvm()` function.

Predictions and classification scores are generated using `predict()`.

## Model Evaluation

The project evaluates the three trained classifiers using multiple performance metrics.

### ROC Curves

ROC curves are generated to compare the classification performance of the three models.

The visualization includes:

- Replay Attack classifier.
- Fuzzing Attack classifier.
- Combined dataset classifier.

The curves display the relationship between the False Positive Rate and True Positive Rate.

### Confusion Matrices

A separate confusion matrix is generated for each classifier to visualize classification results.

The matrices help identify correctly and incorrectly classified observations.

### Performance Metrics

The following evaluation metrics are calculated:

- **Accuracy:** Overall proportion of correctly classified observations.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall (Sensitivity):** Proportion of actual positive observations correctly identified.
- **F1-score:** Harmonic mean of Precision and Recall.

The results are displayed in the MATLAB Command Window for comparison between the three classifiers.

## Project Structure

```text
SVM-Classification/
│
├── README.md
├── script.m
│
└── SSEA/
    ├── replay_Vehicle C - all IDs - IDs_Datafield_Classification.csv
    ├── fuzzing_Vehicle C - all IDs - IDs_Datafield_Classification.csv
    └── combined_Vehicle C - Replay 0x181, 0x161, 0x1a5, Fuzzing 0x244, 0x284, 0x354 - IDs_Datafield_Classification.csv
```

**Note:** The CSV datasets are required to execute the script but are not included in the provided repository archive.

## Running Locally

### Requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- The three required CSV datasets

### Setup

1. Install MATLAB with the Statistics and Machine Learning Toolbox.

2. Place `script.m` in the project root directory.

3. Create an `SSEA` directory alongside `script.m`.

4. Place the three required CSV datasets inside the `SSEA` directory, using the filenames referenced in the MATLAB script.

5. Ensure that each CSV file contains at least 100,000 records and that the final column contains the classification labels.

6. Open MATLAB and set the current working directory to the project root.

7. Run the script:

   ```matlab
   script
   ```

8. The script trains the three SVM classifiers, displays their evaluation metrics in the Command Window and generates ROC curves and confusion matrices.

## Purpose

This academic project demonstrates practical knowledge of machine learning, binary classification, Support Vector Machines, data preprocessing and model evaluation using MATLAB.

It also explores the application of classification algorithms to vehicle-related datasets containing Replay Attack and Fuzzing Attack scenarios.
