# Concrete Autoencoder Experiments

This project compares:

* **Unsupervised CAE** → feature selection via reconstruction loss  
* **Supervised CAE** → feature selection via classification loss  

Based on:

> Abid, A., Balin, M. F., & Zou, J. (2019).  
> *Concrete Autoencoders for Differentiable Feature Selection and Reconstruction*.  
> https://arxiv.org/abs/1901.09346

---

## Overview

Both models use a **Concrete selector layer** to choose **k input features** in a differentiable way.

* **Unsupervised CAE**:  
  `X → select k → reconstruct X` (MSE loss)

* **Supervised CAE**:  
  `X → select k → predict y` (cross-entropy loss)

---

## Models

* Reconstruction: linear layer (regression)  
* Classification: softmax layer (logistic regression)

---

## Datasets

* MNIST  (70,000 samples, 784 features, 10 classes)
* Fashion-MNIST  (70,000 samples, 784 features, 10 classes)
* ISOLET  (7,797 samples, 617 features, 26 classes)
* COIL-20 (1,440 samples, 400 features, 20 classes)
* Mice Protein (1,080 samples, 77 features, 8 classes)
* Human Activity Recognition (5,744 samples, 561 features, 6 classes)
All datasets are flattened, normalized, and split into train/val/test.

---

## Metrics

Each selected feature set is evaluated using separate models:

* **Reconstruction**: MSE,
* **Classification**: Accuracy, Balanced Accuracy, Macro-F1  

---

## Experiments

### 1. Baseline Experiment

For each dataset:

1. Train unsupervised CAE  
2. Train supervised CAE  
3. Extract selected features  
4. Evaluate both subsets on classification and reconstruction task

Run with multiple seeds:

```python
seeds = [0, 1, 2, 3, 4]
````

Results are reported as **mean ± std** per:

```
dataset × k × method
```

---

### 2. Generalization Experiment

This experiment evaluates how well selected features transfer to a **different downstream task**.

Setup:

* Combine MNIST and Fashion-MNIST
* Shared input space (784 features)
* Construct two tasks:

  * Domain classification (MNIST vs Fashion)
  * Full classification (20 classes)

Feature sets are learned using different objectives and then evaluated on the **20-class task**.

---

## Run

### Baseline experiment

```bash
python -m src.experiment --experiment baseline
```

### Generalization experiment

```bash
python -m src.experiment --experiment generalization
```

### Fast debug run

```bash
python -m src.experiment --experiment baseline --fast
```

```bash
python -m src.experiment --experiment generalization --fast
```

---

## Outputs

Baseline:

```
results/
├── raw_results.json
├── raw_results.csv
└── summary_mean_std.csv
```

Generalization:

```
results_generalization/
├── raw_generalization_results.json
├── raw_generalization_results.csv
└── summary_generalization_mean_std.csv
```

---

## Note

`concrete_autoencoder.py` is based on the **[original implementation by Abid et al.](https://github.com/mfbalin/Concrete-Autoencoders/blob/master/concrete_autoencoder/concrete_autoencoder/__init__.py)**,
but adapted to support both **unsupervised** and **supervised** variants.

```
```
