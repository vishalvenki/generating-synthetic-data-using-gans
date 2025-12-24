# Generating Tabular Synthetic Data

#### Table of Contents

- Executive Summary
- Introduction
- History of the Project
- Why GANs?
- Existing Research in Synthetic Data Generation
- Methodology
- MIMIC - III Dataset
- Use Case 1: Length of Stay
- Use Case 2: Mortality Prediction
- Synthetic Data Generation
- Scalability Tests
- Statistical Similarity
- Privacy Risk Module
- Model Compatibility
- Ideas for Further Research and Improvements

# Executive Summary

##### Problem context

Advanced analytics is transforming all industries and is inherently data hungry. In health care, data privacy rules detract data sharing for collaboration. Synthetic data, that retains the original characteristics and model compatibility, can make data sharing easy and enable analytics for health care data.

##### Need for change

Conventionally statistical methods have been used, but with limited success. Current de-identification techniques are not sufficient to mitigate re-identification risks. Emerging technologies in Deep Learning such as GAN are very promising to solve this problem.

##### Key Question

How can you certify that the generated data is as similar and as useful as original data for the intended uses?

##### Proposed Solution

The proposed solution involves generating Synthetic Data using Generative Adversarial Networks or GANs and with the help of conventionally available sources such as TGAN and CTGAN.

The project includes modules which can test the generated synthetic data against the original datasets on the following three areas:

- Statistical Similarity: Create standardized modules to check if the generated datasets are similar to the original dataset.
- Privacy Risk: Create standardized metrics to check if generated synthetic data protects privacy of all data points in the original dataset.
- Model Compatibility: Compare performance of Machine Learning techniques on original and synthetic datasets.

##### Results:

- Statistical Similarity: The generated datasets were similar to each other when compared using PCA and Auto-Encoders.
- Privacy module: The generated Privacy at Risk (PaR) metric and modules help identify columns which are at risk to expose the privacy of data points from original data. The generated datasets using TGAN and CTGAN had sufficiently high privacy scores and protected the privacy of original data points.
- Model Compatibility: The synthetic data has comparable model performance for both classification and regression problems.

# Introduction

Healthcare organizations deal with sensitive data that has Personal Identifiable Information (PII) of millions of people. The healthcare industry is particularly sensitive as Patient Identifiable Information data is strictly regulated by the Health Insurance Portability and Accountability Act (HIPAA) of 1996. Firms need to keep customer data secure while leveraging it to innovate research and drive growth. However, current data sharing practices to ensure de-identification have resulted in long wait times for data access. This has proved to be a hindrance to fast innovation. The need of the hour is to reduce the time for data access and enable innovation while protecting the information of patients. The key question to answer here is:

"How can we safely and efficiently share healthcare data that is useful?"

##### Complication

The key questions involve the inherent trade-off between safety and efficiency. With the inception of big data, efficiency in the data sharing process is of paramount importance. Availability and accessibility of data ensure rapid prototyping and lay down the path for quick innovation. Efficient data sharing also unlocks the full potential of analytics and data sciences through use cases like the diagnosis of cancer, predicting response for drug therapy, and drug discovery. While efficient data sharing is crucial, the safety of patient data cannot be ignored. Existing regulations like HIPAA and recent privacy laws like the California Consumer Privacy Act are focused on maintaining the privacy of sensitive information. As per reports on cost data breaches, the cost per record is approximately $150. But the goodwill and trust lost by companies cannot be quantified. So, the balance between data sharing and privacy is tricky.

# History of the Project

Existing de-identification techniques involve two main techniques: 1) Anonymization Techniques and 2) Differential Privacy. Almost every firm relies on these techniques to deal with sensitive information in PII data.

1. Anonymization techniques: These techniques try to remove the columns which contain sensitive information. Methods include deleting columns, masking elements, quasi-identifiers, k-anonymity, l-diversity, and t-closeness.

2. Differential privacy: This is a perturbation technique which adds noise to columns which introduce randomness to data and thus maintain privacy. It is a mechanism to help maximize the aggregate utility of databases ensuring high levels of privacy for the participants by striking a balance between utility and privacy.

However, these techniques are not cutting edge when it comes to maintaining privacy and data sharing. Research has shown that a high percentage of individuals can be correctly re-identified in any dataset using as few as 15 demographic attributes. This challenges the technical and legal adequacy of the de-identification release-and-forget model.

# Why GANs?

## Introduction

A generative adversarial network (GAN) is a class of machine learning systems invented by Ian Goodfellow in 2014. GAN uses algorithmic architectures that use two neural networks, pitting one against the other (the "adversarial") in order to generate new, synthetic instances of data that can pass for real data.

GANs consist of two neural networks contesting with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set. The two Neural Networks are named Generator and a Discriminator.

## GAN Working Overview

Generator: The generator is a neural network that models a transform function. It takes as input a simple random variable and must return, once trained, a random variable that follows the targeted distribution. The generator starts with generating random noise and changes its outputs as per the Discriminator. If the Discriminator is successfully able to identify that generated input is fake, then its weights are adjusted to reduce the error.

Discriminator: The Discriminator's job is to determine if the data fed by the generator is real or fake. The discriminator is first trained on real data so that it can identify it to acceptable accuracy.

This is continued for multiple iterations until the discriminator can identify the real/fake images purely by chance only.

# Existing Research in Synthetic Data Generation

### TGAN

TGAN focuses on generating tabular data with mixed variable types (multinomial/discrete and continuous). To achieve this, it uses LSTM with attention in order to generate data column by column.

### CTGAN

CTGAN is a GAN-based method to model tabular data distribution and sample rows from the distribution. CTGAN implements mode-specific normalization to overcome the non-Gaussian and multimodal distribution.

### Differentially Private GAN

DPGAN focuses on preserving the privacy during the training procedure instead of adding noise on the final parameters directly. Noise is added to the gradient of the Wasserstein distance with respect to the training data.

### PATE-GAN

PATE GAN consists of two generator blocks called student block and teacher block on top of the existing generator block. It modifies the existing GAN algorithm in a way that guarantees privacy.

# Methodology

In order to validate the efficacy of GANs, we propose a methodology for thorough evaluation of synthetic data generated by GANs.

## MIMIC - III Dataset

MIMIC-III stands for Medical Information Mart for Intensive Care Unit. It is a comprehensive clinical dataset of 40,000+ patients admitted in ICU.

- 26 tables: Comprehensive clinical dataset.
- 45,000+ patients: Includes details like Gender, Ethnicity, Marital Status.
- 65,000+ ICU admissions: Data from 2001 to 2012.

## Use Case 1: Length of Stay

Goal: Predicting the length of stay in ICU using 4 out of 26 tables.

## Use Case 2: Mortality Prediction

Goal: Predict mortality based on the number of interactions between patient and hospital (lab tests, prescriptions, procedures).

# Synthetic Data Generation

Synthetic data can be tuned to add privacy without losing utility or exposing individual data points.

##### Required Python Packages

```python
pip install tensorflow 
```

To check if Tensorflow is properly configured with the GPU:

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

# Scalability Tests

Training time for GANs is a significant factor. Tests were conducted to observe how training time varies with increasing number of rows and columns for TGAN and CTGAN.

1. For CTGAN, training time is affected by both the number of rows and columns.
2. For TGAN, training time is mainly affected by the number of columns.
3. CTGAN generally takes less time to train than TGAN.

# Statistical Similarity

To calculate similarity, we measure how different the synthetic data is from the original data.

1. Column-wise Similarity: Using KL-divergence and Cosine Similarity.
2. Table-wise Similarity: Using Autoencoders, PCA, and Clustering metrics.

# Privacy Risk Module

The goal is to define how to assess the privacy risk inherent in synthetic data. We developed the "Privacy At Risk (PaR)" metric.

- Internal Similarity > External Similarity: Ideal data point.
- External Similarity > Internal Similarity: Risky data point.
- Privacy Risk = Number of Risky Points / Total Number of Rows.

# Model Compatibility

We evaluate if models generated using synthetic data are compatible with original data.

- Use Case 1 (Length of Stay): Regression algorithms (Random Forest, XGBoost, KNN, Neural Networks).
- Use Case 2 (Mortality Prediction): Classification algorithms (Logistic Regression, XGBoost, Neural Networks).

# Ideas for Further Research and Improvements

1. Generating Larger Datasets from Small Samples: Using GANs to augment training data for deep learning.
2. Primary Keys: Modifying loss functions to ensure primary key consistency.
3. Privacy Tuning: Balancing the tradeoff between utility and privacy.
4. Model Efficiency: Optimizing code to parallelize operations and reduce time complexity.

# Sources

1. Synthesizing Tabular Data using Generative Adversarial Networks (arXiv:1811.11264)
2. Differentially Private Generative Adversarial Networks (arXiv:1802.06739)
3. PATE-GAN: Generating Differentially Private Synthetic Data (arXiv:1906.09338)
4. Scikit-learn Documentation for Machine Learning Metrics.

---

# Maintainer

**Vishal Sreeramareddy**
Supply Chain and Operations Professional | PMP Certified | Lean Six Sigma Green Belt

Vishal is a project management and operations professional with over 3 years of experience in data-driven process optimization. He maintains this repository to explore advanced analytics and deep learning applications for synthetic data generation and privacy preservation.

- **Email**: vishal.venki163@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/vishal-sreeramareddy
- **Skills**: SQL, Python (Pandas, NumPy), VBA, Project Management (Agile/Waterfall)

This project is maintained to provide a robust framework for generating and validating synthetic tabular data while respecting the original research and methodologies established in the field.