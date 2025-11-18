# Physics-Informed Representation Learning for COVID-19 CT Classification

## Overview
[cite_start]This project implements an **Attribute-Regularized Soft Introspective Variational Autoencoder (ARSIVAE)** for classifying COVID-19 from Chest CT scans[cite: 10]. [cite_start]The study investigates a fundamental question in medical AI: *Can physics-based global features achieve sufficient discriminative power for classification while remaining physically interpretable?*[cite: 11].

## Methodology
* [cite_start]**Dataset:** Utilizes the MosMedData dataset, processed into a balanced cohort of 5,000 2D slices (2,500 COVID-positive, 2,500 Normal)[cite: 35].
* [cite_start]**Leak-Free Splitting:** Data was split at the patient level (70% Train, 15% Val, 15% Test) to prevent data leakage[cite: 36].
* [cite_start]**Feature Engineering:** A comprehensive pipeline extracts 14 domain-specific radiological physics features, including **Hounsfield Unit (HU) statistics**, **Gradient** (boundary sharpness), **Texture** (GLCM), and **Shape**[cite: 11, 42, 47].
* [cite_start]**Model Architecture:** The ARSIVAE uses a Dual-Pathway Predictor to constrain the latent space, forcing it to be physically meaningful (optimized for $R^2$) while simultaneously attempting classification[cite: 59, 60].

## Key Results
1.  [cite_start]**High Physics Fidelity:** The model successfully learns a physically interpretable latent space, achieving an average **$R^2$ of 0.89** in reconstructing the 14 attributes[cite: 12, 94].
2.  **Classification Ceiling:** Despite high interpretability, the model plateaus at a moderate performance:
    * [cite_start]**End-to-End:** 74.3% Accuracy[cite: 93].
    * [cite_start]**Linear Probe:** 70.0% Accuracy[cite: 93].

## The "Spatial Information Gap"
The analysis uncovers a critical limitation in using global physics features for localized pathologies:
* [cite_start]**Feature Overlap:** Statistical analysis reveals a **35-70% overlap** in global physics features between COVID-19 and Normal cases[cite: 13].
* [cite_start]**Localized Pathology:** Global averages fail to capture localized patterns like Ground-Glass Opacities (GGOs); a mild COVID case can statistically resemble a healthy lung[cite: 140].

## Conclusion
[cite_start]While global statistics are physically meaningful and well-reconstructed, they are insufficient for high-performance discrimination on their own[cite: 14]. [cite_start]This motivates the development of **hybrid architectures** that combine physics priors with learned spatial representations to bridge the "physics-discrimination gap"[cite: 15].

---
**Authors:** Anshull M Udyavar, Aarya Upadhya, Abhay Bhargav, A Haveesh Kumar  
[cite_start]**Affiliation:** PES University, Bengaluru, India [cite: 3-8]