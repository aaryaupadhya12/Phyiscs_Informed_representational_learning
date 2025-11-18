# Physics-Informed Representation Learning for COVID-19 CT Classification

## Overview
This project implements an **Attribute-Regularized Soft Introspective Variational Autoencoder (ARSIVAE)** for classifying COVID-19 from Chest CT scans.The study investigates a fundamental question in medical AI: *Can physics-based global features achieve sufficient discriminative power for classification while remaining physically interpretable?*.

## Methodology
* **Dataset:** Utilizes the MosMedData dataset, processed into a balanced cohort of 5,000 2D slices (2,500 COVID-positive, 2,500 Normal).
* **Leak-Free Splitting:** Data was split at the patient level (70% Train, 15% Val, 15% Test) to prevent data leakage.
* **Feature Engineering:** A comprehensive pipeline extracts 14 domain-specific radiological physics features, including **Hounsfield Unit (HU) statistics**, **Gradient** (boundary sharpness), **Texture** (GLCM), and
* **Model Architecture:** The ARSIVAE uses a Dual-Pathway Predictor to constrain the latent space, forcing it to be physically meaningful (optimized for $R^2$) while simultaneously attempting classification

## Key Results
1.  **High Physics Fidelity:** The model successfully learns a physically interpretable latent space, achieving an average **$R^2$ of 0.89** in reconstructing the 14 attributes.
2.  **Classification Ceiling:** Despite high interpretability, the model plateaus at a moderate performance:
    * **End-to-End:** 74.3% Accuracy.
    * **Linear Probe:** 70.0% Accuracy.

## The "Spatial Information Gap"
The analysis uncovers a critical limitation in using global physics features for localized pathologies:
* **Feature Overlap:** Statistical analysis reveals a **35-70% overlap** in global physics features between COVID-19 and Normal cases.
* **Localized Pathology:** Global averages fail to capture localized patterns like Ground-Glass Opacities (GGOs); a mild COVID case can statistically resemble a healthy lung.

## Conclusion
While global statistics are physically meaningful and well-reconstructed, they are insufficient for high-performance discrimination on their own. This motivates the development of **hybrid architectures** that combine physics priors with learned spatial representations to bridge the "physics-discrimination gap".

---
**Authors:** Anshull M Udyavar, Aarya Upadhya, Abhay Bhargav, A Haveesh Kumar  

**Affiliation:** PES University, Bengaluru, India
