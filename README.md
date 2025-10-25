# Course Project One Pager
**Advanced Foundations of Machine Learning (CSE-AIML)**

---

## Team Details
| Name | Section | SRN |
|------|---------|-----|
| Aarya Upadhya | A | PES1UG23AM006 |
| A Haveesh Kumar | A | PES1UG23AM001 |
| Anshull M Udyavar | A | PES1UG23AM057 |
| Abhay H Bhargav | A | PES1UG23AM008 |

---

## Project Title
**Physics-Informed Disentangled Representations for Uncertainty-Aware Chest X-Ray Analysis: A Rigorous Hypothesis-Driven Study**

---

## Dataset and Sources
- **Primary Dataset:** NIH ChestX-ray14 (15,000 samples) + Kaggle Pneumonia (5,856 samples)
- **Total:** 20,856 images (12,901 NORMAL, 7,955 PNEUMONIA)
- **Splits:** 70% train (14,599), 15% val (3,128), 15% test (3,129)
- **Access/Sources:** Kaggle datasets, TorchXRayVision for preprocessing and lung segmentation
- **Preprocessing:** Self-supervised attenuation map (μ) extraction using Beer-Lambert law inversion

---

## Research Motivation

### The Clinical Problem
Current deep learning models for chest X-ray analysis suffer from three critical failures:
1. **Poor Calibration:** High accuracy but unreliable confidence estimates (dangerous for clinical deployment)
2. **Uninterpretable Reasoning:** Black-box latent representations lack semantic meaning
3. **Undefined Uncertainty:** Cannot distinguish between data noise, model limitations, and spurious correlations

### The Scientific Gap
- Existing uncertainty quantification methods (MC Dropout, ensembles) treat all uncertainty as monolithic
- Physics-informed neural networks (PINNs) focus on PDEs/reconstruction, not classification or representation learning
- No prior work rigorously validates whether physics constraints **causally enforce** semantic meaning in latent space

---

## Four Fundamental Hypotheses (Top-Tier Publishable)

### **H1: Physics Constraints Causally Enforce Disentanglement** ⭐ Theory Foundation
**Hypothesis:** *Physics-constrained latent factors (z_μ) exhibit significantly higher correlation with ground-truth physical properties (μ_true) compared to unconstrained factors, demonstrating causal encoding rather than spurious correlation.*

**Research Question:** Does forcing `z_μ` to reconstruct physics-based attenuation **actually make** it encode density, or does it just learn a different random pattern?

**Metrics:**
- **Primary:** Pearson correlation ρ(z_μ, μ_ground_truth)
  - Baseline (Cohort A): ρ < 0.40 (weak/random)
  - Physics (Cohort B): ρ > 0.75 (strong causal)
- **Secondary:** Mutual Information I(z_μ ; μ_true), Intervention test (perturb I₀ by ±20%)

**Statistical Test:** Paired t-test across 3 seeds, p < 0.01, Cohen's d > 2.0 (large effect)

**Why Foundational:** Establishes that physics loss is not just regularization—it **causally enforces** semantic meaning. Future papers can cite this when justifying physics-informed architectures.

---

### **H2: Physics-Mismatch Identifies Distribution Shift & Annotation Errors** ⭐ Novel Contribution
**Hypothesis:** *Samples with high physics-mismatch uncertainty (top 10% by ||μ_pred - μ_physical||²) are enriched for out-of-distribution cases and annotation errors at rates significantly above random.*

**Research Question:** Is physics-mismatch a **genuine signal** of model failure, or just noise?

**Metrics:**
- **Primary:** Precision@10% for identifying annotation errors
  - Random baseline: 5-10%
  - Physics-mismatch: >20%
- **Secondary:** Correlation with prediction errors, OOD detection on external dataset

**Validation:** Manual review of 200 high-mismatch cases by team (inter-rater agreement κ > 0.7)

**Statistical Test:** Chi-squared test for enrichment, p < 0.01

**Why Foundational:** Introduces **physics-mismatch as a new uncertainty type** beyond aleatoric/epistemic. Future uncertainty quantification papers will cite this taxonomy.

---

### **H3: Physics Constraints Improve Calibration Without Sacrificing Accuracy** ⭐ Practical Impact
**Hypothesis:** *Physics-informed models achieve statistically equivalent classification accuracy while demonstrating superior calibration (lower ECE) compared to baseline models, particularly in low-confidence regions.*

**Research Question:** Is there a trade-off between interpretability and performance, or can we achieve a Pareto improvement?

**Metrics:**
- **Equivalence Test:** |ΔAUROC| < 0.02 (non-inferiority)
- **Superiority Test:** ECE_physics < ECE_baseline by >0.03
- **Stratified Analysis:** Accuracy in uncertain region (P ∈ [0.3, 0.7])

**Statistical Test:** Wilcoxon signed-rank test, p < 0.01, Bootstrap CI (1000 iterations)

**Why Foundational:** Addresses skepticism that "interpretability hurts accuracy." Proves you can have **both** calibration and performance.

---

### **H4: Compositional Uncertainty Decomposition is Algebraically Valid** ⭐ Methodological Rigor
**Hypothesis:** *Total predictive uncertainty can be compositionally decomposed into independent contributions from aleatoric (σ²_data), epistemic (σ²_model), and physics-mismatch (σ²_physics) sources, with σ²_total ≈ σ²_aleatoric + σ²_epistemic + σ²_physics (R² > 0.85).*

**Research Question:** Are the three uncertainty types **actually independent**, or just different names for the same thing?

**Metrics:**
- **Primary:** Linear regression R² for σ²_total ~ β₁·σ²_aleatoric + β₂·σ²_epistemic + β₃·σ²_physics
  - Expected: R² > 0.85 (good decomposition)
- **Secondary:** Mutual Information I(σ²_i ; σ²_j) for i ≠ j (should be low)
- **Qualitative:** Case studies showing distinct failure modes per uncertainty type

**Statistical Test:** Cross-validated R², residual analysis, independence tests

**Why Foundational:** Provides a **validated template** for uncertainty decomposition. Future papers will cite this as the methodological standard for multi-source uncertainty quantification.

---

## Experimental Design: Three Cohorts

### **Cohort A: Baseline (No Physics Constraints)**
- Architecture: Standard β-VAE + MC Dropout Classifier
- Latent space: Single unified latent vector `z` (64-dim)
- Loss: L_recon + β·L_KL + L_focal (classification)
- Purpose: Establish baseline for H1, H3, H4

**Expected Results:**
- H1: ρ(z, μ) ≈ 0.30 (weak correlation)
- H3: AUROC ≈ 0.89, ECE ≈ 0.12 (poor calibration)
- H4: Only 2 uncertainty sources (no physics-mismatch)

### **Cohort B: Physics-Informed Disentangled VAE**
- Architecture: VAE with factorized latents `[z_μ, z_anatomy, z_pathology]`
- Physics decoder: `z_μ` → μ_reconstructed (forced to match μ_ground_truth)
- Loss: L_recon + β·L_KL + **λ_physics·L_physics** + γ·L_disentangle + L_focal
- Purpose: Test all 4 hypotheses

**Expected Results:**
- H1: ρ(z_μ, μ) > 0.75 (strong causal encoding)
- H2: Physics-mismatch identifies 20-25% errors
- H3: AUROC ≈ 0.89, ECE ≈ 0.05 (improved calibration)
- H4: R² > 0.85 for 3-way decomposition

### **Cohort C: Ablation Studies**
- Vary λ_physics ∈ {0.1, 1.0, 10.0, 50.0}
- Remove disentanglement loss
- Perturb I₀ by ±20% in Beer-Lambert
- Purpose: Sensitivity analysis, validate H1

**Key Analysis:**
- How does λ_physics affect ρ(z_μ, μ)?
- Is physics constraint robust to I₀ misspecification?

---

## Technical Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: PREPROCESSING (Self-Supervised, Run Once)     │
├─────────────────────────────────────────────────────────┤
│ X-ray → Lung Segmentation (TorchXRayVision UNet)       │
│      → Estimate I₀ from background air regions          │
│      → Beer-Lambert Inversion: μ = -ln(I / I₀)         │
│      → Save μ_map as .npy file                          │
│                                                         │
│ Output: train/val/test CSVs with 'mu_path' column      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: TRAINING (Physics-Constrained Learning)       │
├─────────────────────────────────────────────────────────┤
│ Forward Pass:                                           │
│   X → Encoder → [z_μ, z_anatomy, z_pathology]          │
│   z_μ → Physics Decoder → μ_reconstructed               │
│   [all z] → Image Decoder → X_reconstructed             │
│   [all z] → Classifier → P(pneumonia)                   │
│                                                         │
│ Loss Function:                                          │
│   L_total = L_recon + β·L_KL + λ_physics·||μ_recon - μ_target||² │
│           + γ·L_disentangle + L_focal                   │
│                                                         │
│ Physics Loss ← Forces z_μ to encode attenuation!       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: INFERENCE (Uncertainty Quantification)        │
├─────────────────────────────────────────────────────────┤
│ Aleatoric Uncertainty:                                  │
│   σ²_aleatoric = Var[z|X] from encoder variance        │
│                                                         │
│ Epistemic Uncertainty:                                  │
│   σ²_epistemic = Var[P(y|z)] via MC Dropout (20 passes)│
│                                                         │
│ Physics-Mismatch Uncertainty:                           │
│   σ²_physics = ||μ_predicted - μ_physical||²           │
│                                                         │
│ Total: σ²_total = σ²_aleatoric + σ²_epistemic + σ²_physics │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics (Comprehensive)

### **H1: Physics Encoding Quality**
- Pearson correlation ρ(z_μ, μ_true)
- Mutual Information I(z_μ ; μ_true)
- Disentanglement score (MIG - Mutual Information Gap)
- Intervention causality test (perturb I₀, measure prediction shift)

### **H2: Physics-Mismatch Validation**
- Precision@k (k=10%) for error detection
- Recall for OOD detection
- Confusion matrix conditioned on mismatch level
- Qualitative analysis (manual review of 200 cases)

### **H3: Calibration vs Accuracy**
- AUROC (classification performance)
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Stratified accuracy by confidence bins

### **H4: Uncertainty Decomposition**
- Linear regression R² (σ²_total ~ sources)
- Mutual Information I(σ²_i ; σ²_j) for independence
- Variance explained by each source
- Predictive entropy H[P(y|X)]
- Case studies (5 samples × 4 uncertainty types)

### **Additional Quality Metrics**
- Reconstruction MSE
- GradCAM attention alignment with high-μ regions
- Training stability (loss curves)
- Computational efficiency (inference time)

---

## Implementation Details

### **Training Configuration**
- Optimizer: Adam (lr=1e-4)
- Batch size: 32
- Epochs: 50
- Hardware: GPU (CUDA)
- Seeds: 42, 123, 456 (for reproducibility)

### **Hyperparameters**
- Latent dimension: 64 per factor
- β (KL weight): 1.0
- λ_physics (physics loss weight): 10.0
- γ (disentanglement weight): 1.0
- MC Dropout: 20 forward passes
- Dropout rate: 0.3

### **Loss Function Weights**
```python
L_total = L_recon                    # Image reconstruction
        + 1.0 · L_KL                 # VAE regularization
        + 10.0 · L_physics           # Physics constraint (KEY!)
        + 1.0 · L_disentangle        # Factor independence
        + L_focal                    # Classification (α=0.25, γ=2.0)
```

---

## Expected Contributions

### **1. Theoretical Foundation (H1)**
- First rigorous proof that physics constraints **causally enforce** semantic latent representations
- Intervention-based causality analysis in representation learning
- Citation: "Following [Your Work], we use physics constraints to enforce disentanglement..."

### **2. Novel Uncertainty Type (H2)**
- Introduction of **physics-mismatch uncertainty** as a distinct source beyond aleatoric/epistemic
- Validated method for detecting annotation errors and distribution shift
- Citation: "We decompose uncertainty following [Your Work]: aleatoric, epistemic, and physics-mismatch..."

### **3. Practical Impact (H3)**
- Proof that physics-informed models achieve **Pareto improvements** (calibration without accuracy loss)
- Evidence against "interpretability hurts performance" narrative
- Citation: "Contrary to common belief, [Your Work] showed physics constraints improve calibration while maintaining accuracy..."

### **4. Methodological Template (H4)**
- Validated framework for **compositional uncertainty decomposition**
- Reproducible experimental design for multi-source uncertainty
- Citation: "We follow the uncertainty decomposition methodology from [Your Work]..."

---

## Timeline (7 Weeks)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| 1 | Preprocess dataset, extract μ_maps | train/val/test_with_mu.csv |
| 2 | Implement Cohort A (Baseline VAE) | Trained models (3 seeds) |
| 3 | Implement Cohort B (Physics VAE) | Trained models (3 seeds) |
| 4 | Test H1: Correlation analysis, interventions | H1 results, visualizations |
| 5 | Test H2: Manual review of high-mismatch cases | H2 results, error taxonomy |
| 6 | Test H3 & H4: Calibration, decomposition | H3/H4 results, plots |
| 7 | Write paper, create figures | Final report |

---

## Why This is Top-Tier Publishable

### **1. Rigorous Hypothesis Testing**
- Four pre-registered hypotheses with clear success criteria
- Statistical tests with effect sizes, confidence intervals
- Multiple baselines and ablations

### **2. Novel Scientific Contributions**
- Physics-mismatch as new uncertainty type (H2)
- Causal validation of physics constraints (H1)
- First work on physics-informed representation learning for classification

### **3. Practical Clinical Impact**
- Actionable uncertainty information for radiologists
- Interpretable failure modes
- Improved calibration for clinical deployment

### **4. Reproducible & Extensible**
- Clear methodology others can follow
- Open questions for future work (e.g., extend to other modalities)
- Code and data will be released

### **5. Strong Baselines**
- Compare against standard uncertainty methods (MC Dropout, ensembles)
- Ablation studies validate each component
- External validation on held-out data

---

## Reference Papers

| Paper Title | Link | Relevance |
|-------------|------|-----------|
| 1. PINNs for Medical Image Analysis: A Survey | https://arxiv.org/html/2408.01026v1 | PINNs background |
| 2. Quantification of Total Uncertainty in PINNs | https://royalsocietypublishing.org/doi/10.1098/rsta.2024.022 | Uncertainty theory |
| 3. Focal Loss for Dense Object Detection | https://arxiv.org/abs/1708.02002 | Class imbalance |
| 4. Dropout as Bayesian Approximation | https://proceedings.mlr.press/v48/gal16.html | Epistemic uncertainty |
| 5. β-VAE: Learning Basic Visual Concepts | https://openreview.net/forum?id=Sy2fzU9gl | Disentanglement |
| 6. Measuring Calibration in Deep Learning | https://arxiv.org/abs/1904.01685 | ECE, Brier score |

---

## Research Question Summary

**Central Question:** *Can physics-informed disentangled representations simultaneously improve calibration, interpretability, and uncertainty quantification compared to purely data-driven approaches?*

**Answer (Expected):** Yes—through four validated mechanisms:
1. Physics **causally enforces** semantic meaning (H1)
2. Physics-mismatch **detects failures** (H2)
3. Physics **improves calibration** without accuracy loss (H3)
4. Multi-source uncertainty is **compositionally valid** (H4)

This establishes **physics-informed disentangled representations as a principled approach** for building trustworthy medical AI systems.
