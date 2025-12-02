# Space Biofilms: Predicting Bacterium-Material Interaction Dynamics in Microgravity Using Sequential Experimental Data

**Scientific Machine Learning Course Project (CSE 8803-SML)**

**Authors:** Ankit Bansal, Aarushi Biswas, Aarushi Gajri, Kashvi Mundra

**Repository:** https://github.com/sciankit/space_biofilm

---

## Overview

Biofilm growth poses significant risks to astronaut health, life-support systems, and spacecraft materials during long-duration missions. This creates an imperative to study their growth parameters; however, experimental characterization of biofilm behavior in microgravity remains costly and limited, restricting the ability to evaluate microbial risk in spaceflight environments.

To mitigate the challenge of data availability, we develop a multimodal scientific machine learning framework that predicts biofilm growth on engineering surfaces in microgravity by leveraging Earth-based biofilm measurements, along with available spaceflight observations. Our approach integrates confocal microscopy-derived morphological features, material surface descriptors, and optional gene-expression profiles into a convolutional Long Short-Term Memory (ConvLSTM) architecture capable of modeling biofilm structure and coverage. Domain insights from diffusion-reaction transport physics are incorporated into the design of feature representations and learning objectives.

Using this framework, we evaluate whether growth characteristics observed on Earth can be translated to microgravity conditions and estimate biofilm coverage on materials not experimentally tested in space. Time series prediction using ConvLSTM showed more than 90% accuracy in predicting growth patterns. Notably, a ground-only Random Forest model explained 99% of variance in terrestrial biofilm coverage and retained 96% explanatory power when predicting flight biofilm coverage despite never being exposed to microgravity data during training (R²<sub>ground</sub> = 0.99, R²<sub>flight</sub> = 0.96). These findings demonstrate the potential of scientific machine learning to guide material selection and microbial risk mitigation for future space missions.

---

## Repository Structure

```
space_biofilm/
├── Archive/                          # Archival code not used in final submission
├── Baseline-ConvLSTM_Physics/        # Physics-informed ConvLSTM model
│   └── convlstm_physics_integration.ipynb
├── Baseline-Synthetic_Code/          # Synthetic data generation and ConvLSTM training
│   └── synthetic.ipynb
├── Datasets/                         # NASA open-access datasets
│   ├── OSD-554/                      # RNA-seq data
│   └── OSD-627/                      # Microscopy image data
├── Figures/                          # Generated figures and visualizations
├── Final_Model/                      # Main Random Forest models and results
│   ├── sciml_MoE.ipynb              # Comprehensive experimentation notebook
│   ├── methodA.py                    # Method A: Ground-to-Flight paired training
│   ├── methodB.py                    # Method B: Expert A ground-trained model
│   ├── lsds55_outputs_methodA/       # Method A output directory
│   ├── lsds55_outputs_expertA_methodB/  # Method B output directory
│   └── expertA_finalResults/         # Final Method B results with comparison plots
└── README.md
```

---

## Models Overview

### 1. Synthetic Data ConvLSTM (Proof of Concept)

**Location:** `Baseline-Synthetic_Code/synthetic.ipynb`

**Purpose:** Validates the ConvLSTM architecture using synthetically generated biofilm growth data based on diffusion equation solving.

**Inputs:**
- `timesteps`: Number of frames per trajectory (default: 50)
- `n_runs`: Number of growth trajectories (default: 10)

**Outputs:**
- 144×144 pixel images of stochastic biofilm growth
- Saved in `bacteria_growth/` directory:
  ```
  bacteria_growth/
  ├── run_000/
  │   ├── frame_000.png
  │   ├── frame_001.png
  │   └── ...
  ├── run_001/
  └── ...
  ```

**Key Result:** Demonstrates that ConvLSTM can successfully learn spatiotemporal biofilm-like growth patterns in an idealized setting.

---

### 2. Physics-Informed ConvLSTM

**Location:** `Baseline-ConvLSTM_Physics/convlstm_physics_integration.ipynb`

**Purpose:** The LSTM model introduces background noise, which can reduce the overall loss but becomes detrimental for long-term predictions. To address this issue, the model incorporates a diffusion-based term and a total directional gradient term into the loss function, ensuring that such noise is not favored during training. The diffusion-based loss is computed using a discrete Laplacian kernel
  ```
  L = [0  1  0]
      [1 -4  1]
      [0  1  0]
  ```
which measures the curvature at each pixel. This curvature is then added to the loss, penalizing regions of high curvature and enforcing smoothness across the field. This smoothing effect reflects the outward, diffusive growth of biofilms, capturing essential physical behavior.

Additionally, a total directional gradient loss is included to suppress artifacts that may arise from background noise such as checkerboard patterns. By penalizing abrupt variations in pixel intensity, this term further promotes a smooth and physically realistic background.

**Key Result:** Physics-informed loss terms lead to smoother, more physically plausible predictions for long-term forecasting.

---

### 3. Random Forest Models (Final Model)

**Location:** `Final_Model/`

These models address the key finding that the OSD-627 dataset lacks true temporal connectivity (Day 1, 2, 3 samples are independent, not sequential). Therefore, we reframe the problem as predicting biofilm **surface coverage** rather than morphological evolution.

#### **Baseline Random Forest (Ground-Only)**

**Location:** `Final_Model/sciml_MoE.ipynb` (top code block)

**Purpose:** Establishes baseline performance using only Earth-gravity samples.

**Training & Testing:** Both performed on ground data only.

**Features:**
- Morphological descriptors (maximum thickness, roughness coefficient, surface coverage)
- Material properties
- No cross-gravity transfer

---

#### **Method A: Ground-to-Flight Paired Training**

**Location:** `Final_Model/methodA.py`

**Purpose:** Learn direct mapping from ground biofilm characteristics to flight biofilm coverage using paired material samples.

**Training:**
- **Inputs:** Ground material properties, ground biofilm coverage, ground morphological features (roughness, max thickness)
- **Outputs:** Flight biofilm coverage

**Testing:**
- **Inputs:** Unseen ground material data
- **Outputs:** Predicted flight biofilm coverage

**How to Run:**
```bash
cd Final_Model
python methodA.py
```

**Output Directory:** `Final_Model/lsds55_outputs_methodA/`

**Results:** 
- Initial R² = 0.190 (systematic bias observed)
- After linear calibration: R² = 0.502
- Model captures relative ordering but struggles with domain shift

---

#### **Method B: Expert A (Ground-Trained, Flight-Tested)**

**Location:** `Final_Model/methodB.py`

**Purpose:** Learn morphology-to-coverage relationships on Earth and test generalization to microgravity **without seeing any flight labels during training**.

**Model Definition (Expert A):**
- Random Forest regressor with 600 estimators
- Features: biofilm mass, mean thickness, max thickness, roughness coefficient, material type, incubation day, gravity condition
- Hyperparameter tuning via 3-fold cross-validation

**Training:**
- **Inputs:** Ground morphological features, day, condition
- **Outputs:** Ground biofilm surface coverage
- **Training set:** Ground samples only

**Testing:**
- **Inputs:** Flight morphological features, material properties, day, condition
- **Outputs:** Predicted flight biofilm coverage

**How to Run:**
```bash
cd Final_Model
python methodB.py
```

**Output Directories:**
- `Final_Model/lsds55_outputs_expertA_methodB/` - Initial outputs
- `Final_Model/expertA_finalResults/` - Final results with comparison plots

**Results:**
- **Random 70/30 split (mixed ground/flight):**
  - Training R² = 0.998
  - Testing R² = 0.98
  - Test RMSE = 5.36% coverage
  - Test MAE = 3.42% coverage

- **Ground-to-Flight blind test:**
  - Ground training R² = 0.99
  - **Flight prediction R² = 0.96** (never trained on flight data!)
  - Flight RMSE = 8.0% coverage
  - Flight MAE = 6.3% coverage

**Key Finding:** Morphology-to-coverage relationships learned at 1g generalize remarkably well to microgravity, suggesting fundamental biological patterns persist across gravity regimes.

---

## Dataset Information

### NASA OSD-627 (Microscopy Data)
- Confocal microscopy images of *Pseudomonas aeruginosa* biofilms
- Six base materials tested (Cellulose membrane, LIS, SS316, Silicone, Silicone DLIP, etc.)
- Three time points: Day 1, Day 2, Day 3
- Parallel experiments: Ground (1g) and Spaceflight (μg)
- **Important:** Samples at different days are **independent specimens**, not sequential observations

### NASA OSD-554 (RNA-seq Data)
- Gene expression profiles for biofilms under different conditions
- Can be integrated as optional input for multimodal models

### Creating Trajectories from OSD-627

To build growth trajectories for modeling, parse `s_OSD-627.txt`:

1. Extract: Spaceflight condition (Ground vs Space Flight), Material, Time (1, 2, 3 days)
2. Group by: (Spaceflight, Material, Medium)
3. Sort by Time within each group
4. Link corresponding sample IDs

Example trajectory:
```
(Space Flight, SS316, LB+KNO₃) → {1.*, 7.*, 13.*}
Day 1 → Day 2 → Day 3 = Sample IDs {1.*, 7.*, 13.*}
```

**Note:** These are **independent samples**, not true temporal sequences.

---

## Running the Code

### Requirements
- Python 3.8+
- TensorFlow/Keras (for ConvLSTM models)
- scikit-learn (for Random Forest models)
- NumPy, Pandas, Matplotlib
- OpenCV or PIL (for image processing)

### Execution Steps

1. **Synthetic Data ConvLSTM:**
   ```bash
   jupyter notebook Baseline-Synthetic_Code/synthetic.ipynb
   ```

2. **Physics ConvLSTM:**
   ```bash
   jupyter notebook Baseline-ConvLSTM_Physics/convlstm_physics_integration.ipynb
   ```

3. **Method A (Ground-to-Flight):**
   ```bash
   cd Final_Model
   python methodA.py
   ```

4. **Method B (Expert A):**
   ```bash
   cd Final_Model
   python methodB.py
   ```

5. **Full Experimentation Notebook:**
   ```bash
   jupyter notebook Final_Model/sciml_MoE.ipynb
   ```

---

## Key Results Summary

| Model | Training Data | Test Data | R² Score | Key Insight |
|-------|--------------|-----------|----------|-------------|
| Baseline RF | Ground only | Ground only | 0.99 | Strong morphology-coverage relationship on Earth |
| Method A | Ground → Flight (paired) | Flight | 0.502 (calibrated) | Direct transfer struggles with domain shift |
| Method B (Expert A) | Ground only | Flight | **0.96** | Ground-learned patterns generalize to microgravity |
| ConvLSTM (Synthetic) | Synthetic data | Synthetic data | >90% accuracy | Architecture validated for temporal prediction |

---

## Future Directions

1. **Acquire Temporally-Connected Data:** Enable true ConvLSTM-based morphological prediction
2. **Multi-Material Transfer Learning:** Improve generalization across diverse spacecraft materials
3. **Physics-Hybrid Models:** Combine ConvLSTM with diffusion-reaction PDEs
4. **Gene Expression Integration:** Incorporate RNA-seq data for mechanism-aware predictions

---

## Conclusions

This project demonstrates that scientific machine learning can effectively bridge the gap between Earth-based biofilm experiments and spaceflight observations. Our Expert A model achieved 96% explanatory power on flight biofilm coverage after training exclusively on ground data, suggesting that fundamental morphology-coverage relationships are preserved across gravity regimes. This enables predictive material selection for future space missions without requiring costly spaceflight experiments for every material combination.

The validated ConvLSTM architecture on synthetic data provides a proof-of-concept for spatiotemporal biofilm modeling, which can be applied when temporally-connected datasets become available. Together, these approaches establish a scalable SciML framework for microbial risk assessment in spaceflight environments, supporting safer and more efficient long-duration space missions.

---

## Acknowledgements

We thank Dr. Pamela Flores (CU Boulder), Dr. Luis Zea (CU Boulder), and NASA for making the OSD-554 and OSD-627 datasets publicly available.

---

## Citation

```bibtex
@misc{2025spacebiofilms,
  title={Space Biofilms: Predicting Bacterium-Material Interaction Dynamics in Microgravity Using Sequential Experimental Data},
  author={Bansal, Ankit and Biswas, Aarushi and Gajri, Aarushi and Mundra, Kashvi},
  year={2025},
  howpublished={CSE 8803 Scientific Machine Learning Course Project}
}
```

---

## License

This project uses NASA open-access datasets (OSD-554, OSD-627) and follows their respective data usage policies.
