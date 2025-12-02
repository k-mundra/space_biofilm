# Space Biofilms: Predicting Biofilm Growth in Microgravity

**CSE 8803 - Scientific Machine Learning Project**

Ankit Bansal, Aarushi Biswas, Aarushi Gajri, Kashvi Mundra

---

## Overview

Biofilm growth poses significant risks to astronaut health, life-support systems, and spacecraft materials during long-duration missions. This creates an imperative to study their growth parameters; however, experimental characterization of biofilm behavior in microgravity remains costly and limited, restricting the ability to evaluate microbial risk in spaceflight environments.

To mitigate the challenge of data availability, we develop a multimodal scientific machine learning framework that predicts biofilm growth on engineering surfaces in microgravity by leveraging Earth-based biofilm measurements, along with available spaceflight observations. Our approach integrates confocal microscopy-derived morphological features, material surface descriptors, and optional gene-expression profiles into a convolutional Long Short-Term Memory (ConvLSTM) architecture capable of modeling biofilm structure and coverage. Domain insights from diffusion-reaction transport physics are incorporated into the design of feature representations and learning objectives.

Using this framework, we evaluate whether growth characteristics observed on Earth can be translated to microgravity conditions and estimate biofilm coverage on materials not experimentally tested in space. Time series prediction using ConvLSTM showed more than 90% accuracy in predicting growth patterns. Notably, a ground-only Random Forest model explained 99% of variance in terrestrial biofilm coverage and retained 96% explanatory power when predicting flight biofilm coverage despite never being exposed to microgravity data during training (R²<sub>ground</sub> = 0.99, R²<sub>flight</sub> = 0.96). These findings demonstrate the potential of scientific machine learning to guide material selection and microbial risk mitigation for future space missions.

---

## Key Results

| Model | Functionality | Performance |
|-------|--------------|-------------|
| **Method B (Expert A)** | Trained on ground data, tested on flight data | R² = 0.96 on flight prediction |
| Baseline Random Forest | Ground → Ground prediction | R² = 0.99 |
| Method A | Direct ground-to-flight mapping | R² = 0.50 (struggles with domain shift) |
| Synthetic ConvLSTM | Proof-of-concept temporal prediction | >90% accuracy |

---

## Repo Structure

```
space_biofilm/
├── Baseline-ConvLSTM_Physics/
│   └── convlstm_physics_integration.ipynb    # Physics-informed ConvLSTM
├── Baseline-Synthetic_Code/
│   └── synthetic.ipynb                        # Synthetic data generation
├── Datasets/
│   ├── OSD-554/                               # RNA-seq data
│   └── OSD-627/                               # Microscopy images
├── Figures/                                   # Output plots
├── Final_Model/
│   ├── sciml_MoE.ipynb                        # Main experimentation notebook
│   ├── methodA.py                             # Method A implementation
│   ├── methodB.py                             # Method B (Expert A) implementation
│   ├── lsds55_outputs_methodA/                # Method A results
│   ├── lsds55_outputs_expertA_methodB/        # Method B results
│   └── expertA_finalResults/                  # Final results + comparison plots
└── Archive/                                   # Old code (not used)
```

---

## Model Overview

### 1. Synthetic ConvLSTM (Proof of Concept)

**File:** `Baseline-Synthetic_Code/synthetic.ipynb`

We needed to verify that ConvLSTMs could learn biofilm growth patterns, so we generated fake data using diffusion equations. The model learned to predict biofilm evolution with >90% accuracy on synthetic sequences.

**Inputs:** 
- `timesteps=50` (frames per sequence)
- `n_runs=10` (number of trajectories)

**Output:** 144×144 pixel images saved to `bacteria_growth/run_XXX/frame_XXX.png`

This validated our architecture.

---

### 2. Physics-Informed ConvLSTM

**File:** `Baseline-ConvLSTM_Physics/convlstm_physics_integration.ipynb`

Regular LSTMs produce noisy outputs, so we added two physics-based loss terms:

1. **Laplacian smoothing** - Penalizes sharp edges, enforces smooth diffusive growth
   ```
   Kernel: [0  1  0]
           [1 -4  1]
           [0  1  0]
   ```

2. **Gradient penalty** - Removes checkerboard artifacts

Result: cleaner predictions that look like actual biofilm growth.

---

### 3. Random Forest Models (Main Results)

After starting the project, we discovered the NASA dataset doesn't have true time-series data - "Day 1", "Day 2", "Day 3" are actually different samples, not the same biofilm over time. So we pivoted to predicting **biofilm coverage percentage** instead of morphological evolution.

#### Baseline: Ground-Only Random Forest

**Location:** Top of `Final_Model/sciml_MoE.ipynb`

Simple baseline trained and tested only on Earth data. R² = 0.99 on ground samples.

---

#### Method A: Paired Ground-to-Flight Training

**File:** `Final_Model/methodA.py`

Tries to learn a direct mapping from ground biofilm features to flight coverage using matched material samples.

**Training:**
- Input: ground coverage, material properties, morphological features
- Output: flight coverage

**Run it:**
```bash
cd Final_Model
python methodA.py
```

**Results:** R² = 0.50 after calibration. Works okay but struggles with the Earth→Space domain shift.

**Outputs saved to:** `lsds55_outputs_methodA/`

---

#### Method B: Expert A (Final)

**File:** `Final_Model/methodB.py`

Key idea: train on Earth data, then test on space data to see if the patterns generalize.

**Training:**
- Data: Ground samples only
- Features: biofilm mass, thickness, roughness, material type, day
- Output: ground coverage percentage

**Testing:**
- Input: Flight morphological features (thickness, roughness, etc.) - **no material properties used**
- Output: predicted flight coverage

**Run it:**
```bash
cd Final_Model
python methodB.py
```

**Results:**
- Ground training: R² = 0.99
- **Flight testing: R² = 0.96** (never saw space data during training!)
- Flight RMSE: 8.0%, MAE: 6.3%

The model learned relationships from Earth that generalized to microgravity.

**Outputs:**
- Initial results: `lsds55_outputs_expertA_methodB/`
- Final plots: `expertA_finalResults/`

---

## The Data

### NASA OSD-627 (Microscopy Images)
- Confocal microscopy of *P. aeruginosa* biofilms
- 6 materials: Cellulose, LIS, SS316, Silicone, Silicone DLIP
- 3 time points: Day 1, 2, 3
- Parallel ground and flight experiments

**Important caveat:** Different days = different physical samples (not a true time series)

### NASA OSD-554 (RNA-seq)
- Gene expression data
- Not used in final models but available for future work

---

## Running Models

**Prerequisites:**
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib opencv-python
```

**1. Synthetic ConvLSTM:**
```bash
jupyter notebook Baseline-Synthetic_Code/synthetic.ipynb
```

**2. Physics ConvLSTM:**
```bash
jupyter notebook Baseline-ConvLSTM_Physics/convlstm_physics_integration.ipynb
```

**3. Method A:**
```bash
cd Final_Model
python methodA.py
```

**4. Method B (recommended):**
```bash
cd Final_Model
python methodB.py
```

**5. Full notebook with all experiments:**
```bash
jupyter notebook Final_Model/sciml_MoE.ipynb
```

---

## Takeaways

1. **Ground patterns transfer to space** - Morphology-to-coverage relationships learned on Earth work in microgravity (R² = 0.96)
2. **ConvLSTM works for biofilm prediction** - Validated on synthetic data, ready for real time-series when available
3. **Physics-informed losses help** - Adding diffusion constraints reduces artifacts
4. **Domain shift is real** - Direct ground→flight mapping (Method A) struggles without careful feature engineering

---

## Future Directions

- Acquire Temporally-Connected Data: Enable true ConvLSTM-based morphological prediction
- Multi-Material Transfer Learning: Improve generalization across diverse spacecraft materials
- Physics-Hybrid Models: Combine ConvLSTM with diffusion-reaction PDEs
- Gene Expression Integration: Incorporate RNA-seq data for mechanism-aware predictions
- Active Learning: Guide future space experiments by identifying high-uncertainty material combinations

---

## Conclusion

This project demonstrates that scientific machine learning can effectively bridge the gap between Earth-based biofilm experiments and spaceflight observations. Despite the absence of true temporal connectivity in the available data, our Expert A model achieved 96% explanatory power on flight biofilm coverage after training exclusively on ground data. This suggests that fundamental morphology-coverage relationships are preserved across gravity regimes, enabling predictive material selection for future space missions.
The validated ConvLSTM architecture on synthetic data provides a proof-of-concept for spatiotemporal biofilm modeling, which can be applied when temporally-connected datasets become available. Together, these approaches establish a scalable SciML framework for microbial risk assessment in spaceflight environments.

---

## Acknowledgments

Thank you to Dr. Pamela Flores (CU Boulder), Dr. Luis Zea (CU Boulder), and NASA for the open-access OSD-554 and OSD-627 datasets.

---

## Citation

```bibtex
@misc{bansal2025spacebiofilms,
  title={Space Biofilms: Predicting Bacterium-Material Interaction Dynamics in Microgravity},
  author={Bansal, Ankit and Biswas, Aarushi and Gajri, Aarushi and Mundra, Kashvi},
  year={2025},
  note={CSE 8803 Scientific Machine Learning Course Project}
}
```
