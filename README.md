# Space Biofilms: Predicting Biofilm Growth in Microgravity

**CSE 8803 - Scientific Machine Learning Project**

Ankit Bansal, Aarushi Biswas, Aarushi Gajri, Kashvi Mundra

[GitHub Repository](https://github.com/sciankit/space_biofilm)

---

## Overview

Biofilms (bacterial colonies) growing on spacecraft surfaces are a real problem for long missions - they clog equipment, mess with life support systems, and can make astronauts sick. The issue is that running experiments in space is ridiculously expensive and slow, so we can only test a handful of materials up there.

This project uses machine learning to predict how biofilms will grow in space based on what we observe on Earth. We built models that can tell you which spacecraft materials are most resistant to biofilm growth without having to actually test them in microgravity.

**Main result:** Our model trained only on Earth data predicted space biofilm coverage with 96% accuracy (R² = 0.96). Pretty wild that Earth patterns transfer so well to space.

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

This validated our architecture - if we ever get real time-series data from space, ConvLSTM should work.

---

### 2. Physics-Informed ConvLSTM

**File:** `Baseline-ConvLSTM_Physics/convlstm_physics_integration.ipynb`

Regular LSTMs produce noisy outputs that don't respect physics. We added two physics-based loss terms:

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

This is our best model. Key idea: train on Earth data, then test on space data to see if the patterns generalize.

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

The model learned relationships from Earth that generalized to microgravity. This is huge because it means we don't need to test every material in space.

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

## Future Work

- Get actual time-series data from space (same sample over time) to use ConvLSTM properly
- Test on more materials to improve generalization
- Integrate RNA-seq data for mechanism-aware predictions
- Combine physics PDEs with neural networks (hybrid models)

---

## Conclusion

We built a machine learning pipeline that predicts biofilm growth in space using only Earth-based training data. The Expert A model (Method B) achieved 96% accuracy on flight predictions, showing that fundamental biological patterns persist across gravity regimes. This could help NASA and other space agencies select materials for spacecraft without expensive orbital experiments.

The ConvLSTM architecture is validated and can be used when proper temporal datasets become available. Overall, this demonstrates that scientific machine learning can meaningfully contribute to space mission planning and astronaut safety.

---

## Acknowledgments

Thanks to Dr. Pamela Flores (CU Boulder), Dr. Luis Zea (CU Boulder), and NASA for the open-access OSD-554 and OSD-627 datasets.

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
