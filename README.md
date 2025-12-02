<h1>Scientific Machine Learning for Microgravity Biofilm Prediction</h1>

<p>
Biofilm growth poses significant risks to astronaut health, life-support systems, and spacecraft materials during long-duration missions. Experimental characterization of biofilm behavior in microgravity is expensive and sparse, making predictive modeling essential for microbial risk mitigation.
</p>

<p>
This repository develops a multimodal <strong>scientific machine learning (SciML)</strong> framework that uses Earth-based biofilm data, limited spaceflight observations, and material surface properties to estimate microgravity biofilm growth. Our pipeline combines synthetic ConvLSTM modeling, physics-informed learning, and a Random Forest regression approach for cross-gravity generalization.
</p>

<hr>

<h2>⭐ Project at a Glance</h2>

<ul>
  <li><strong>Goal:</strong> Predict microgravity biofilm coverage using only Earth-based measurements and material properties.</li>
  <li><strong>Data Sources:</strong>
    <ul>
      <li>NASA OSD-627 (confocal biofilm morphology)</li>
      <li>NASA OSD-554 (RNA transcriptomics; optional multimodal input)</li>
    </ul>
  </li>
  <li><strong>Methods:</strong>
    <ul>
      <li>Synthetic ConvLSTM (sanity check for temporal prediction)</li>
      <li>Physics-informed ConvLSTM (Laplacian smoothing)</li>
      <li>Random Forest Ground→Flight model (core method)</li>
    </ul>
  </li>
  <li><strong>Key Result:</strong>  
    A model trained exclusively on ground data explained <strong>96% of variance</strong> in flight biofilm coverage  
    (<em>R²_ground = 0.99, R²_flight = 0.96</em>).
  </li>
</ul>

<hr>

<h2>📂 Repository Structure</h2>

<pre>
├── Synthetic/
│   ├── Synthetic_ConvLSTM_Code/
│   └── Synthetic Data Code Explanation & Results
│
├── Physics/
│   ├── Code
│   └── Experiments with Laplacian Kernel (Physics-Informed ConvLSTM)
│
├── Final_Model/
│   ├── strict_ground_to_flight_regression.py
│   ├── cleaned_combined.csv
│   ├── Outputs (metrics, plots, prediction tables)
│   └── Material property–enhanced multimodal regression
│
└── figures/
    ├── synthetic_results.png
    ├── physics_kernel_example.png
    └── rf_results.png
</pre>

<hr>

<h2>🧪 Datasets</h2>

<h3>1. NASA OSD-627 — Confocal Microscopy Biofilm Dataset</h3>
<ul>
  <li>3D confocal scans of biofilms grown on engineering surfaces</li>
  <li>Collected on <strong>ground</strong> and <strong>spaceflight</strong></li>
  <li>Includes:
    <ul>
      <li>Percent surface coverage</li>
      <li>Maximum thickness</li>
      <li>Roughness coefficient (Ra*)</li>
    </ul>
  </li>
</ul>

<h3>2. NASA OSD-554 — RNA Transcriptomics</h3>
<p>This dataset is included as an optional leg. Gene expression profiles for matched conditions; used for multimodal extensions.</p>

<h3>3. Material Property Features</h3>

<table>
  <tr><th>Feature</th><th>Description</th></tr>
  <tr><td>Material_Roughness</td><td>Surface nano/micro roughness</td></tr>
  <tr><td>Contact_Angle</td><td>Measured wettability</td></tr>
  <tr><td>Cos(theta)</td><td>Wetting energy representation</td></tr>
  <tr><td>Wadh</td><td>Work of adhesion</td></tr>
</table>

<hr>

<h2>🧠 Methodology</h2>

<h3>1. Synthetic ConvLSTM (Baseline Sanity Check)</h3>
<p>
To verify temporal learnability, we generated <strong>synthetic biofilm sequences</strong> using a diffusion-reaction surrogate model.  
A ConvLSTM was trained to predict future frames.
</p>
<ul>
  <li>&gt;90% accuracy on synthetic forecasting</li>
  <li>Demonstrated learnability of spatiotemporal transitions</li>
  <li>Provided architectural validation before real-data training</li>
</ul>

<h3>2. Physics-Informed ConvLSTM</h3>
<p>
We applied a <strong>Laplacian smoothing kernel</strong> to encourage predictions that obey diffusion-like spatial coherence.
</p>
<ul>
  <li>Reduces noise artifacts</li>
  <li>Produces smoother, more physically plausible predictions</li>
  <li>Aligns with diffusion-driven microbial biofilm behavior</li>
</ul>

<h3>3. Ground→Flight Random Forest Model (Final Method)</h3>

<p>This is the <strong>main contribution</strong> of the repository.</p>

<p><strong>What it does:</strong></p>
<ul>
  <li>Uses <strong>only ground-side features</strong> (coverage, thickness, Ra*, material properties)</li>
  <li>Maps these features to <strong>flight coverage outcomes</strong></li>
  <li>Tests whether Earth-derived biofilm behavior predicts microgravity behavior</li>
</ul>

<p><strong>Training Input:</strong></p>
<ul>
  <li>DayInt</li>
  <li>Ground coverage</li>
  <li>Ground thickness</li>
  <li>Ground roughness (biofilm)</li>
  <li>Material properties (4 descriptors)</li>
  <li>BaseMaterial (categorical)</li>
</ul>

<p><strong>Training Target:</strong> Mean <strong>flight</strong> coverage for the matched (material, day) pair.</p>

<p><em>Importantly, no flight data is used as input.</em></p>

<p><strong>Key Insight:</strong></p>
<p>
Despite training <strong>exclusively</strong> on Earth data, the model retained strong predictive structure in microgravity:
</p>
<ul>
  <li><strong>R²_ground = 0.99</strong></li>
  <li><strong>R²_flight = 0.96</strong></li>
</ul>

<p>This suggests biofilm-material relationships are surprisingly stable across gravity regimes.</p>

<hr>

<h2>📈 Key Results</h2>

<ul>
  <li>Synthetic ConvLSTM → Successful sanity check</li>
  <li>Physics-informed ConvLSTM → Physically smoother predictions</li>
  <li>Random Forest Ground→Ground: <strong>R² ≈ 0.99</strong></li>
  <li>Random Forest Ground→Flight: <strong>R² ≈ 0.96</strong></li>
</ul>

<p>
These results show that Earth-derived measurements contain reliable signatures that remain predictive in microgravity.
</p>

<hr>

<h2>🚀 How to Run the Final Model</h2>

<pre>
cd Final_Model
python strict_ground_to_flight_regression.py
</pre>

<p>Outputs saved to:</p>

<pre>
lsds55_outputs_strict_g2f/
    metrics.json
    strict_ground_to_flight_predictions.csv
    strict_ground_to_flight_pred_vs_actual.png
</pre>

<hr>

<h2>🖼️ Figures</h2>

<details>
<summary><strong>Synthetic ConvLSTM Results</strong></summary>
<p>(Insert image here)</p>
</details>

<details>
<summary><strong>Physics-Informed Kernel Example</strong></summary>
<p>(Insert image here)</p>
</details>

<details>
<summary><strong>Ground→Flight Random Forest Predictions</strong></summary>
<p>(Insert image here)</p>
</details>

<hr>

<h2>🙏 Acknowledgements</h2>
<p>
This work uses data from:
</p>
<ul>
  <li><strong>NASA GeneLab OSD-627</strong> (Confocal Biofilm Morphology)</li>
  <li><strong>NASA GeneLab OSD-554</strong> (RNA Transcriptomics)</li>
</ul>

<hr>

<h2>✔️ Final Notes</h2>
<p>
This repository demonstrates how <strong>scientific machine learning</strong>, guided by domain insight, can extract meaningful structure from extremely limited microgravity datasets. While not a full simulator of 3D biofilm dynamics, the models reveal stable cross-gravity relationships that can inform <strong>material selection</strong>, <strong>microbial risk assessment</strong>, and <strong>future spaceflight experiment design</strong>.
</p>
