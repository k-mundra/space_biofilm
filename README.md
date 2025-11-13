read me file

Overleaf document: https://www.overleaf.com/6681458395rnncqfcpswqm#5edb39 


Missing : Code documentation 

## Datasets
- OSD-554
    - rna-seq data
- OSD-627
    - microscopy image data


## synthetic data code.
This code is a based on diffusion equation solving to create biofilm growth data.

inputs
-- timesteps=50 (control how many images per run are saved.)
-- n_runs = 10 (controls how many trajectories are saved.)

outputs
-- 144X144 pixel image of the stochastic biofilm growth, timesteps number of images for each trajectory. In total n_runs trajectories.

everything is saved in bacteria_growth folder.

- bacteria_growth
    - run_000
        - frame_000.png
        - frame_001.png
        - ...
    - run_001
    - ...


## Baseline-convLSTM
this is directly downloaded from the reference model. No changes are made. However when the code was tested, it was not producing results shown online.

## Baseline-Conv_LSTM_with_biofilm.
It is one of the most updated, model right now. It is made to work with synthetic data generated from "synthetic data code". Output of the code is copied into the folder "bacteria_growth".

Inputs
-- bacteria_growth

Outputs
-- trained_model.h5
-- comparison plots in the "results" folder.


## Background-physics-Conv_LSTM_with_biofilm
The LSTM model adds a background noise which in principle this reduces the overall loss (add more information). However this noise is deterimental to the lond time frame prediction. To resolve this problem, this model will incorporate diffusion based equation into loss, ensuring those background noise are not favoured.
