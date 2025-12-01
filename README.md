## Space Biofilms: Predicting Bacterium-Material Interaction Dynamics in Microgravity Using Sequential Experimental Data

Scientific Machine Learning Course Project (CSE 8803-SML)

Ankit Bansal, Aarushi Biswas, Aarushi Gajri, Kashvi Mundra

Overleaf document: https://www.overleaf.com/6681458395rnncqfcpswqm#5edb39

## Overview

Biofilm growth poses significant risks to astronaut health, life-support systems, and spacecraft materials during long-duration missions. This creates an imperative to study their growth
parameters; however, experimental characterization of biofilm behavior in microgravity remains costly and limited, restricting the ability to evaluate microbial risk in spaceflight environments. To mitigate the challenge of data availability, we develop a multimodal scientific machine learning framework that predicts biofilm growth on engineering surfaces in microgravity
by leveraging Earth-based biofilm measurements, along with available spaceflight observations.
Our approach integrates confocal microscopy–derived morphological features, material surface
descriptors, and optional gene-expression profiles into a convolutional Long Short-Term Memory (ConvLSTM) architecture capable of modeling biofilm structure and coverage. Domain
insights from diffusion–reaction transport physics are incorporated into the design of feature
representations and learning objectives. Using this framework, we evaluate whether growth
characteristics observed on Earth can be translated to microgravity conditions and estimate
biofilm coverage on materials not experimentally tested in space. Time series prediction using
convLSTM showed more than 90% accuracy in predicting growth patterns. !! [Insert key
quantitative results here] !!. These findings demonstrate the potential of scientific machine
learning to guide material selection and microbial risk mitigation for future space missions.

## Datasets provided by NASA
- OSD-554
    - rna-seq data
- OSD-627
    - microscopy image data
 

## Synthetic Data Code
This code is a based on diffusion equation solving to create biofilm growth data.

inputs
-- timesteps=50 (control how many images per run are saved.)
-- n_runs = 10 (controls how many trajectories are saved.)

outputs
-- 144X144 pixel image of the stochastic biofilm growth, timesteps number of images for each trajectory. In total n_runs trajectories.

everything is saved in the bacteria_growth folder.

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

## Data 
If you want to build growth trajectories for modeling:

Parse s_OSD-627.txt to get:

Spaceflight (Ground vs Space Flight)

Material

Time (1, 2, 3 days)

Group by (Spaceflight, Material, Medium)

Inside each group, sort by Time and then link the corresponding sample IDs:

e.g. (Space Flight, SS316, LB+KNO₃) → {1.*, 7.*, 13.*}

Then you have clean trajectories like:

“Spaceflight SS316, LB+KNO₃: Day 1 → Day 2 → Day 3 = {1., 7., 13.*}”



## pngs_max
multilayer .tif files are converted to .png files by taking the maximum intensity of each pixel from the layers.

## pngs_mean
multilayer .tif files are converted to .png files by taking the average intensity of each pixel across layers.


## pngs_max_split
images split into 256X256 from pngs_max

## pngs_mean_split
images split into 256X256 from pngs_mean
