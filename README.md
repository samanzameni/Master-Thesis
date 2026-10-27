# Inference-Based Localization Attacks on Primary Users in Geolocation Database-Assisted Spectrum Sharing

**Saman Zameni, MohammadReza Taheri, Hadi Pouladi** · builds on M.Sc. thesis research at the University of Tehran
**Published:** S. Zameni, M. R. Taheri, H. Pouladi, *Ad Hoc Networks*, vol. 195, 104465, 2027. [doi:10.1016/j.adhoc.2026.104465](https://doi.org/10.1016/j.adhoc.2026.104465)

> [!IMPORTANT]
> **This repository holds a preliminary version of the simulation code.** The final code, configuration files and saved simulation results used in the published paper are archived on Zenodo: [doi:10.5281/zenodo.23060444](https://doi.org/10.5281/zenodo.23060444).

## Abstract

Geolocation database (GDB)-assisted spectrum sharing returns maximum-transmit-power (MTP) limits to secondary users (SUs) while protecting primary-user (PU) receivers from harmful interference. Because these limits depend on the position of the protected receiver, repeated responses can also carry spatial information. We formulate response-based PU localization as an inference problem and evaluate, in controlled simulations, a sequential particle filter (PF) and supervised coordinate regressors. Each MTP response is modelled as an interference-constrained ceiling with a conservative downward release perturbation and a matched likelihood, and posterior concentration is evaluated separately from localization error. Over 200 independent runs in a 500 m × 500 m region with 1,000 unconstrained random queries, the PF attains a final position root-mean-square error (RMSE) of 97.4 m (95% bootstrap interval 75.6–119.2 m), compared with 202.1 m for a motion-prior filter that ignores the responses. Its error varies non-monotonically, and simple same-information estimators reach comparable final-epoch error. A synthetic spatially varying propagation field, discrete power levels, query monitoring, limited query budgets and small multi-PU configurations each increase the error, in several cases to the level of the motion prior. None of the tested particle-concentration thresholds met the prespecified false-declaration criterion. On 20 held-out trajectories, the artificial neural network has the lowest coordinate mean squared error among the regressors, but all regressors have a larger trajectory RMSE than the PF on the same trajectories. Downward coarsening of the released power reduced retained power without increasing the attacker’s error. The conclusions are limited to the stated synthetic models and do not extend to operational GDBs.

**Keywords:** Cognitive radio networks, Primary user privacy, Geolocation database, Particle filter, Machine learning, Location inference attack

## Repository contents

| File | Purpose |
|---|---|
| `newmain.m` | Main script: simulates a mobile PU, SU queries to the GDB and the particle-filter attack; saves the datasets and figures |
| `DBreq.m` | GDB request model: returns the MTP, channel and time for an SU query |
| `h_continuous.m` | Measurement model mapping PU–SU distance to the expected MTP |
| `calculateChannel.m`, `calculateTime.m` | Channel and time helpers used by the GDB model |
| `compareML.m` | Trains and compares supervised regressors (neural network, SVM, regression tree, ensemble) for offline localization |
| `*.xlsx`, `*.mat` | Generated input/output datasets |
| `*.png` | Figures produced by the scripts |

## How to run

Requires MATLAB (R2021a or later) with the Statistics and Machine Learning Toolbox.

```matlab
newmain      % 1. particle-filter attack; writes inputData.mat, outputData1.mat, outputData2.mat and figures
compareML    % 2. machine-learning localization using the datasets from step 1
```

## Example outputs (preliminary code)

| Particle-filter RMSE | Regressor test MSE |
|---|---|
| ![RMSE of the particle filter](RMSE%20of%20Particle%20Filter.png) | ![Test MSE comparison](Test%20MSE%20Comparison%20(Line%20%26%20Bar).png) |

These figures come from this preliminary version and may differ from the figures in the paper.

## Contact

Saman Zameni · [Website](https://samanzameni.netlify.app) · [ORCID](https://orcid.org/0009-0006-8774-6391) · [saman.zameni@gmail.com](mailto:saman.zameni@gmail.com)
