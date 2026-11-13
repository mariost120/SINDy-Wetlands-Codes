
# SINDy-Wetlands-Codes

Welcome to our Python repository for the manuscript **"Data-driven Discovery of Dynamical Systems for Treatment Wetlands"**, submitted to the *SIAM Journal on Life Sciences*.

This repository contains the experimental datasets and Python codes used to identify nonlinear dynamical systems for eight treatment wetlands using Sparse Identification of Nonlinear Dynamics (SINDy) with Sequentially Thresholded Least Squares with LASSO Regression (STLSLR).

## Repository contents

The repository contains two Python scripts and one directory with the experimental datasets.

### 1. SINDY-STLSLR Code.py

**Executable file for SINDy model identification and figure generation.**

- Implements the SINDy framework using STLSLR for sparse coefficient identification and hyperparameter selection.
- Produces **Figures 5–7** for W20P: hyperparameter assessment, forward-prediction assessment, and synthetic-perturbation assessment.
- Produces the individual plots used in **Figures 9 and 10** for the remaining seven systems: hyperparameter assessment and number of active coefficients.
- Exports normalized and dimensional coefficient matrices and forward-prediction metrics.

**Input and parameters:**

At the beginning of the script, the user can select the wetland system and initialize the main parameters:

- `CSV_FILE`: Specifies the experimental dataset to analyse (e.g., `W20P.csv`).
- `NUM_POINTS`: Specifies the interpolation resolution \(M\) used to construct the SINDy model.
- `ALPHA_SET` and `DELTA_SET`: Define the regularization and thresholding parameters used by STLSLR.
- Additional settings control the training-data split, maximum STLSLR iterations, and figure generation.

To reproduce the results for a different wetland system, update `CSV_FILE` and `NUM_POINTS` accordingly. The remaining parameters are already defined in the script.

### 2. M Sensitivity Analysis.py

**Executable file for interpolation-resolution selection.**

- Evaluates the combined objective function \(f(M)\) using normalized forward-prediction RMSE and the fraction of active coefficients.
- Uses precomputed results from SINDy models identified at different interpolation resolutions.
- Produces **Figure 8**, showing the objective function and selected \(M\) for all eight wetland systems.

### 3. Wetlands Data

Contains eight experimental CSV datasets:

`W20P.csv`, `W20U.csv`, `C20P.csv`, `C20U.csv`, `W5P.csv`, `W5U.csv`, `C5P.csv`, and `C5U.csv`.

Each dataset contains time-series measurements of soluble chemical oxygen demand (CODs), ammonia (NH3), nitrate (NO3), phosphate (PO4), dissolved oxygen (DO), oxidation-reduction potential (ORP), and pH.

## Running the codes

1. Download the repository and install the required Python packages.
2. Open `SINDY-STLSLR Code.py` and select the desired experimental dataset using `CSV_FILE`.
3. Set `NUM_POINTS` to the corresponding interpolation resolution and run the script to obtain the SINDy model, figures, and numerical results.
4. Run `M Sensitivity Analysis.py` separately to reproduce Figure 8.

The experimental datasets are stored in the `Wetlands Data` directory. Set `CSV_FILE` to the corresponding file path (e.g., `Wetlands Data/W20P.csv`).

## Python requirements

The codes require Python with the following packages:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- CVXPY (with OSQP, ECOS, or SCS solvers)
