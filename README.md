# PINN: Physics-Informed Polytope Approximation for DER Aggregation

This repository contains research code for learning parametric polytope approximations of the aggregate flexibility of distributed energy resources (DERs). It combines PyTorch neural networks with Pyomo optimization models to construct and evaluate reduced representations of aggregate feasible power trajectories.

The main case study aggregates electric vehicles (EVs), thermostatically controlled loads (TCLs), and photovoltaic (PV) units over a 24-hour horizon. Additional scripts investigate energy storage, alternative resource configurations, parameter generalization, supervised learning baselines, and price-guided dispatch.

Repository: [WangZiyue1009/PINN](https://github.com/WangZiyue1009/PINN)

## Method Overview

The aggregate feasible region is approximated by a polytope of the form

```text
P(theta) = { x : A(theta) x <= b(theta) },
```

where `x` is the normalized aggregate power trajectory and `theta` describes the resource configuration. The main case uses three input features: EV count, average ambient temperature, and PV energy. Variant case files introduce other parameterizations.

The implementation provides three model classes:

- **PreTrainNet** learns a fixed pair of polytope coefficients `A` and `b` for a reference case.
- **FullNet** predicts both `A(theta)` and `b(theta)` using separate neural network branches. Each branch uses a hidden layer with ReLU activation; the default hidden width is 64.

`PreTrainNet` and `FullNet` normalize the constraint rows and their corresponding offsets. The initial constraint matrix in the main case combines upper and lower power bounds with cumulative-energy constraints, giving `4T` inequalities for a horizon of `T` time steps.

Physics-informed training uses optimization-based feasibility and optimality calculations against the detailed resource model. The trainer combines the two residual losses as

```text
L = L_feas + lambda * L_opt,
```

with `lambda` configured through `rate_opt_feas`. Boundary-sampling effort is controlled by `n_cal`. The repository also includes a supervised baseline trained on saved polytope coefficients.

## Repository Structure

```text
PINN/
└── pinn/
    ├── Simulator/
    │   ├── __init__.py            # Defines PROJECT_ROOT as the pinn/ directory
    │   ├── Approximator.py        # Networks, error calculations, losses, trainer
    │   ├── Counter.py             # Geometric comparison utilities
    │   ├── Plotter.py             # Visualization utilities
    │   ├── cases/                 # Resource models and scenario sampling
    │   ├── runners/               # Training and selected evaluation scripts
    │   ├── test_experiments/      # Evaluation, dispatch, and ablation scripts
    │   └── pictures/
    │       ├── code_converged/    # Publication figure scripts
    │       ├── data/              # Figure source tables and training histories
    │       └── figures_converged/ # Included PDF figures and source exports
    ├── data/
    │   ├── aggregator_data/       # Building parameters
    │   └── profiles_data/         # Temperature, PV, and other profile data
    │   └── price_data/            # Electricity-price scenarios
    └── results/
        └── aggregation/           # Checkpoints, histories, and evaluation tables
```

## Requirements

The source imports the following main third-party packages:

- PyTorch
- NumPy
- SciPy
- pandas
- Pyomo
- Matplotlib
- Pillow

Optimization-based training and evaluation use **Gurobi**, including solver calls that explicitly select `gurobi`. A working Gurobi installation and license are required for these workflows. Installing its Python package alone does not verify solver availability through Pyomo.

Several scripts use f-string syntax supported by Python 3.12 and later. Use **Python 3.12 or later** as a starting environment, subject to the Python versions supported by the selected dependency releases. The repository does not include a pinned dependency specification or a documented, validated environment.

The training and evaluation scripts generally select CUDA when available and otherwise use the CPU. CSV-based figure generation can be performed separately from model training and solver-based evaluation.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/WangZiyue1009/PINN.git
cd PINN
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the imported dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pandas pyomo matplotlib pillow gurobipy
```

For GPU execution, select a PyTorch build compatible with the local CUDA environment. Configure Gurobi separately and check the interface used by the code:

```bash
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available(exception_flag=False))"
```

The check should print `True`; a successful optimization solve is also needed to confirm license operation.

## Running the Code

Run the following commands from the **`pinn/` directory** so that Python can import the `Simulator` package:

```bash
cd pinn
```

The commands below identify the relevant entry points. Review their configuration and the path adjustments described below before running them. Training and optimization-based evaluation have not been verified end to end for this README.

### 1. Evaluate Included Checkpoints

```bash
python -m Simulator.test_experiments.1
```

`Simulator/test_experiments/1.py` evaluates the main three-feature model. Its current configuration uses 40 test cases, 80 TCLs, 20 PV units, and the interpolation test mode, with an EV training range of 60–100. It loads these included checkpoints:

```text
results/aggregation/pretrainnet_weights_20260504_161742.pth
results/aggregation/fullnet_weights_20260808_235920.pth
```

The script evaluates feasibility and optimality errors using 10 sampling directions and reports statistics. It also contains CSV export. Other evaluation scripts include `same_test.py` for varying configurations at fixed input features, `SNET.py` for the supervised baseline, and `pinn_5.py` / `pinn_7.py` for parameter variants, `Key Parameters Plot.py` for price-guided sensitivity tests on the paper's key parameters, and `lambda0.py`/`lambda0.py`/`lambda_down.py`/`no_feas.py`/`no_opt.py`/`SNET.py`/`3.py` /`5.py`/`7.py`  for ablation experiments on the proposed method.

For a separate extrapolation experiment:

```bash
python -m Simulator.runners.test_extrapolation
```

Review that script's checkpoint selection and out-of-distribution configuration before execution.

### 2. Train the Main Parametric Model

```bash
python -m Simulator.runners.main_agg
```

The active configuration in `main_agg.py` samples 150 resource configurations with 60–100 EVs, 80 TCLs, 20 PV units, `T=24`, and `discrete_rate=0.0`. It initializes `FullNet` from a pretrained reference polytope, trains for 6,000 iterations with `rate_opt_feas=1.0`, and saves model weights and training history.


Reference-case pretraining code is present as commented sections in the training runners. Training a new reference model requires explicitly configuring the reference `Aggregator` case, selecting `model_type='pretrainnet'`, enabling its training block, and then using the resulting checkpoint in the parametric training script.

Related runners include `main_small.py`, `main_add5.py`, `main_add7.py`, `no_pre.py`, `sl.py`,`DER1.py` and `DER2.py`. Their settings implement different experiments; inspect the active values in each file rather than treating all runners as equivalent training commands.

### 3. Train the Supervised Baseline

```bash
python -m Simulator.runners.supervised_train
```

The supervised runner reads saved `polytope_model_*.json` coefficients from `results/aggregation/history/Ab/`. Its current configuration specifies 150 samples, a hidden width of 64, a batch size of 32, and 36,000 epochs.
 `runners/sl.py` contains code for generating individual training polytopes, but its output path and training iteration setting also require review.

### 4. Run Price-Guided Dispatch Comparisons

```bash
python "Simulator/test_experiments/Comparison Results.py"
```

This script compares the learned polytope, supervised baseline, axis-aligned box (`cube`), and power–energy approximation (`power_energy`) against the detailed optimization model. It includes dispatch-cost and disaggregation calculations.

Update `PLOT_CONFIG['real_price_path']` to the local price CSV before running. The included `data1/Random_Price.csv` has columns `scenario`, `hour`, and `total_lmp_da`. Confirm the three selected checkpoint names and any export paths in the script.

### 5. Regenerate Figures from Included Data

Figure generation offers a separate workflow using the included result tables:

```bash
python Simulator/pictures/code_converged/plot_test_error_cloud.py
python Simulator/pictures/code_converged/DER_converged.py
python Simulator/pictures/code_converged/ablation_joint_error_2x2.py
python Simulator/pictures/code_converged/price_comparison_with_snet_converged.py
python Simulator/pictures/code_converged/same_theta_varying_xi_grouped_distributions.py
```

These scripts read source tables from `Simulator/pictures/data/` and write PDF figures under `Simulator/pictures/figures_converged/`. They cover test errors, resource comparisons, ablations, dispatch performance, and fixed-feature configuration variability.

## Data and Saved Results

| Location | Contents |
| --- | --- |
| `data/aggregator_data/ZH_buildings.csv` | Building parameters used by the TCL model |
| `data/profiles_data/profiles_data.npz` | Profile archive, including ambient-temperature data |
| `data/profiles_data/PV_samples.mat` | PV profiles used by the main case |
| `data/price_data`| Typical double-peak, random, extreme-low, and extreme-high price CSVs |
| `results/aggregation/*.pth` | Reference, parametric, and supervised model checkpoints |
| `results/aggregation/history/` | Training histories and saved polytope coefficients |
| `results/aggregation/figures_pictures/` | Evaluation and dispatch result tables |
| `Simulator/pictures/data/` | Source data used by publication figure scripts |

Checkpoint files correspond to different experiments. Match the model class, input-feature count, horizon, and case implementation before loading a checkpoint. Updating a training checkpoint also requires updating the filenames selected by evaluation scripts.

## Reproducibility Notes

- **Paths:** Several case files build paths with Windows backslashes, and some experiment scripts contain absolute paths such as `D:\important\share\...`. Update these paths for the local installation. Linux/macOS execution requires adapting Windows-specific path construction, for example with `pathlib.Path`.
- **Configuration:** Experiment parameters are embedded in the scripts. Record the selected case file, checkpoint, seeds, resource counts, horizon, sampling settings, loss weights, and training schedule for each run.
- **Outputs:** Scripts may create or replace checkpoints, CSV files, histories, and figures. Use a separate checkout or output directory when preserving the supplied results.
- **Evaluation:** Report feasibility and optimality errors using the same tolerances and sampling settings. Dispatch comparisons also depend on price scenarios and disaggregation tolerances.
- **Environment:** Record the Python, PyTorch, Pyomo, Gurobi, CUDA, and other dependency versions used for a reproduced experiment. A validated environment lockfile is not included in the current repository.
