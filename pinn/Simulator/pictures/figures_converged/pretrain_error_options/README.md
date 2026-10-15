# Pretraining error phase portrait

The output contains a single joint feasibility-optimality phase portrait.

- Data source: the top-level feas_error and opt_error arrays.
- Feasibility-error axis: logarithmic, 10^-2 to 10^0.
- Optimality-error axis: linear, 0 to 0.12.
- Start marker: raw iteration 0.
- End marker: mean of the final 51 iterations (0.0238716, 0.0546558).

Data summary

- Iterations: 5,001
- Smoothing: start-preserving causal moving mean (up to 51 iterations).

Export: PDF only.
