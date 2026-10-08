# Fixed-theta, varying-xi converged figure

## Figure contract

- Core conclusion: hidden device-level heterogeneity mainly affects boundary tightness,
  while physical feasibility and downstream dispatch implementability remain robust.
- Archetype: compact quantitative grid.
- Backend: Python / Matplotlib only.
- Final size: 183 mm × 78 mm, two panels.
- Panel a: joint feasibility and optimality errors for 30 device realizations.
- Panel b: joint dispatch cost and relative disaggregation error for 450 dispatches.
- Statistics: all raw points are shown; dashed lines mark panel-a medians and the
  panel-b 2% disaggregation threshold.
- Exports: editable SVG, PDF, 600-dpi TIFF, PNG preview, and merged source-data CSV.

## Run

From any directory:

```powershell
python "画图代码R1\code_converged\same_theta_varying_xi_converged.py"
```

Outputs are written to `画图代码R1/figures/same_theta_converged/`.
