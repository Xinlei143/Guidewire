# SoftRobot

This project keeps each physical modeling stage as a separate runnable script.
The stretching-only script is the baseline and should stay frozen after it is
accepted.

## Scripts

```text
soft_robot_chain_stretching.py
```

3D stretching-only chain. This is the baseline model.

```bash
conda run -n Env python soft_robot_chain_stretching.py --no-plot
```

```text
soft_robot_chain_bending_stretching.py
```

3D stretching + bending chain. It keeps the same nodal centerline structure as
the baseline, adds 3D centerline bending energy, and uses explicit dynamics.

```bash
conda run -n Env python soft_robot_chain_bending_stretching.py --no-plot
conda run -n Env python soft_robot_chain_bending_stretching.py --bend-stiffness 0.05 --force-z 0.1
```

```text
soft_robot_chain_bending_stretching_torsion.py
```

3D stretching + bending + torsion chain. Centerline positions are 3D, and
torsion is represented with one scalar twist angle per edge.

```bash
conda run -n Env python soft_robot_chain_bending_stretching_torsion.py --no-plot
conda run -n Env python soft_robot_chain_bending_stretching_torsion.py --initial-tip-twist 0.5 --twist-stiffness 0.02
```

```text
soft_robot_common.py
```

Shared low-level utilities used by the stage scripts, such as edge geometry,
stretching force/energy, simple potentials, arc-length helpers, and lazy
Matplotlib loading. The three model scripts remain the main reading and
execution entry points.

## Parameters

- `--stretch-stiffness`: axial stretching stiffness for all stage scripts.
- `--bend-stiffness`: bending stiffness for the second and third stage scripts.
- `--twist-stiffness`: torsional stiffness for the third stage script.
- `--force-x`, `--force-y`, `--force-z`: external force components. All stage
  scripts now use a 3D centerline.
- `--gravity`: gravitational acceleration. Gravity acts in the negative z
  direction.
- `--no-plot`: run without opening interactive plot windows.
- `--plot-history`: show energy and convergence histories.
- `--animate` or `--animation-output path.gif`: show or save the saved states.

## Implementation Notes

- The current bending force is computed as the negative central finite
  difference gradient of the 3D centerline bending energy. This is slower than
  an analytic force formula but easier to verify while developing the model.
- The torsion stage uses a minimal twist-angle model. It is a practical third
  stage scaffold, not a complete material-frame DER implementation yet.
- Scripts are intentionally independent so results from one physical stage can
  be preserved and compared against later stages.
- Python dependencies are expected to come from the `Env` conda environment.
