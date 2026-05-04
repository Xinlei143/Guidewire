# SoftRobot

This repository currently contains a Torch-native stretching + bending soft-rod
simulation.  The active implementation is in `Stretching_bending/`.

The model is a 3D extensible, bendable, no-twist centerline rod.  It tracks only
the positions of rod centerline nodes.  Twist angles, material frames, contact,
friction, shear, and active control are outside the current model scope.

## Active Layout

```text
Stretching_bending/
  run_cantilever_demo.py      Command-line cantilever demo
  der_no_twist_model.py       Rest state, physical parameters, energies, forces
  semi_implicit_solver.py     Damped semi-implicit Euler dynamic relaxation
  soft_robot_common.py        Torch geometry and arc-length utilities
  visualization.py            Matplotlib plotting and animation helpers
  tests/
    test_der_geometry.py      Geometry checks
    test_der_energy_force.py  Energy and force consistency checks
    test_cantilever_demo.py   Solver smoke test
```

`Discrete elastic rod without twist.docx` is a project note/reference document.

Generated cache files under `__pycache__/` are not part of the model logic.

## Physical Model

The rod is discretized into `num_links` edges and `num_links + 1` nodes.  Each
node has a 3D position:

```text
x_i = [x, y, z]
```

For each edge,

```text
e_i = x_{i+1} - x_i
L_i = ||e_i||
t_i = e_i / L_i
```

### Stretching

Stretching energy is computed from the axial stiffness `EA`:

```text
E_stretch = 1/2 * sum_i (EA / L_i^0) * (L_i - L_i^0)^2
```

Here `L_i^0` is the rest length of edge `i`.  In code, each edge stiffness is
`EA / L_i^0`.

### Bending

Bending uses the discrete curvature-binormal at each interior node:

```text
kappa_b_i = 2 * (t_{i-1} x t_i) / (1 + t_{i-1} . t_i)
```

The bending energy compares the current curvature-binormal with the rest
curvature-binormal:

```text
E_bend = 1/2 * B * sum_i ||kappa_b_i - kappa_b_i^0||^2 / l_i^0
```

`B` is the bending stiffness and `l_i^0` is the Voronoi-style rest length around
the interior node.

### Forces

Elastic internal forces are computed by PyTorch autograd from the total elastic
energy:

```text
F_internal = -grad(E_stretch + E_bend)
```

External force, gravity, and damping are then added during time integration:

```text
F_total = F_internal + F_external + F_gravity - damping * velocity
```

Gravity acts in the negative z direction.

## Solver

The simulation uses damped dynamic relaxation with semi-implicit Euler:

```text
v_{n+1} = v_n + dt * F_total / m
x_{n+1} = x_n + dt * v_{n+1}
```

The demo fixes the first two nodes:

```python
fixed_indices = (0, 1)
```

This represents a clamped cantilever boundary: the root position and initial
root-edge direction are both constrained.  Fixing only node `0` gives a softer
boundary and will produce larger deflections.

The solver stops early if both the maximum free-node velocity and maximum
free-node residual force are below `--convergence-tol`.

## Run

Use the project conda environment:

```bash
cd Stretching_bending
conda run -n Env python run_cantilever_demo.py --no-plot
```

Run with plots and animation:

```bash
cd Stretching_bending
conda run -n Env python run_cantilever_demo.py
```

Save an animation GIF:

```bash
cd Stretching_bending
conda run -n Env python run_cantilever_demo.py --animation-output bending_stretching.gif
```

## Common Parameters

- `--num-links`: number of rod edges.  Default: `10`.
- `--total-length`: rest length of the full rod.  Default: `2.0`.
- `--EA`: axial stiffness.  Each edge uses stiffness `EA / L_i^0`.
- `--stretch-stiffness`: compatibility alias for `--EA`.
- `--bend-stiffness`: bending stiffness `B`.
- `--node-mass`: mass assigned to each free node.
- `--gravity`: gravitational acceleration in the negative z direction.
- `--force-node`: node receiving the user-specified external force.  Negative
  indices are supported.
- `--force-x`, `--force-y`, `--force-z`: external force components.
- `--dt`: integration time step.
- `--steps`: maximum number of time steps.
- `--damping`: nodal viscous damping coefficient.
- `--convergence-tol`: convergence tolerance for velocity and residual force.
- `--save-every`: save one history frame every N steps.
- `--no-plot`: disable interactive plots and GIF output.
- `--animation-output path.gif`: save the dynamic evolution as a GIF.

Example with a smaller load:

```bash
cd Stretching_bending
conda run -n Env python run_cantilever_demo.py --force-x 0 --force-y 0 --force-z -0.2 --no-plot
```

Example loading the tip node:

```bash
cd Stretching_bending
conda run -n Env python run_cantilever_demo.py --force-node -1 --force-z -0.2 --no-plot
```

## Tests

Run the unit tests from inside `Stretching_bending/`:

```bash
cd Stretching_bending
conda run -n Env python -m unittest discover tests
```

The current tests check:

- straight-rest geometry and arc-length helpers;
- zero rest energy for a straight rod;
- translation invariance of elastic energy;
- autograd force consistency against a finite-difference bending-energy check;
- a short dynamic-relaxation smoke test.

## Notes

- The active implementation is Torch-native.  NumPy conversion is kept at the
  visualization boundary.
- The model is no-twist: it is appropriate for centerline stretching and
  bending studies, but not for torsional mechanics.
- The modules currently use local absolute imports such as
  `from der_no_twist_model import ...`, so running commands from inside
  `Stretching_bending/` is the safest path.
