# SoftRobot Guidewire Simulation Model Catalog

This CPU code package is intended for coursework and research review. The 19 independent models are listed by physical task and version in Section 3. Version v4 covers only some of the straight- and curved-tube implementations; it does not replace the earlier models. The code directories correspond to the tasks:

```text
├── src/softrobot/
│   ├── foundations/       Basic rods: stretching, bending, implicit Newton, cantilever
│   ├── contact/           Basic PI soft contact: dynamic and quasi-static
│   ├── straight_tube/     Straight tubes: 2-D, 3-D v2/v3, mesh, current v4
│   ├── curved_tube/       Curved tubes: 2-D, legacy 3-D analytic/mesh, current v4
│   ├── common/            Centerlines and shared command-line tools
│   └── backends/          C++ extensions, mesh contact, and Python/C++ interfaces
└── assets/geometry/        Canonical OBJ/CSV geometry required to run (repository root)
```

The instructor ZIP excludes reports, raw experimental data, historical outputs, media, caches, third-party source code, and the sinusoidal and free-insertion research lines. The current merged research workspace retains the latter two (see Section 6). See `docs/model_lineage.md` for the complete provenance and content hashes of the instructor-package models.

## 1. CPU Environment and Installation

The minimum acceptance baseline is Linux, Python 3.10, CPU PyTorch, NumPy, SciPy, Matplotlib, Pillow, pytest, setuptools, and g++/Ninja. CUDA is neither a dependency nor an acceptance criterion. See `requirements-cpu.lock` and `docs/environment.md` for the exact tested versions and commands for a new environment.

The interpreter used for this acceptance run was Python 3.10.20, with `torch.__version__` equal to `2.11.0+cu130` (a CUDA-enabled distribution). All actual model runs and smoke tests explicitly used the CPU; no CUDA cases were run or accepted. In the full pytest suite, `torch.cuda.is_available()` was used only as an availability gate for CUDA-only tests; skips resulting from this gate do not indicate a CPU functionality defect. A CPU-only wheel was not separately tested for acceptance.

If the dependencies are already installed, install offline with:

```bash
python -m pip install --no-deps --no-build-isolation .
softrobot models
```

On the first run, PyTorch's extension mechanism invokes Ninja/g++ to build the C++ backend and writes the compilation cache to a temporary directory. Build artifacts are not written or committed to the source tree.

## 2. Unified Run Interface

```bash
# List models
softrobot models

# CPU smoke tests for individual models across tasks and versions;
# output is written only to the specified directory
softrobot run foundation-2d-stretch-numpy --smoke-test --output-dir /tmp/foundation-smoke
softrobot run contact-2d-quasistatic-pi-v1 --smoke-test --output-dir /tmp/contact-v1-smoke
softrobot run straight-tube-v2-3d --smoke-test --output-dir /tmp/straight-v2-smoke
softrobot run curved-tube-3d-legacy-mesh --smoke-test --output-dir /tmp/curved-legacy-smoke

# Run smoke tests for all 19 models
softrobot smoke --all --output-dir /tmp/softrobot-smoke
```

Each model can also be run independently as a module. The examples below cover a basic rod, contact, a straight tube, and a curved tube; see Section 3 for all module entry points:

```bash
python -m softrobot.foundations.implicit_newton_3d --smoke-test --output-dir /tmp/newton3d
python -m softrobot.contact.quasistatic_pi_v2 --smoke-test --output-dir /tmp/contact-v2
python -m softrobot.straight_tube.v3 --smoke-test --output-dir /tmp/straight-v3
python -m softrobot.curved_tube.arc_2d --smoke-test --output-dir /tmp/arc2d
```

Each Model ID in the registry has its own module entry point; different contact configurations that share an underlying solver also have separate entry points. Without `--smoke-test`, the models retain their original default parameters and case semantics. Only smoke mode reduces the number of nodes, steps, and output size, and fixes execution to the CPU, the headless Matplotlib Agg backend, and a deterministic random seed.

## 3. Models by Task and Version

Each row below is an independently runnable Model ID. Use `softrobot run <Model ID>` as the unified entry point, or append the module entry point to `python -m`. “Legacy” denotes a retained independent physical or numerical implementation; “current” denotes a research implementation current at the time of organization. **It does not mean that legacy models have been superseded.** Smoke tests check only small-parameter CPU execution end to end; they do not establish physical accuracy. See `docs/model_lineage.md` for provenance, content hashes, and more detailed validation evidence.

### 3.1 Basic Rod Models (5)

| Model ID | Version/branch | Dimensions | Task and contact | Backend | Module entry point (`python -m`) | CPU smoke |
|---|---|---:|---|---|---|---|
| `foundation-2d-stretch-numpy` | Legacy: stretching | 2-D | Rod stretching without contact | NumPy | `softrobot.foundations.stretch_2d_numpy` | Covered |
| `foundation-2d-stretch-bend-torch` | Legacy: stretching + bending | 2-D | Rod elasticity without contact | Torch | `softrobot.foundations.stretch_bend_2d_torch` | Covered |
| `foundation-3d-stretch-bend-torch` | Legacy: 3-D elasticity | 3-D | Rod stretching + bending without contact | Torch | `softrobot.foundations.stretch_bend_3d_torch` | Covered |
| `foundation-3d-implicit-newton` | Legacy: implicit integration | 3-D | Implicit Euler/Newton rod without contact | Torch | `softrobot.foundations.implicit_newton_3d` | Covered |
| `foundation-cantilever-der` | Legacy: final cantilever | 3-D | Fixed-end rod under point load, without contact | Torch | `softrobot.foundations.cantilever.run_cantilever_demo` | Covered |

### 3.2 Basic PI Contact (3)

| Model ID | Version/branch | Dimensions | Task and contact | Backend | Module entry point (`python -m`) | CPU smoke |
|---|---|---:|---|---|---|---|
| `contact-2d-dynamic-pi` | Legacy: dynamic | 2-D | Parallel boundaries / PI soft contact | Torch | `softrobot.contact.dynamic_pi_2d` | Covered |
| `contact-2d-quasistatic-pi-v1` | Legacy: quasi-static v1 | 2-D | Parallel boundaries / PI framework, integral force term disabled | Torch | `softrobot.contact.quasistatic_pi_v1` | Covered |
| `contact-2d-quasistatic-pi-v2` | Legacy: quasi-static v2 | 2-D | Parallel boundaries / PI soft contact, integral force term enabled | Torch | `softrobot.contact.quasistatic_pi_v2` | Covered |

### 3.3 Straight-Tube Tasks (5)

| Model ID | Version/branch | Dimensions | Task and contact | Backend | Module entry point (`python -m`) | CPU smoke |
|---|---|---:|---|---|---|---|
| `straight-tube-2d-pi` | Legacy: 2-D | 2-D | Straight tube / PI soft wall | Torch | `softrobot.straight_tube.legacy_pi_2d` | Covered |
| `straight-tube-v2-3d` | Legacy: v2 | 3-D | Straight tube / PI soft wall, quasi-static | Torch | `softrobot.straight_tube.v2` | Covered |
| `straight-tube-v3` | Current: v3 | 3-D | Straight tube / PI node contact | Torch | `softrobot.straight_tube.v3` | Covered |
| `straight-tube-mesh-3d` | Legacy: independent mesh branch | 3-D | Straight tube / triangle mesh contact | Torch | `softrobot.straight_tube.quasi_static_mesh_3d` | Covered |
| `straight-tube-v4` | Current: v4 | 3-D | Straight tube / analytic tube or mesh contact | Python + C++ | `softrobot.straight_tube.v4` | 4 configurations |

Normal operation of `straight-tube-v4` also supports continuation loading, pre-equilibration, an analytic straight tube, and OBJ meshes. The four CPU smoke configurations run were Python/C++ × analytic straight tube/straight-tube mesh. It is a different implementation from the legacy `straight-tube-mesh-3d`, which retains its own entry point. See `python -m softrobot.straight_tube.v4 --help` for parameters.

### 3.4 Curved-Tube Tasks (6)

| Model ID | Version/branch | Dimensions | Task and contact | Backend | Module entry point (`python -m`) | CPU smoke |
|---|---|---:|---|---|---|---|
| `curved-tube-2d-legacy-python` | Legacy: 2-D Python | 2-D | Circular-arc tube / PI soft wall | Torch | `softrobot.curved_tube.legacy_arc_2d` | Covered |
| `curved-tube-2d-arc` | Current: accelerated 2-D branch | 2-D | Circular-arc tube / PI soft wall | Python + C++ | `softrobot.curved_tube.arc_2d` | 2 backends |
| `curved-tube-3d-legacy-analytic` | Legacy: 3-D analytic branch | 3-D | Circular-arc tube / analytic contact | Torch | `softrobot.curved_tube.legacy_arc_3d_analytic` | Covered |
| `curved-tube-3d-legacy-mesh` | Legacy: 3-D mesh branch | 3-D | Circular-arc tube / triangle mesh contact | Torch | `softrobot.curved_tube.legacy_arc_3d_mesh` | Covered |
| `curved-tube-v4-analytic` | Current: v4 analytic configuration | 3-D | Curved-centerline tube / analytic contact | Python + C++ | `softrobot.curved_tube.v4_analytic` | 2 backends |
| `curved-tube-v4-mesh` | Current: v4 mesh configuration | 3-D | Curved tube / triangle mesh contact | Python + C++ | `softrobot.curved_tube.v4_mesh` | 2 backends |

The analytic and mesh entry points for the legacy 3-D circular-arc tube and the current v4 curved tube each preset their contact configuration. Configurations sharing an underlying implementation do not duplicate the source code. By default, `curved-tube-v4-mesh` pairs `arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj` with `arc90_L1p5m_xz_centerline_800.csv`. The CSV columns are `x,y,z`, with `y=0`; the curve lies in the xz plane. Explicitly supplied long options with the same names can override the preset assets or contact configuration.

## 4. Migration from Legacy Paths

The following seven root-level entry points remain as thin compatibility scripts; their implementations reside under `src/softrobot/`:

| Legacy root path | New module |
|---|---|
| `centerline_3d.py` | `softrobot.common.centerline_3d` |
| `der_cpp_backend.py` | `softrobot.backends.der_cpp_backend` |
| `mesh_contact_v4.py` | `softrobot.backends.mesh_contact_v4` |
| `quasi_static_mesh_3d.py` | `softrobot.straight_tube.quasi_static_mesh_3d` |
| `rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py` | `softrobot.curved_tube.arc_2d` |
| `rod_soft_contact_PI_dynamic_quasi_static_3d_v3.py` | `softrobot.straight_tube.v3` |
| `rod_soft_contact_PI_dynamic_quasi_static_3d_v4.py` | `softrobot.straight_tube.v4` |

These seven entry points are thin compatibility scripts in the instructor package. The current research workspace also retains root-level research entry points for sinusoidal and free insertion; they are not among the instructor package's 19 Model IDs and are not included in the delivered ZIP. Historical directory names are used only for provenance checks in `docs/model_lineage.md`; they are not runtime dependencies of the instructor package.

## 5. Output and Tests

The unified CLI's `--output-dir` specifies the output directory for each run. Smoke tests write strict JSON results and check finite states, completed steps/convergence, evidence of actual contact where applicable, and nonzero state increments. Test artifacts should be placed in `/tmp` or another directory outside the repository.

```bash
python -m pytest -q
python -m compileall -q src
softrobot smoke --all --output-dir /tmp/softrobot-all-smoke
```

Smoke tests are not physical validation and cannot support claims about accuracy, stability, performance, hardware safety, or clinical effectiveness. They show only that small-parameter end-to-end paths complete and produce structurally valid results in the stated CPU environment. Formal results still require independent mesh and time-step convergence studies, benchmark comparisons, and task-level validation.

## 6. Follow-on Experiments Retained in the Current Research Workspace

The merged working branch also retains the sinusoidal-tube insertion entry point `rod_soft_contact_PI_dynamic_quasi_static_2d_sine_insert.py`, its design and evidence document `docs/2D正弦三连续弯导丝插入与摩擦Benchmark设计.md`, and the straight-tube free-insertion entry point `rod_soft_contact_PI_dynamic_quasi_static_3d_straight_free_insert.py`. C++ source code for these research lines is stored with the shared backend in `src/softrobot/backends/cpp/`; the root-level `der_cpp_backend.py` remains a compatibility import. These additions do not increase the instructor package's Model ID count. The 360 mm parameters and visualization changes for free insertion reflect the local research state in backup commit `5ac40fb`; they should not be interpreted as completed validation for formal operating conditions.

`wire_report/` and `血管导丝/` are locally retained, Git-ignored materials directories. They are not included in the remote backup or instructor ZIP.
