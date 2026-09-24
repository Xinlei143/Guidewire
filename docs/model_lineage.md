# Model lineage

This inventory records source provenance and classification only. It does not assign authorship or claim validation beyond the listed smoke status.

## Foundations

| Model ID | Classified source | Provenance | Deduplication / status |
|---|---|---|---|
| `foundation-2d-stretch-numpy` | `血管导丝/00_导丝受力变化物理模型/rod_simulation_2d_stretch.py` plus `rodtoolbox_2d_stretch.py` | SHA256 `b025273cf4ec0f5a2293bf291e186558bdcb8fd9abed5451c6c7855b6d0218c3`; toolbox `52e17a49ba998b5c72641d6d1ee4ef1fdb3b4a5d420da9a2e20295ae509e48c5` | NumPy stretching lineage; retained with toolbox; CPU smoke supported |
| `foundation-2d-stretch-bend-torch` | `血管导丝/00_导丝受力变化物理模型/rod_stretch_bend_2d_pytorch.py` plus `toolbox_ks_kb_pytorch_2d.py` | SHA256 `90dba8f8c29cbcdcf71b593e72215f6a534e372025d0f86ab104eecea7f16cde`; toolbox `0984944bf07cca6f9a12119c8c05aeabc64c083b56a818bbced2c1d27f9c8351` | Distinct 2-D stretch-bend/autograd model; CPU smoke supported |
| `foundation-3d-stretch-bend-torch` | `血管导丝/00_导丝受力变化物理模型/rodsim_str_bend_pytorch3d.py` plus `rodtoolbox_ks_kb_torch3d.py` | SHA256 `b1319cccaf858c5301e8f9d9243037138edf5f2de14adf516a8d743517e29380`; toolbox `edbbe36e535798433879f72c9eaefca871daa057274fc36aaa769958de73ab15` | Toolbox reused by implicit model rather than duplicated; CPU smoke supported |
| `foundation-3d-implicit-newton` | `血管导丝/00_导丝受力变化物理模型/rodsim_implicit_euler_newton3d.py` | SHA256 `16a219776723eb0c46c43552af5464bf0478b691412d509f186cbe8ede650050` | Distinct implicit-Euler/Newton integrator; CPU smoke supported |
| `foundation-cantilever-der` | attachment `SoftRobot.zip`, tracked `Stretching_bending/` | embedded repository commit `eb33bf3`, tree `7e2355872dcd889ec5d5822db15f5265aa1442b3` | Uses tracked modular final plus EB validation. The attachment working-tree force-default edit was excluded; CPU smoke supported |

`proof_analysis.py` (source SHA256 `bdced1a03b34c342ba2c2a8974a9c8da1ec8df857de82371ed740568878f3bf4`) is retained under `softrobot.foundations` as a supporting analytical tool and is intentionally not registered as a model.

## Contact

| Model ID | Classified source | Provenance | Deduplication / status |
|---|---|---|---|
| `contact-2d-dynamic-pi` | `血管导丝/01_基础施力模型/rod_soft_contact_PI_dynamic.py` | SHA256 `578735e5da80fc9c2d6f8ccc54c5e0ba75b1ba507b02495c65f091a7d46ddd84` | Dynamic implicit contact model; CPU smoke supported |
| `contact-2d-quasistatic-pi-v1` | attachment root `rod_soft_contact_PI_dynamic_quasi_static.py` | untracked attachment file SHA256 `1a0b234ef0693053d2296a99e5b023888c31fe6f913d099f6b551610a40a894b` | Retained separately: pure-proportional `KI=0` stability-check branch and distinct solver/tolerance logic; CPU smoke supported |
| `contact-2d-quasistatic-pi-v2` | attachment root `rod_soft_contact_PI_dynamic_quasi_static_v2.py` | untracked attachment file SHA256 `baa0c14e3b9b12bd93b14b0def26c03006b494b4eed61f5f69cb5398945d76cf` | Retained separately: `KI=400` and distinct Newton/contact tolerance logic, so not content-duplicate of v1; CPU smoke supported |

Excluded from import: attachment `.git`, caches, `.DS_Store`, DOCX, MATLAB, NPZ, media, and all generated outputs.

## Straight and curved tube lineage

All source hashes below are SHA-256 of the classified file before package-relative import and UTF-8 BOM normalization. The classified historical paths are untracked snapshots in the source workspace, so no source commit exists; the full content hash is the authoritative provenance. Current packaged models are anchored to base commit `0b473ae21073e53430d84ef6b41780e63684a9fb`. The stage-3 import excludes repository copies, `third_party`, caches, reports, media, ZIP/NPZ files, and generated `output`/`outputs` trees.

| Model ID | Source path and provenance | Task / dimension / geometry / contact / backend | Deduplication / smoke status |
|---|---|---|---|
| `straight-tube-2d-pi` | `血管导丝/02_直圆管模型/二维直圆管/rodsim_2d_vessel_boundary_PI.py`; SHA256 `f57951b40dabcad686ac2e6471be0236f0c0858afe9fb9c8ad59e8d3f55ea7f6` | guidewire / 2-D / straight / PI soft wall / Python-Torch CPU | Distinct historical implicit-Euler/Newton node-contact solver; one real Newton increment in smoke |
| `straight-tube-v2-3d` | `血管导丝/02_直圆管模型/三维直圆管/rod_soft_contact_PI_dynamic_quasi_static_3d_v2.py`; SHA256 `df2814a3b33d3b8ee1062c7e30a7eaa37261c51f397d7bf112282f603f035cd7` | guidewire / 3-D / straight / PI soft wall / Python-Torch CPU | Distinct historical v2 quasi-static solver; not content-identical to packaged v3; real Newton smoke |
| `straight-tube-v3` | stage-1 packaged current source; stage-3 pre-edit SHA256 `990b59784249a64f7caed207e8ba3e4726579656808b617ec680f7a7f8c0ff51` | guidewire / 3-D / straight / PI soft wall / Python-Torch CPU | Current v3 node-contact branch; real CPU Newton smoke |
| `straight-tube-mesh-3d` | stage-1 packaged `quasi_static_mesh_3d.py`; classified source under `血管导丝/02_直圆管模型/OBJ圆管网格/` | guidewire / 3-D / straight / triangle mesh / Python-Torch CPU | Separate historical mesh-contact branch; real energy/contact/Newton smoke |
| `straight-tube-v4` | stage-1 packaged current source; stage-3 pre-edit SHA256 `8f03681ea8c59e5847eeb18ab352cfff5492a2a7169170a60447bc68b5db5748` | guidewire / 3-D / straight / analytic tube or mesh / Python and C++ CPU | Three executed variants: `python+tube`, `cpp+tube`, `cpp+straight-mesh`; Python-mesh is not repeated because the independent `straight-tube-mesh-3d` node covers that lineage |
| `curved-tube-2d-legacy-python` | `血管导丝/03_arc_tube_model/二维弧形管/rod_soft_contact_PI_dynamic_quasi_static_2d_arc.py`; SHA256 `ebb793d0e7eba0e3968ff92507939bb722c1bcf50f1411b8267e8b29946320d5` | guidewire / 2-D / circular arc / PI soft wall / Python-Torch CPU | Retained: differs from accelerated candidate SHA256 `6ba191453d63c073534875b6baf66fa1f7b83465dd115580bb444d7efd1656dc` and current packaged pre-edit SHA256 `17a16515587789d2d053992a027081c670fefa53273432fa75ca59baa6632814` |
| `curved-tube-2d-arc` | stage-1 packaged current source; pre-edit SHA256 `17a16515587789d2d053992a027081c670fefa53273432fa75ca59baa6632814` | guidewire / 2-D / circular arc / PI soft wall / Python and C++ CPU | Current accelerated model; real `python` and `cpp` Newton variants |
| `curved-tube-3d-legacy-analytic` | `血管导丝/03_arc_tube_model/三维弧形管/rod_soft_contact_PI_dynamic_quasi_static_3d_arc.py`; SHA256 `c1a0a7109e9ce0cb858766cd1d1964c5b140bb4106f8a084c458c1245efd7e6b` | guidewire / 3-D / circular arc / analytic open tube / Python-Torch CPU | Registry preset `--contact-type analytic`; shares one source module with the mesh ID rather than duplicating code |
| `curved-tube-3d-legacy-mesh` | same legacy 3-D arc source plus `arc_mesh_contact.py`; helper SHA256 `6fa64a91c5a5e2aadbf61b39770d48a83883905b8897f056a45f6da0ca9b1e10` | guidewire / 3-D / circular arc / triangle mesh / Python-Torch CPU | Registry preset `--contact-type mesh`; distinct physical contact task, shared solver source; real mesh-contact energy/Newton smoke |
| `curved-tube-v4-analytic` | current v4 solver at repository commit `0b473ae21073e53430d84ef6b41780e63684a9fb`; packaged arc centerline resource | guidewire / 3-D / curved centerline / analytic tube / Python and C++ CPU | Physical-task alias/preset of v4; real `python` and `cpp` Newton variants |
| `curved-tube-v4-mesh` | current v4 solver at repository commit `0b473ae21073e53430d84ef6b41780e63684a9fb`; packaged minimal canonical arc OBJ | guidewire / 3-D / curved / triangle mesh / Python and C++ CPU | Physical-task alias/preset of v4; real `python` and `cpp` variants, including C++ mesh configuration/contact path |

Preset semantics are deterministic: registry defaults are prepended, but an explicit user option with the same long-option name replaces the preset. Both split (`--backend python`) and inline (`--backend=python`) forms are tested. All listed stage-3 model IDs implement `--smoke-test --output-dir`, force CPU/Agg with a fixed seed, and atomically write strict JSON with finite state, iterations, residual, convergence, and per-variant timing evidence.
