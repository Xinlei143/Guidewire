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
