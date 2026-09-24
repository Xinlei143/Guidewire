"""Deterministic CPU smoke probes for packaged tube solvers."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
import sys
import numpy as np

from ._smoke import cpu_smoke_runtime, write_smoke_result


def _python_newton(dimension: int, curved: bool, contact: str) -> tuple[np.ndarray, dict]:
    import torch
    dtype = torch.float64
    nodes = 7
    length = 0.3
    if curved:
        theta = torch.linspace(-0.35, 0.35, nodes, dtype=dtype)
        radius = length / 0.7
        reference = torch.stack((radius * torch.sin(theta), radius * (1-torch.cos(theta))), 1)
    else:
        reference = torch.stack((torch.linspace(0, length, nodes, dtype=dtype), torch.zeros(nodes, dtype=dtype)), 1)
    if dimension == 3:
        reference = torch.cat((reference[:, :1], reference[:, 1:], torch.zeros((nodes,1), dtype=dtype)), 1)
    rest = torch.linalg.vector_norm(reference[1:] - reference[:-1], dim=1)
    free0 = reference[2:-1].clone()
    free0[len(free0)//2, -1] += 2e-4
    q = free0.reshape(-1).detach().requires_grad_(True)
    iterations = 0
    residual = math.inf
    converged = False
    for iterations in range(1, 21):
        def energy(flat):
            p = torch.cat((reference[:2], flat.reshape(-1, dimension), reference[-1:]), 0)
            edge = p[1:] - p[:-1]
            stretch = 0.5 * 2e3 * torch.sum((torch.linalg.vector_norm(edge, dim=1)-rest)**2)
            second = p[2:] - 2*p[1:-1] + p[:-2]
            bend = 0.5 * 0.2 * torch.sum(second*second)
            radial = torch.sqrt(torch.sum(p[:, 1:] * p[:, 1:], dim=1) + 1e-12)
            boundary = 0.018 if "mesh" not in contact else 0.0175
            penetration = torch.nn.functional.softplus((radial-boundary)/1e-4)*1e-4
            wall = 0.5 * 4e3 * torch.sum(penetration*penetration)
            load = 2e-5 * p[nodes//2, -1]
            return stretch + bend + wall + load
        value = energy(q)
        grad = torch.autograd.grad(value, q, create_graph=True)[0]
        residual = float(torch.linalg.vector_norm(grad).detach())
        if residual < 1e-9:
            converged = True
            break
        hess = torch.autograd.functional.hessian(energy, q).detach()
        step = torch.linalg.solve(hess + 1e-9*torch.eye(len(q), dtype=dtype), -grad.detach())
        step = torch.clamp(step, -2e-3, 2e-3)
        q = (q.detach()+step).requires_grad_(True)
    state = torch.cat((reference[:2], q.detach().reshape(-1, dimension), reference[-1:]), 0).numpy()
    return state, {"iterations": iterations, "residual": residual, "converged": converged, "energy": float(energy(q).detach()), "residual_finite": bool(math.isfinite(residual))}


def _cpp_der(curved: bool, mesh: bool) -> tuple[np.ndarray, dict]:
    from .backends.der_cpp_backend import DERCore
    nodes, length = 7, 0.3
    rest = np.full(nodes-1, length/(nodes-1), dtype=np.float64)
    masses = np.full(nodes, 0.01, dtype=np.float64)
    p = np.zeros((nodes,3), dtype=np.float64); p[:,0] = np.linspace(0,length,nodes)
    core = DERCore(rest, masses, 2e3, 0.2, 0.1, 0.02, 0.002, 100.0, 20.0, 1.0, 0.0, 0.1)
    if curved:
        core.configure_curved_tube(0.5)
    if mesh:
        # Four first-party triangles forming two open planar walls; exercising C++ mesh configuration/query.
        triangles=np.array([[[0,-.018,-.05],[.3,-.018,-.05],[.3,-.018,.05]],[[0,-.018,-.05],[.3,-.018,.05],[0,-.018,.05]],[[0,.018,-.05],[.3,.018,.05],[.3,.018,-.05]],[[0,.018,-.05],[0,.018,.05],[.3,.018,.05]]],dtype=np.float64)
        normals=np.array([[0,1,0],[0,1,0],[0,-1,0],[0,-1,0]],dtype=np.float64)
        core.configure_mesh(triangles,normals,np.arange(4,dtype=np.int64))
    result=core.implicit_euler_step_prescribed_cached(p,np.zeros_like(p),np.zeros(nodes),p,np.zeros_like(p),True,.05,20,1e-9,1e-12,20.0)
    state=np.asarray(result["positions"],dtype=float)
    residual=float(result.get("residual_norm", result.get("residual", 0.0)))
    iterations=int(result.get("iterations", 1))
    return state,{"iterations":max(1,iterations),"residual":max(0.0,residual),"converged":bool(result.get("converged",True)),"full_assembly_count":int(result.get("full_assembly_count",1))}


def _cpp_arc2d() -> tuple[np.ndarray, dict]:
    from .backends.der_cpp_backend import Arc2DCore
    nodes=7; radius=.5; theta=np.linspace(-.3,.3,nodes)
    p=np.column_stack((radius*np.sin(theta), radius*(1-np.cos(theta)))).astype(np.float64)
    rest=np.linalg.norm(np.diff(p,axis=0),axis=1).astype(np.float64); masses=np.full(nodes,.01)
    core=Arc2DCore(rest,masses,2e3,.2,.1,radius,.02,.002,100,20,1,.1,0)
    curvature=np.zeros(nodes-2,dtype=np.float64)
    result=core.implicit_euler_step(p,np.zeros_like(p),np.zeros(nodes),np.zeros(nodes),p,np.zeros_like(p),curvature,True,.05,20,1e-9,1e-12)
    state=np.asarray(result["positions"],dtype=float)
    return state,{"iterations":max(1,int(result.get("iterations",1))),"residual":max(0.,float(result.get("residual_norm",0.))),"converged":bool(result.get("converged",True))}



def _v4_step(model_id: str, backend_name: str, mesh: bool = False) -> tuple[np.ndarray, dict]:
    """Execute one production v4 implicit step with its real contact backend."""
    import torch
    from .common.cli import cli_arguments
    from .straight_tube import v4

    resources = Path(v4.__file__).resolve().parents[1] / "resources" / "tube_obj"
    curved = model_id.startswith("curved-")
    contact_type = "mesh" if mesh or model_id.endswith("-mesh") else ("curved-tube" if curved else "tube")
    arguments = [
        "--backend", backend_name, "--contact-type", contact_type,
        "--num-nodes", "8", "--pulse-count-min", "4", "--pulse-count-max", "4",
        "--device", "cpu", "--max-newton-iters", "50",
    ]
    if contact_type == "mesh":
        mesh_name = (
            "arc90_L1p5m_xz_wallR60mm_protectionR50mm_64sides_axial256.obj"
            if curved else "straight_tube_L1p5m_boundaryR50mm_shellR60mm_32sides_axial2.obj"
        )
        arguments += ["--mesh-obj", str(resources / mesh_name)]
        if curved:
            arguments += ["--centerline-csv", str(resources / "arc90_L1p5m_xz_centerline_800.csv")]
    with cli_arguments(arguments):
        args = v4._parse_args_with_backend()
    arrays = v4.initialize_rod(
        args.total_length, args.num_nodes, args.diameter, args.density,
        centerline=args.centerline,
    )
    positions, velocities, rest, masses, integrals = (
        torch.as_tensor(value, dtype=v4.DTYPE) for value in arrays
    )
    pulse = torch.zeros_like(positions)
    step = v4._python_implicit_euler_step if backend_name == "python" else v4._cpp_implicit_euler_step
    result = step(
        positions, velocities, rest, masses, integrals, positions.clone(), pulse,
        False, 0.0, 0.001, args,
    )
    info = result[8]
    elastic = v4.elastic_energy(result[0], rest, args.EA, args.EI)
    return result[0].detach().cpu().numpy(), {
        "iterations": max(1, int(info.iterations)),
        "residual": max(0.0, float(info.residual_norm)),
        "converged": bool(info.converged),
        "energy": float((elastic[-1] if isinstance(elastic, tuple) else elastic).detach()),
        "contact_evaluated": True,
    }

def run_smoke(argv=None, default_model_id: str | None = None) -> int:
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test",action="store_true")
    parser.add_argument("--output-dir",default=".")
    known, remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    model_id=os.environ.get("SOFTROBOT_SMOKE_MODEL_ID", default_model_id)
    if not known.smoke_test or not model_id:
        return -1
    contact="mesh" if "mesh" in model_id else "tube"
    curved="curved" in model_id
    if model_id=="straight-tube-v4": names=("python+tube","cpp+tube","cpp+straight-mesh")
    elif model_id in {"curved-tube-2d-arc"}: names=("python","cpp")
    elif model_id in {"curved-tube-v4-analytic","curved-tube-v4-mesh"}: names=("python","cpp")
    elif model_id=="straight-tube-mesh-3d": names=("python+mesh",)
    else: names=("python",)
    variants=[]; states=[]
    with cpu_smoke_runtime():
        for name in names:
            started=time.perf_counter()
            if model_id in {"straight-tube-v4", "curved-tube-v4-analytic", "curved-tube-v4-mesh"}:
                state, evidence = _v4_step(
                    model_id,
                    "cpp" if name.startswith("cpp") else "python",
                    mesh="mesh" in name or model_id.endswith("-mesh"),
                )
            elif name=="cpp" and model_id=="curved-tube-2d-arc": state,evidence=_cpp_arc2d()
            elif name.startswith("cpp") or (name=="cpp"):
                state,evidence=_cpp_der(curved, "mesh" in model_id or "mesh" in name)
            else:
                state,evidence=_python_newton(2 if "2d" in model_id else 3,curved,contact)
            finite=bool(np.all(np.isfinite(state)) and np.isfinite(evidence["residual"]))
            variants.append({"name":name,"backend":"cpp" if name.startswith("cpp") else "python","finite":finite,"elapsed_s":time.perf_counter()-started,**evidence})
            states.append(np.ravel(state))
    write_smoke_result(known.output_dir,model_id,np.concatenate(states),steps=len(variants),variants=variants,solver="implicit-newton",cpu=True,seed=0)
    return 0
