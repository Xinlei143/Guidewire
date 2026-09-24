"""Command-line cantilever-style demo for the phase-1 no-twist rod model."""

from __future__ import annotations

import argparse

import torch

from .der_no_twist_model import (
    RodParams,
    make_straight_rest_state,
    make_uniform_node_masses,
)
from .semi_implicit_solver import (
    BoundaryCondition,
    ExternalLoads,
    SolverOptions,
    SimulationResult,
    build_external_forces,
    run_dynamic_relaxation,
)
from .visualization import (
    animate_chain_evolution,
    plot_time_history,
    visualize_chain,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stretching + bending DER-style semi-implicit dynamics for a 3D soft "
            "robot chain. Twisting is not implemented in this stage."
        )
    )
    parser.add_argument("--num-links", type=int, default=200, help="Number of serial links.")
    parser.add_argument("--total-length", type=float, default=1.0, help="Initial total length in meters.")
    parser.add_argument(
        "--EA",
        dest="EA",
        type=float,
        default=1000.0,
        help="Axial stiffness EA in N; each edge uses k_i = EA / L_i^0.",
    )
    parser.add_argument("--stretch-stiffness", dest="EA", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--bend-stiffness", type=float, default=0.5, help="Bending stiffness B=EI in N*m^2.")
    parser.add_argument("--node-mass", type=float, default=0.1, help="Mass of each free node in kg.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in m/s^2.")
    parser.add_argument(
        "--force-node",
        type=int,
        default=-1,
        help="Node index where the external force is applied; default -1 means the tip node.",
    )
    parser.add_argument("--force-x", type=float, default=0.0, help="External force x component in N.")
    parser.add_argument("--force-y", type=float, default=0.0, help="External force y component in N.")
    parser.add_argument("--force-z", type=float, default=0.0, help="External force z component in N.")
    parser.add_argument("--dt", type=float, default=2e-5, help="Time step size in seconds.")
    parser.add_argument("--steps", type=int, default=600000, help="Maximum number of time steps.")
    parser.add_argument("--damping", type=float, default=0.5, help="Nodal viscous damping coefficient.")
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-5,
        help="Tolerance for free-node max velocity and undamped residual force.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save history every N steps.",
    )
    parser.add_argument("--animate", action="store_true", help="Compatibility flag; animation is shown unless --no-plot is used.")
    parser.add_argument("--animation-output", type=str, default=None, help="Path for saving the dynamic-process GIF.")
    parser.add_argument("--plot-history", action="store_true", help="Compatibility flag; diagnostics are shown unless --no-plot is used.")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot windows.")
    parser.add_argument(
        "--EB_beam",
        action="store_true",
        dest="eb_beam",
        help="Enable Euler-Bernoulli beam validation comparison.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args(argv)


def resolve_force_node(force_node: int, num_nodes: int) -> int:
    """Normalize negative force-node indexing."""
    if force_node < 0:
        force_node += num_nodes
    if force_node < 0 or force_node >= num_nodes:
        raise ValueError(f"force_node must be between 0 and {num_nodes - 1}.")
    return force_node


def print_displacements(
    initial_positions: torch.Tensor,
    deformed_positions: torch.Tensor,
    force_node: int,
) -> None:
    """Print nodal displacement vectors and selected-node displacement magnitude."""
    displacements = deformed_positions - initial_positions

    print("Node displacements:")
    for node_index, displacement in enumerate(displacements):
        magnitude = torch.linalg.norm(displacement)
        print(
            f"  Node {node_index:2d}: "
            f"ux = {displacement[0].item(): .8e} m, "
            f"uy = {displacement[1].item(): .8e} m, "
            f"uz = {displacement[2].item(): .8e} m, "
            f"|u| = {magnitude.item(): .8e} m"
        )

    force_node_magnitude = torch.linalg.norm(displacements[force_node])
    print(
        f"\nTotal displacement magnitude of force-applied node {force_node}: "
        f"{force_node_magnitude.item():.8e} m"
    )


def print_summary(result: SimulationResult, max_steps: int) -> None:
    """Print solver termination summary."""
    if result.status.unstable:
        print(f"Simulation stopped at unstable step {result.status.unstable_step}.")
    elif result.status.converged:
        print(f"Simulation converged at step {result.status.converged_step}.")
    else:
        print(f"Simulation did not converge within {max_steps} steps.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        from softrobot._smoke import configure_cpu_runtime

        configure_cpu_runtime()
        args.num_links = 3
        args.steps = 2
        args.save_every = 1
        args.gravity = 0.0
        args.force_y = 0.01
        args.dt = 1.0e-4
        args.no_plot = True
    if args.num_links < 2:
        raise ValueError("num-links must be at least 2 for the default clamped edge.")

    rest = make_straight_rest_state(args.num_links, args.total_length)
    initial_positions = rest.positions.clone()
    num_nodes = initial_positions.shape[0]
    fixed_indices = (0, 1)
    force_node = resolve_force_node(args.force_node, num_nodes)

    node_masses = make_uniform_node_masses(
        num_nodes=num_nodes,
        value=args.node_mass,
        fixed_indices=fixed_indices,
    )
    external_forces = build_external_forces(
        num_nodes,
        force_node,
        (args.force_x, args.force_y, args.force_z),
        fixed_indices=fixed_indices,
    )

    result = run_dynamic_relaxation(
        initial_positions=initial_positions,
        rest=rest,
        params=RodParams(EA=args.EA, B=args.bend_stiffness, node_masses=node_masses),
        bc=BoundaryCondition(fixed_indices=fixed_indices),
        loads=ExternalLoads(nodal_forces=external_forces, gravity=args.gravity),
        options=SolverOptions(
            dt=args.dt,
            steps=args.steps,
            damping=args.damping,
            convergence_tol=args.convergence_tol,
            save_every=args.save_every,
        ),
    )
    history = result.history_dict()

    print_summary(result, args.steps)
    print_displacements(initial_positions, result.final_positions, force_node)

    if args.smoke_test:
        from softrobot._smoke import write_smoke_result

        write_smoke_result(
            args.output_dir, "foundation-cantilever-der",
            result.final_positions.detach().cpu().numpy(), steps=args.steps,
            saved_frames=len(history["time"]),
        )

    if args.no_plot:
        if args.animate:
            print("--animate was requested, but --no-plot is enabled; skipping interactive animation.")
        if args.animation_output:
            print("--animation-output was requested, but --no-plot is enabled; skipping GIF output.")
        if args.plot_history:
            print("--plot-history was requested, but --no-plot is enabled; skipping diagnostics.")
        if args.eb_beam:
            print("--EB_beam was requested, but --no-plot is enabled; skipping EB comparison.")
        return 0

    animate_chain_evolution(
        initial_positions=initial_positions,
        history=history,
        rest_lengths=rest.lengths,
        external_forces=external_forces,
        force_node=force_node,
        fixed_indices=fixed_indices,
        animation_output=args.animation_output,
        show=True,
    )
    visualize_chain(
        initial_positions,
        result.final_positions,
        rest.lengths,
        external_forces,
        fixed_indices=fixed_indices,
    )
    if args.eb_beam:
        from .eb_beam_validation import (
            eb_tip_load_skip_reason,
            euler_bernoulli_tip_load_positions,
        )
        from .visualization import plot_eb_beam_comparison

        eb_skip_reason = eb_tip_load_skip_reason(
            force_node=force_node,
            num_nodes=num_nodes,
            force_vector=external_forces[force_node],
            B=args.bend_stiffness,
            gravity=args.gravity,
        )
        if eb_skip_reason is None:
            eb_positions = euler_bernoulli_tip_load_positions(
                initial_positions=initial_positions,
                force_vector=external_forces[force_node],
                B=args.bend_stiffness,
                fixed_indices=fixed_indices,
            )
            plot_eb_beam_comparison(
                initial_positions=initial_positions,
                numerical_positions=result.final_positions,
                eb_positions=eb_positions,
                fixed_indices=fixed_indices,
            )
        else:
            print(f"Skipping Euler-Bernoulli comparison: {eb_skip_reason}")
    plot_time_history(history, rest.lengths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
