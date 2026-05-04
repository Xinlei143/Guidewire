"""Torch-native semi-implicit Euler dynamic relaxation for no-twist rods."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from der_no_twist_model import (
    RodParams,
    RodRestState,
    elastic_internal_forces,
    total_energy,
)
from soft_robot_common import (
    Tensor,
    compute_curvature_binormal_magnitudes,
    compute_edge_geometry,
    validate_positions,
)


@dataclass(frozen=True)
class BoundaryCondition:
    """Fixed-node boundary condition for nodal dynamics."""

    fixed_indices: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class ExternalLoads:
    """External nodal loads and uniform gravity."""

    nodal_forces: Tensor
    gravity: float = 0.0


@dataclass(frozen=True)
class SolverOptions:
    """Options for semi-implicit dynamic relaxation."""

    dt: float
    steps: int
    damping: float
    convergence_tol: float
    save_every: int = 1


@dataclass
class SimulationStatus:
    """Termination state and stability warnings."""

    converged: bool = False
    converged_step: int | None = None
    unstable: bool = False
    unstable_step: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StepDiagnostics:
    """Per-step quantities saved to history."""

    energies: dict[str, Tensor]
    metrics: dict[str, Tensor]
    kappa_b_magnitudes: Tensor


@dataclass
class SimulationHistory:
    """Saved dynamic states and diagnostics as detached Torch tensors."""

    time: list[Tensor] = field(default_factory=list)
    positions: list[Tensor] = field(default_factory=list)
    velocities: list[Tensor] = field(default_factory=list)
    energies: list[dict[str, Tensor]] = field(default_factory=list)
    metrics: list[dict[str, Tensor]] = field(default_factory=list)
    kappa_b_magnitudes: list[Tensor] = field(default_factory=list)

    def append(
        self,
        time: Tensor,
        positions: Tensor,
        velocities: Tensor,
        diagnostics: StepDiagnostics,
    ) -> None:
        """Save one simulation frame."""
        self.time.append(time.detach().clone())
        self.positions.append(positions.detach().clone())
        self.velocities.append(velocities.detach().clone())
        self.energies.append({key: value.detach().clone() for key, value in diagnostics.energies.items()})
        self.metrics.append({key: value.detach().clone() for key, value in diagnostics.metrics.items()})
        self.kappa_b_magnitudes.append(diagnostics.kappa_b_magnitudes.detach().clone())

    def append_unstable(
        self,
        time: Tensor,
        positions: Tensor,
        velocities: Tensor,
        num_interior_vertices: int,
    ) -> None:
        """Save a final NaN diagnostic frame for an unstable state."""
        nan = positions.new_tensor(float("nan"))
        diagnostics = StepDiagnostics(
            energies={
                "stretching_energy": nan,
                "bending_energy": nan,
                "gravitational_energy": nan,
                "external_energy": nan,
                "kinetic_energy": nan,
                "total_energy": nan,
            },
            metrics={
                "max_displacement": nan,
                "max_velocity_norm": nan,
                "max_force_norm": nan,
                "max_kappa_b_magnitude": nan,
                "max_stretch_ratio": nan,
                "min_edge_length": nan,
                "energy_growth_ratio": nan,
            },
            kappa_b_magnitudes=positions.new_full((num_interior_vertices,), float("nan")),
        )
        self.append(time, positions, velocities, diagnostics)

    def as_dict(self, status: SimulationStatus) -> dict[str, Tensor | bool | int | list[str] | None]:
        """Return stacked history tensors for plotting/tests."""
        energy_keys = (
            "total_energy",
            "stretching_energy",
            "bending_energy",
            "gravitational_energy",
            "external_energy",
            "kinetic_energy",
        )
        metric_keys = (
            "max_displacement",
            "max_velocity_norm",
            "max_force_norm",
            "max_kappa_b_magnitude",
            "max_stretch_ratio",
            "min_edge_length",
            "energy_growth_ratio",
        )
        history: dict[str, Tensor | bool | int | list[str] | None] = {
            "positions": torch.stack(self.positions) if self.positions else torch.empty(0),
            "velocities": torch.stack(self.velocities) if self.velocities else torch.empty(0),
            "time": torch.stack(self.time) if self.time else torch.empty(0),
            "kappa_b_magnitudes": (
                torch.stack(self.kappa_b_magnitudes)
                if self.kappa_b_magnitudes
                else torch.empty(0)
            ),
            "converged": status.converged,
            "converged_step": status.converged_step,
            "unstable": status.unstable,
            "unstable_step": status.unstable_step,
            "stability_warnings": list(status.warnings),
        }
        for key in energy_keys:
            history[key] = torch.stack([values.get(key, torch.tensor(float("nan"))) for values in self.energies])
        for key in metric_keys:
            history[key] = torch.stack([values.get(key, torch.tensor(float("nan"))) for values in self.metrics])
        return history


@dataclass(frozen=True)
class SimulationResult:
    """Complete result returned by dynamic relaxation."""

    final_positions: Tensor
    final_velocities: Tensor
    history: SimulationHistory
    status: SimulationStatus

    def history_dict(self) -> dict[str, Tensor | bool | int | list[str] | None]:
        """Return stacked history tensors."""
        return self.history.as_dict(self.status)


def _fixed_index_tensor(
    fixed_indices: tuple[int, ...],
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    if any(index < 0 or index >= num_nodes for index in fixed_indices):
        raise ValueError("fixed_indices contains an out-of-range node index.")
    return torch.tensor(fixed_indices, dtype=torch.long, device=device)


def build_external_forces(
    num_nodes: int,
    force_node: int,
    force_vector,
    fixed_indices: tuple[int, ...] = (0, 1),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create the 3D nodal external-force tensor."""
    if force_node < 0:
        force_node += num_nodes
    if force_node < 0 or force_node >= num_nodes:
        raise ValueError(f"force_node must be between 0 and {num_nodes - 1}.")
    if force_node in set(fixed_indices):
        raise ValueError("External force cannot be applied to a fixed node.")

    external_forces = torch.zeros((num_nodes, 3), dtype=dtype, device=device)
    external_forces[force_node] = torch.as_tensor(force_vector, dtype=dtype, device=device)
    return external_forces


def compute_gravity_forces(
    node_masses: Tensor,
    gravity: float,
    fixed_indices: Tensor,
) -> Tensor:
    """Return 3D nodal gravity forces in the negative z direction."""
    gravity_forces = node_masses.new_zeros((node_masses.numel(), 3))
    gravity_forces[:, 2] = -node_masses * gravity
    gravity_forces[fixed_indices] = 0.0
    return gravity_forces


def compute_stability_metrics(
    positions: Tensor,
    rest: RodRestState,
    total_energy_value: Tensor,
    previous_total_energy: Tensor | None,
) -> dict[str, Tensor]:
    """Compute lightweight explicit-step stability diagnostics."""
    _, edge_lengths, _ = compute_edge_geometry(positions)
    safe_rest_lengths = torch.clamp(rest.lengths.to(dtype=positions.dtype, device=positions.device), min=1e-12)
    stretch_ratios = edge_lengths / safe_rest_lengths
    kappa_b_magnitudes = compute_curvature_binormal_magnitudes(positions)

    if previous_total_energy is None or torch.abs(previous_total_energy) <= 1e-12:
        energy_growth_ratio = positions.new_tensor(float("nan"))
    else:
        energy_growth_ratio = torch.abs(total_energy_value) / torch.abs(previous_total_energy)

    return {
        "max_stretch_ratio": torch.max(stretch_ratios) if stretch_ratios.numel() else positions.new_tensor(0.0),
        "min_edge_length": torch.min(edge_lengths) if edge_lengths.numel() else positions.new_tensor(float("inf")),
        "max_kappa_b_magnitude": (
            torch.max(kappa_b_magnitudes) if kappa_b_magnitudes.numel() else positions.new_tensor(0.0)
        ),
        "energy_growth_ratio": energy_growth_ratio,
    }


def stability_warning_messages(metrics: dict[str, Tensor]) -> list[tuple[str, str]]:
    """Return warning keys/messages for suspicious explicit-step diagnostics."""
    warnings: list[tuple[str, str]] = []
    if metrics["max_stretch_ratio"].item() > 2.0:
        warnings.append(
            (
                "large_stretch",
                "Large stretch ratio detected; reduce --dt or EA, or increase damping.",
            )
        )
    if metrics["min_edge_length"].item() < 1e-8:
        warnings.append(
            (
                "edge_collapse",
                "Edge collapse detected; reduce --dt or inspect boundary/loading setup.",
            )
        )
    energy_growth_ratio = metrics["energy_growth_ratio"]
    if torch.isfinite(energy_growth_ratio).item() and energy_growth_ratio.item() > 10.0:
        warnings.append(
            (
                "energy_blowup",
                "Energy growth ratio exceeded 10; the semi-implicit step may be unstable.",
            )
        )
    return warnings


def semi_implicit_step(
    positions: Tensor,
    velocities: Tensor,
    total_forces: Tensor,
    node_masses: Tensor,
    free_indices: Tensor,
    fixed_indices: Tensor,
    fixed_positions: Tensor,
    dt: float,
) -> tuple[Tensor, Tensor]:
    """Return one semi-implicit Euler step."""
    next_velocities = velocities.clone()
    next_positions = positions.clone()
    next_velocities[free_indices] = (
        velocities[free_indices]
        + dt * total_forces[free_indices] / node_masses[free_indices, None]
    )
    next_positions[free_indices] = positions[free_indices] + dt * next_velocities[free_indices]
    next_positions[fixed_indices] = fixed_positions
    next_velocities[fixed_indices] = 0.0
    return next_positions.detach(), next_velocities.detach()


def run_dynamic_relaxation(
    initial_positions: Tensor,
    rest: RodRestState,
    params: RodParams,
    bc: BoundaryCondition,
    loads: ExternalLoads,
    options: SolverOptions,
) -> SimulationResult:
    """Run damped semi-implicit Euler dynamics toward static equilibrium."""
    if options.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if options.steps < 0:
        raise ValueError("steps must be non-negative.")
    if options.damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if options.convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be positive.")
    if options.save_every < 1:
        raise ValueError("save_every must be at least 1.")

    positions = validate_positions(initial_positions).detach().clone()
    if positions.shape != rest.positions.shape:
        raise ValueError("initial_positions and rest.positions must have the same shape.")
    if loads.nodal_forces.shape != positions.shape:
        raise ValueError("loads.nodal_forces must have the same shape as positions.")

    device = positions.device
    dtype = positions.dtype
    num_nodes = positions.shape[0]
    fixed_indices = _fixed_index_tensor(bc.fixed_indices, num_nodes, device)
    all_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
    fixed_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    fixed_mask[fixed_indices] = True
    free_indices = all_indices[~fixed_mask]
    if free_indices.numel() == 0:
        raise ValueError("At least one free node is required.")

    node_masses = params.node_masses.to(dtype=dtype, device=device)
    if torch.any(node_masses[free_indices] <= 0.0):
        raise ValueError("All free nodes must have positive mass.")

    velocities = torch.zeros_like(positions)
    fixed_positions = positions[fixed_indices].clone()
    gravity_forces = compute_gravity_forces(node_masses, loads.gravity, fixed_indices)
    nodal_forces = loads.nodal_forces.to(dtype=dtype, device=device)
    params = RodParams(EA=params.EA, B=params.B, node_masses=node_masses)

    history = SimulationHistory()
    status = SimulationStatus()
    last_saved_step: int | None = None
    previous_total_energy: Tensor | None = None
    emitted_stability_warnings: set[str] = set()

    def make_step_diagnostics(residual_forces: Tensor | None) -> StepDiagnostics:
        nonlocal previous_total_energy
        energies = total_energy(
            positions,
            velocities,
            rest,
            params,
            loads.gravity,
            nodal_forces,
            free_indices,
        )
        metrics = compute_stability_metrics(
            positions,
            rest,
            energies["total_energy"],
            previous_total_energy,
        )
        kappa_b_magnitudes = compute_curvature_binormal_magnitudes(positions)
        metrics["max_displacement"] = torch.max(torch.linalg.norm(positions - initial_positions, dim=1))
        metrics["max_velocity_norm"] = torch.max(torch.linalg.norm(velocities[free_indices], dim=1))
        if residual_forces is None:
            metrics["max_force_norm"] = positions.new_tensor(float("nan"))
        else:
            metrics["max_force_norm"] = torch.max(torch.linalg.norm(residual_forces[free_indices], dim=1))
        previous_total_energy = energies["total_energy"].detach()
        return StepDiagnostics(energies, metrics, kappa_b_magnitudes)

    def handle_stability_warnings(step: int, diagnostics: StepDiagnostics) -> None:
        metrics = diagnostics.metrics
        for warning_key, message in stability_warning_messages(metrics):
            warning_text = (
                f"step {step}: {message} "
                f"max_stretch_ratio={metrics['max_stretch_ratio'].item():.4e}, "
                f"min_edge_length={metrics['min_edge_length'].item():.4e}, "
                f"max_kappa_b={metrics['max_kappa_b_magnitude'].item():.4e}, "
                f"energy_growth_ratio={metrics['energy_growth_ratio'].item():.4e}"
            )
            status.warnings.append(warning_text)
            if warning_key not in emitted_stability_warnings:
                print(f"Stability warning: {warning_text}")
                emitted_stability_warnings.add(warning_key)

    internal = elastic_internal_forces(positions, rest, params)
    residual_forces = internal + nodal_forces + gravity_forces
    diagnostics = make_step_diagnostics(residual_forces)
    handle_stability_warnings(0, diagnostics)
    history.append(positions.new_tensor(0.0), positions, velocities, diagnostics)
    last_saved_step = 0

    final_step = 0
    for step in range(1, options.steps + 1):
        internal = elastic_internal_forces(positions, rest, params)
        residual_forces = internal + nodal_forces + gravity_forces
        damping_forces = -options.damping * velocities
        total_forces = residual_forces + damping_forces

        positions, velocities = semi_implicit_step(
            positions,
            velocities,
            total_forces,
            node_masses,
            free_indices,
            fixed_indices,
            fixed_positions,
            options.dt,
        )
        final_step = step

        if not torch.all(torch.isfinite(positions)) or not torch.all(torch.isfinite(velocities)):
            status.unstable = True
            status.unstable_step = step
            if last_saved_step != step:
                history.append_unstable(
                    positions.new_tensor(step * options.dt),
                    positions,
                    velocities,
                    max(positions.shape[0] - 2, 0),
                )
                last_saved_step = step
            print("Simulation became unstable. Try reducing --dt or increasing --damping.")
            break

        internal = elastic_internal_forces(positions, rest, params)
        residual_forces = internal + nodal_forces + gravity_forces
        diagnostics = make_step_diagnostics(residual_forces)
        handle_stability_warnings(step, diagnostics)

        if step % options.save_every == 0:
            history.append(positions.new_tensor(step * options.dt), positions, velocities, diagnostics)
            last_saved_step = step

        if (
            diagnostics.metrics["max_velocity_norm"].item() < options.convergence_tol
            and diagnostics.metrics["max_force_norm"].item() < options.convergence_tol
        ):
            status.converged = True
            status.converged_step = step
            break

    if not status.unstable and last_saved_step != final_step:
        internal = elastic_internal_forces(positions, rest, params)
        residual_forces = internal + nodal_forces + gravity_forces
        diagnostics = make_step_diagnostics(residual_forces)
        handle_stability_warnings(final_step, diagnostics)
        history.append(positions.new_tensor(final_step * options.dt), positions, velocities, diagnostics)

    return SimulationResult(
        final_positions=positions.detach(),
        final_velocities=velocities.detach(),
        history=history,
        status=status,
    )
