#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace der {

namespace py = pybind11;

// CPU float64 geometry/contact/friction kernels and implicit integration for
// the 2-D sine-tube model.  Constitutive history is input-only here and remains
// transactionally committed by the Python quasi-static controller.
class Sine2DCore {
 public:
  Sine2DCore(
      const py::array& rest_lengths, const py::array& masses, double ea,
      double ei, double damping, double amplitude, double length_x,
      double tube_half_width, double safe_distance, double feeder_half_width,
      double feeder_contact_layer, double feeder_kp_line,
      double contact_kp_line, double contact_ki_line,
      double contact_force_cap_line, double friction_mu,
      double friction_kt_line, double max_integral_state, double leak_rate,
      double contact_smooth_length = 2.0e-5,
      double direction_epsilon = 1.0e-6);

  py::dict geometry_kernel(
      const py::array& points,
      const py::object& initial_q = py::none()) const;

  py::dict tracked_geometry_kernel(
      const py::array& points, const py::array& accepted_q,
      double trust_radius) const;

  py::dict contact_kernel(
      const py::array& positions, const py::array& old_integrals,
      const py::array& previous_errors, const py::array& nodal_lengths,
      const py::array& committed_slip,
      const py::array& increment_start_anchors,
      const py::array& committed_contact, double dt,
      bool friction_enabled, double kp_active, double ki_active,
      const py::object& initial_q = py::none()) const;

  py::dict implicit_euler_step(
      const py::array& positions, const py::array& velocities,
      const py::array& old_integrals, const py::array& previous_errors,
      const py::array& nodal_lengths, const py::array& prescribed,
      const py::array& reference_curvature,
      const py::array& committed_slip,
      const py::array& increment_start_anchors,
      const py::array& committed_contact, bool friction_enabled,
      const py::array& accepted_q, double dt, int max_newton_iters,
      double residual_tol, double step_tol, double projection_trust_radius,
      double kp_active, double ki_active) const;

  double nominal_kp() const { return contact_kp_line_; }
  double nominal_ki() const { return contact_ki_line_; }

 private:
  std::vector<double> rest_lengths_;
  std::vector<double> masses_;
  double ea_;
  double ei_;
  double damping_;
  double amplitude_;
  double length_x_;
  double wave_number_;
  double tube_half_width_;
  double safe_distance_;
  double feeder_half_width_;
  double feeder_contact_layer_;
  double feeder_kp_line_;
  double contact_kp_line_;
  double contact_ki_line_;
  double contact_force_cap_line_;
  double friction_mu_;
  double friction_kt_line_;
  double max_integral_state_;
  double leak_rate_;
  double contact_smooth_length_;
  double direction_epsilon_;
};

}  // namespace der
