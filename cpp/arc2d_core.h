#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace der {

namespace py = pybind11;

// Independent CPU float64 implementation of the 2-D circular-arc rod model.
// It intentionally does not share the 3-D DERCore's cylinder/contact or
// natural-curvature semantics.
class Arc2DCore {
 public:
  Arc2DCore(const py::array& rest_lengths, const py::array& masses, double ea,
            double ei, double damping, double arc_radius,
            double tube_half_width, double safe_distance, double kp, double ki,
            double max_pi_force, double max_integral_state, double leak_rate,
            double contact_smooth_length = 2.0e-5,
            double direction_epsilon = 1.0e-6);

  py::dict contact_kernel(const py::array& positions,
                          const py::array& integrals,
                          const py::array& previous_errors, double dt,
                          double kp_active, double ki_active) const;

  py::dict implicit_euler_step(
      const py::array& positions, const py::array& velocities,
      const py::array& integrals, const py::array& previous_errors,
      const py::array& prescribed, const py::array& pulse_forces,
      const py::array& reference_curvature, bool tip_constrained, double dt,
      int max_newton_iters, double residual_tol, double step_tol,
      double kp_active, double ki_active) const;

  double nominal_kp() const { return kp_; }
  double nominal_ki() const { return ki_; }

 private:
  std::vector<double> rest_lengths_;
  std::vector<double> masses_;
  double ea_;
  double ei_;
  double damping_;
  double arc_radius_;
  double tube_half_width_;
  double safe_distance_;
  double kp_;
  double ki_;
  double max_pi_force_;
  double max_integral_state_;
  double leak_rate_;
  double contact_smooth_length_;
  double direction_epsilon_;
};

}  // namespace der
