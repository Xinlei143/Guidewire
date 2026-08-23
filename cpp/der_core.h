#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

namespace der {

namespace py = pybind11;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

class DERCore {
 public:
  DERCore(const py::array& rest_lengths, const py::array& masses, double ea,
          double ei, double damping, double tube_radius, double safe_distance,
          double kp, double ki, double max_pi_force, double leak_rate,
          double max_integral_state, double contact_smooth_length = 2.0e-5,
          double direction_epsilon = 1.0e-6);

  py::tuple stretch_kernel(const py::array& positions,
                           const py::array& rest_lengths) const;
  py::tuple bending_kernel(const py::array& positions,
                           const py::array& rest_lengths) const;
  py::dict contact_kernel(const py::array& positions,
                          const py::array& integrals,
                          const py::array& previous_errors, double dt,
                          double ki_active) const;
  double nominal_ki() const { return ki_; }
  py::dict implicit_euler_step(
      const py::array& positions, const py::array& velocities,
      const py::array& rest_lengths, const py::array& masses,
      const py::array& integrals, const py::array& initial_positions,
      const py::array& pulse_forces,
      bool tip_constrained, double tip_displacement, double dt,
      int max_newton_iters, double residual_tol, double step_tol,
      double ki_active) const;
  py::dict implicit_euler_step_cached(
      const py::array& positions, const py::array& velocities,
      const py::array& integrals, const py::array& initial_positions,
      const py::array& pulse_forces, bool tip_constrained,
      double tip_displacement, double dt, int max_newton_iters,
      double residual_tol, double step_tol, double ki_active) const;

 private:
  std::vector<double> rest_lengths_;
  std::vector<double> masses_;
  double ea_;
  double ei_;
  double damping_;
  double tube_radius_;
  double safe_distance_;
  double kp_;
  double ki_;
  double max_pi_force_;
  double leak_rate_;
  double max_integral_state_;
  double contact_smooth_length_;
  double direction_epsilon_;
};

}  // namespace der
