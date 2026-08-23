#include "der_core.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CPU Discrete Elastic Rod numerical core";
  py::class_<der::DERCore>(m, "DERCore")
      .def(py::init<const py::array&, const py::array&, double, double, double,
                    double, double, double, double, double, double, double,
                    double, double>(),
           py::arg("rest_lengths"), py::arg("masses"), py::arg("ea"),
           py::arg("ei"), py::arg("damping"), py::arg("tube_radius"),
           py::arg("safe_distance"), py::arg("kp"), py::arg("ki"),
           py::arg("max_pi_force"), py::arg("leak_rate"),
           py::arg("max_integral_state"),
           py::arg("contact_smooth_length") = 2.0e-5,
           py::arg("direction_epsilon") = 1.0e-6)
      .def("stretch_kernel", &der::DERCore::stretch_kernel)
      .def("bending_kernel", &der::DERCore::bending_kernel)
      .def("contact_kernel",
           [](der::DERCore& self, const py::array& positions,
              const py::array& integrals, const py::array& previous_errors,
              double dt, py::object ki_active) {
             const double active = ki_active.is_none()
                 ? self.nominal_ki() : ki_active.cast<double>();
             return self.contact_kernel(positions, integrals, previous_errors,
                                        dt, active);
           },
           py::arg("positions"), py::arg("integrals"),
           py::arg("previous_errors"), py::arg("dt"),
           py::arg("ki_active") = py::none())
      .def("implicit_euler_step", &der::DERCore::implicit_euler_step,
           py::arg("positions"), py::arg("velocities"),
           py::arg("rest_lengths"), py::arg("masses"),
           py::arg("integrals"), py::arg("initial_positions"),
           py::arg("pulse_forces"),
           py::arg("tip_constrained"), py::arg("tip_displacement"),
           py::arg("dt"), py::arg("max_newton_iters"),
           py::arg("residual_tol"), py::arg("step_tol"),
           py::arg("ki_active"))
      .def("implicit_euler_step_cached", &der::DERCore::implicit_euler_step_cached,
           py::arg("positions"), py::arg("velocities"),
           py::arg("integrals"), py::arg("initial_positions"),
           py::arg("pulse_forces"), py::arg("tip_constrained"),
           py::arg("tip_displacement"), py::arg("dt"),
           py::arg("max_newton_iters"), py::arg("residual_tol"),
           py::arg("step_tol"), py::arg("ki_active"));
}
