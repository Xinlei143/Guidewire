#include "arc2d_core.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace der {
namespace {

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;

py::buffer_info checked_array(const py::array& array, const std::string& name,
                              int ndim) {
  auto info = array.request();
  if (info.ndim != ndim ||
      info.format != py::format_descriptor<double>::format() ||
      info.itemsize != static_cast<ssize_t>(sizeof(double))) {
    throw py::type_error(name + " must be a contiguous float64 array");
  }
  if (!(array.flags() & py::array::c_style)) {
    throw py::value_error(name + " must be C-contiguous");
  }
  return info;
}

void require_finite(double value, const char* name) {
  if (!std::isfinite(value))
    throw py::value_error(std::string(name) + " must be finite");
}

void require_finite_values(const double* values, ssize_t count,
                           const char* name) {
  for (ssize_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index]))
      throw py::value_error(std::string(name) + " must contain only finite values");
  }
}

void require_positive_values(const double* values, ssize_t count,
                             const char* name) {
  for (ssize_t index = 0; index < count; ++index) {
    if (!(values[index] > 0.0) || !std::isfinite(values[index])) {
      throw py::value_error(std::string(name) +
                            " must contain only finite positive values");
    }
  }
}

template <int D>
struct Jet {
  double value = 0.0;
  Eigen::Matrix<double, D, 1> gradient =
      Eigen::Matrix<double, D, 1>::Zero();
  Eigen::Matrix<double, D, D> hessian =
      Eigen::Matrix<double, D, D>::Zero();

  Jet() = default;
  explicit Jet(double value_in) : value(value_in) {}

  static Jet variable(double value_in, int index) {
    Jet result(value_in);
    result.gradient(index) = 1.0;
    return result;
  }
};

template <int D>
Jet<D> operator+(const Jet<D>& left, const Jet<D>& right) {
  Jet<D> result;
  result.value = left.value + right.value;
  result.gradient = left.gradient + right.gradient;
  result.hessian = left.hessian + right.hessian;
  return result;
}

template <int D>
Jet<D> operator-(const Jet<D>& left, const Jet<D>& right) {
  Jet<D> result;
  result.value = left.value - right.value;
  result.gradient = left.gradient - right.gradient;
  result.hessian = left.hessian - right.hessian;
  return result;
}

template <int D>
Jet<D> operator-(const Jet<D>& value) {
  Jet<D> result;
  result.value = -value.value;
  result.gradient = -value.gradient;
  result.hessian = -value.hessian;
  return result;
}

template <int D>
Jet<D> operator*(const Jet<D>& left, const Jet<D>& right) {
  Jet<D> result;
  result.value = left.value * right.value;
  result.gradient = right.value * left.gradient + left.value * right.gradient;
  result.hessian = right.value * left.hessian + left.value * right.hessian +
                   left.gradient * right.gradient.transpose() +
                   right.gradient * left.gradient.transpose();
  return result;
}

template <int D>
Jet<D> compose(const Jet<D>& value, double output, double first,
               double second) {
  Jet<D> result;
  result.value = output;
  result.gradient = first * value.gradient;
  result.hessian = first * value.hessian +
                   second * value.gradient * value.gradient.transpose();
  return result;
}

template <int D>
Jet<D> jet_inverse(const Jet<D>& value) {
  // All physical denominators are clamped before reaching this helper.  The
  // zero guard only gives the same finite, zero-derivative convention as the
  // reference's length clamp for degenerate test inputs.
  if (std::abs(value.value) <= 1.0e-30) return Jet<D>(0.0);
  const double inverse = 1.0 / value.value;
  return compose(value, inverse, -inverse * inverse,
                 2.0 * inverse * inverse * inverse);
}

template <int D>
Jet<D> operator/(const Jet<D>& left, const Jet<D>& right) {
  return left * jet_inverse(right);
}

template <int D>
Jet<D> operator+(double left, const Jet<D>& right) {
  return Jet<D>(left) + right;
}
template <int D>
Jet<D> operator+(const Jet<D>& left, double right) {
  return left + Jet<D>(right);
}
template <int D>
Jet<D> operator-(double left, const Jet<D>& right) {
  return Jet<D>(left) - right;
}
template <int D>
Jet<D> operator-(const Jet<D>& left, double right) {
  return left - Jet<D>(right);
}
template <int D>
Jet<D> operator*(double left, const Jet<D>& right) {
  return Jet<D>(left) * right;
}
template <int D>
Jet<D> operator*(const Jet<D>& left, double right) {
  return left * Jet<D>(right);
}
template <int D>
Jet<D> operator/(double left, const Jet<D>& right) {
  return Jet<D>(left) / right;
}
template <int D>
Jet<D> operator/(const Jet<D>& left, double right) {
  return left / Jet<D>(right);
}

template <int D>
Jet<D> jet_sqrt(const Jet<D>& value) {
  // torch.linalg.vector_norm has a zero derivative at an exactly zero vector.
  if (value.value <= 0.0) return Jet<D>(0.0);
  const double root = std::sqrt(value.value);
  return compose(value, root, 0.5 / root,
                 -0.25 / (root * root * root));
}

template <int D>
Jet<D> jet_exp(const Jet<D>& value) {
  const double output = std::exp(value.value);
  return compose(value, output, output, output);
}

template <int D>
Jet<D> jet_log1p(const Jet<D>& value) {
  const double denominator = 1.0 + value.value;
  return compose(value, std::log1p(value.value), 1.0 / denominator,
                 -1.0 / (denominator * denominator));
}

template <int D>
Jet<D> jet_clamp_min(const Jet<D>& value, double minimum) {
  return value.value < minimum ? Jet<D>(minimum) : value;
}

template <int D>
Jet<D> jet_clamp(const Jet<D>& value, double lower, double upper) {
  if (value.value < lower) return Jet<D>(lower);
  if (value.value > upper) return Jet<D>(upper);
  return value;
}

template <int D>
Jet<D> jet_abs(const Jet<D>& value) {
  if (value.value < 0.0) return -value;
  // torch.abs assigns a zero derivative at its kink.
  if (value.value == 0.0) return Jet<D>(0.0);
  return value;
}

template <int D>
Jet<D> jet_softplus(const Jet<D>& value, double beta) {
  const Jet<D> scaled = beta * value;
  // F.softplus uses its default threshold=20 branch.
  if (scaled.value > 20.0) return value;
  return jet_log1p(jet_exp(scaled)) / beta;
}

template <int D>
Jet<D> jet_sigmoid(const Jet<D>& value) {
  if (value.value >= 0.0) {
    const Jet<D> exp_negative = jet_exp(-value);
    return 1.0 / (1.0 + exp_negative);
  }
  const Jet<D> exp_positive = jet_exp(value);
  return exp_positive / (1.0 + exp_positive);
}

template <int D>
Jet<D> dot2(const std::array<Jet<D>, 2>& left,
            const std::array<Jet<D>, 2>& right) {
  return left[0] * right[0] + left[1] * right[1];
}

template <int D>
Jet<D> local_stretch_energy(const std::array<Jet<D>, 6>& positions,
                            double rest_length, double ea) {
  const std::array<Jet<D>, 2> edge = {
      positions[2] - positions[0], positions[3] - positions[1]};
  const Jet<D> length = jet_clamp_min(jet_sqrt(dot2(edge, edge)), 1.0e-12);
  const Jet<D> extension = length - rest_length;
  return 0.5 * ea * extension * extension / rest_length;
}

template <int D>
Jet<D> local_bending_energy(const std::array<Jet<D>, 6>& positions,
                            double left_rest, double right_rest,
                            double reference_curvature, double ei) {
  const std::array<Jet<D>, 2> left = {
      positions[2] - positions[0], positions[3] - positions[1]};
  const std::array<Jet<D>, 2> right = {
      positions[4] - positions[2], positions[5] - positions[3]};
  const Jet<D> left_length =
      jet_clamp_min(jet_sqrt(dot2(left, left)), 1.0e-12);
  const Jet<D> right_length =
      jet_clamp_min(jet_sqrt(dot2(right, right)), 1.0e-12);
  const std::array<Jet<D>, 2> tangent_left = {
      left[0] / left_length, left[1] / left_length};
  const std::array<Jet<D>, 2> tangent_right = {
      right[0] / right_length, right[1] / right_length};
  const Jet<D> cross = tangent_left[0] * tangent_right[1] -
                       tangent_left[1] * tangent_right[0];
  const Jet<D> denominator =
      jet_clamp_min(1.0 + dot2(tangent_left, tangent_right), 1.0e-12);
  const Jet<D> curvature = 2.0 * cross / denominator;
  const double voronoi = 0.5 * (left_rest + right_rest);
  const Jet<D> error = curvature - reference_curvature;
  return 0.5 * ei * error * error / voronoi;
}

template <int D>
struct ArcContactValue {
  std::array<Jet<D>, 2> force;
  Jet<D> integral;
  Jet<D> error;
  Jet<D> weight;
  Jet<D> clearance;
  bool active = false;
};

template <int D>
ArcContactValue<D> evaluate_arc_contact(
    const std::array<Jet<D>, 2>& point, double old_integral,
    double previous_error, double dt, double arc_radius,
    double tube_half_width, double safe_distance, double kp_active,
    double ki_active, double max_pi_force, double max_integral_state,
    double leak_rate, double smooth_length, double direction_epsilon) {
  const std::array<Jet<D>, 2> radial = {
      point[0], point[1] - arc_radius};
  const Jet<D> rho = jet_clamp_min(jet_sqrt(dot2(radial, radial)), 1.0e-12);
  const std::array<Jet<D>, 2> normal = {radial[0] / rho, radial[1] / rho};
  const Jet<D> signed_distance = rho - arc_radius;
  ArcContactValue<D> output;
  output.clearance = tube_half_width - jet_abs(signed_distance);
  const Jet<D> z = jet_abs(signed_distance) -
                   (tube_half_width - safe_distance);
  const double beta = 1.0 / smooth_length;
  output.error = jet_softplus(z, beta);
  output.weight = jet_sigmoid(beta * z);
  const double decay = std::exp(-leak_rate * dt);
  const Jet<D> candidate = jet_clamp(
      output.weight * (old_integral + output.error * dt) +
          (1.0 - output.weight) * old_integral * decay,
      0.0, max_integral_state);
  const double unsaturated =
      kp_active * output.error.value + ki_active * candidate.value;
  const bool blocked =
      (ki_active > 0.0 && unsaturated >= max_pi_force &&
       candidate.value > old_integral) ||
      (ki_active > 0.0 && output.error.value < previous_error &&
       ki_active * old_integral > kp_active * output.error.value &&
       candidate.value > old_integral);
  output.integral = blocked ? Jet<D>(old_integral) : candidate;
  const Jet<D> magnitude = jet_clamp(
      kp_active * output.error + ki_active * output.integral, 0.0,
      max_pi_force);
  const Jet<D> direction_norm =
      jet_sqrt(signed_distance * signed_distance +
               direction_epsilon * direction_epsilon);
  output.force[0] =
      magnitude * (-signed_distance * normal[0] / direction_norm);
  output.force[1] =
      magnitude * (-signed_distance * normal[1] / direction_norm);
  output.active = z.value > 0.0;
  return output;
}

struct BlockPenta2D {
  explicit BlockPenta2D(int block_count)
      : count(block_count),
        diagonal(block_count, Mat2::Zero()),
        upper1(std::max(block_count - 1, 0), Mat2::Zero()),
        lower1(std::max(block_count - 1, 0), Mat2::Zero()),
        upper2(std::max(block_count - 2, 0), Mat2::Zero()),
        lower2(std::max(block_count - 2, 0), Mat2::Zero()) {}

  void reset() {
    std::fill(diagonal.begin(), diagonal.end(), Mat2::Zero());
    std::fill(upper1.begin(), upper1.end(), Mat2::Zero());
    std::fill(lower1.begin(), lower1.end(), Mat2::Zero());
    std::fill(upper2.begin(), upper2.end(), Mat2::Zero());
    std::fill(lower2.begin(), lower2.end(), Mat2::Zero());
  }

  void add_block(int row, int column, const Mat2& value) {
    const int offset = column - row;
    if (offset == 0) diagonal[row] += value;
    else if (offset == 1) upper1[row] += value;
    else if (offset == -1) lower1[column] += value;
    else if (offset == 2) upper2[row] += value;
    else if (offset == -2) lower2[column] += value;
    else if (value.cwiseAbs().maxCoeff() > 1.0e-14)
      throw std::logic_error("Arc2D Jacobian exceeded block-pentadiagonal bandwidth");
  }

  Eigen::VectorXd matvec(const Eigen::VectorXd& value) const {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(2 * count);
    for (int row = 0; row < count; ++row) {
      result.segment<2>(2 * row) += diagonal[row] * value.segment<2>(2 * row);
      if (row + 1 < count) {
        result.segment<2>(2 * row) +=
            upper1[row] * value.segment<2>(2 * (row + 1));
        result.segment<2>(2 * (row + 1)) +=
            lower1[row] * value.segment<2>(2 * row);
      }
      if (row + 2 < count) {
        result.segment<2>(2 * row) +=
            upper2[row] * value.segment<2>(2 * (row + 2));
        result.segment<2>(2 * (row + 2)) +=
            lower2[row] * value.segment<2>(2 * row);
      }
    }
    return result;
  }

  Eigen::MatrixXd dense() const {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(2 * count, 2 * count);
    for (int row = 0; row < count; ++row) {
      result.block<2, 2>(2 * row, 2 * row) = diagonal[row];
      if (row + 1 < count) {
        result.block<2, 2>(2 * row, 2 * (row + 1)) = upper1[row];
        result.block<2, 2>(2 * (row + 1), 2 * row) = lower1[row];
      }
      if (row + 2 < count) {
        result.block<2, 2>(2 * row, 2 * (row + 2)) = upper2[row];
        result.block<2, 2>(2 * (row + 2), 2 * row) = lower2[row];
      }
    }
    return result;
  }

  bool solve(const Eigen::VectorXd& right_hand_side, Eigen::VectorXd* result,
             double* relative_residual) const {
    if (count <= 0) return false;
    auto diagonal_work = diagonal;
    auto upper1_work = upper1;
    auto lower1_work = lower1;
    const auto upper2_work = upper2;
    auto lower2_work = lower2;
    std::vector<Vec2> rhs(count, Vec2::Zero());
    for (int index = 0; index < count; ++index)
      rhs[index] = right_hand_side.segment<2>(2 * index);

    const auto inverse_if_stable = [](const Mat2& matrix, Mat2* inverse) {
      const double scale = std::max(1.0, matrix.cwiseAbs().maxCoeff());
      const double determinant = matrix.determinant();
      if (!matrix.allFinite() || !std::isfinite(determinant) ||
          std::abs(determinant) <= 1.0e-14 * scale * scale) {
        return false;
      }
      *inverse = matrix.inverse();
      return inverse->allFinite();
    };

    for (int index = 0; index < count; ++index) {
      Mat2 inverse;
      if (!inverse_if_stable(diagonal_work[index], &inverse)) return false;
      if (index + 1 < count) {
        const Mat2 factor = lower1_work[index] * inverse;
        diagonal_work[index + 1] -= factor * upper1_work[index];
        if (index + 2 < count)
          upper1_work[index + 1] -= factor * upper2_work[index];
        rhs[index + 1] -= factor * rhs[index];
      }
      if (index + 2 < count) {
        const Mat2 factor = lower2_work[index] * inverse;
        diagonal_work[index + 2] -= factor * upper2_work[index];
        lower1_work[index + 1] -= factor * upper1_work[index];
        rhs[index + 2] -= factor * rhs[index];
      }
    }

    result->setZero(2 * count);
    for (int index = count - 1; index >= 0; --index) {
      Mat2 inverse;
      if (!inverse_if_stable(diagonal_work[index], &inverse)) return false;
      Vec2 value = rhs[index];
      if (index + 1 < count)
        value -= upper1_work[index] * result->segment<2>(2 * (index + 1));
      if (index + 2 < count)
        value -= upper2_work[index] * result->segment<2>(2 * (index + 2));
      result->segment<2>(2 * index) = inverse * value;
    }
    const Eigen::VectorXd difference = matvec(*result) - right_hand_side;
    *relative_residual = difference.norm() /
                         std::max(right_hand_side.norm(), 1.0e-30);
    return result->allFinite() && std::isfinite(*relative_residual) &&
           *relative_residual <= 1.0e-8;
  }

  int count;
  std::vector<Mat2> diagonal;
  std::vector<Mat2> upper1;
  std::vector<Mat2> lower1;
  std::vector<Mat2> upper2;
  std::vector<Mat2> lower2;
};

struct ArcConstants {
  double ea;
  double ei;
  double damping;
  double arc_radius;
  double tube_half_width;
  double safe_distance;
  double max_pi_force;
  double max_integral_state;
  double leak_rate;
  double contact_smooth_length;
  double direction_epsilon;
};

void assemble_residual(
    const Eigen::VectorXd& q, const std::vector<Vec2>& old_positions,
    const std::vector<Vec2>& old_velocities, const std::vector<Vec2>& prescribed,
    const std::vector<Vec2>& pulse_forces, const std::vector<double>& rest,
    const std::vector<double>& masses, const std::vector<double>& integrals,
    const std::vector<double>& previous_errors,
    const std::vector<double>& reference_curvature, int first_free,
    int last_free, double dt, double kp_active, double ki_active,
    const ArcConstants& constants, bool with_jacobian,
    Eigen::VectorXd* residual, BlockPenta2D* system) {
  const int nodes = static_cast<int>(old_positions.size());
  const int free_count = last_free - first_free;
  std::vector<Vec2> positions = prescribed;
  for (int node = first_free; node < last_free; ++node)
    positions[node] = q.segment<2>(2 * (node - first_free));
  std::vector<Vec2> gradient(nodes, Vec2::Zero());
  if (with_jacobian) system->reset();

  const auto free_block = [first_free, last_free](int node) {
    return node >= first_free && node < last_free ? node - first_free : -1;
  };
  const auto scatter_local = [&](const Jet<6>& energy,
                                 const std::array<int, 3>& local_nodes,
                                 int local_node_count) {
    for (int row = 0; row < local_node_count; ++row) {
      const int row_block = free_block(local_nodes[row]);
      if (row_block >= 0)
        gradient[local_nodes[row]] += energy.gradient.segment<2>(2 * row);
      if (!with_jacobian || row_block < 0) continue;
      for (int column = 0; column < local_node_count; ++column) {
        const int column_block = free_block(local_nodes[column]);
        if (column_block >= 0) {
          system->add_block(row_block, column_block,
                            energy.hessian.block<2, 2>(2 * row, 2 * column));
        }
      }
    }
  };

  // Every local contribution uses a six-DOF second-order Jet.  Stretch uses
  // the first four coordinates and bending uses all three adjacent nodes.
  for (int edge = 0; edge < nodes - 1; ++edge) {
    std::array<Jet<6>, 6> local;
    for (int coordinate = 0; coordinate < 2; ++coordinate) {
      local[coordinate] = Jet<6>::variable(positions[edge](coordinate), coordinate);
      local[2 + coordinate] =
          Jet<6>::variable(positions[edge + 1](coordinate), 2 + coordinate);
      local[4 + coordinate] = Jet<6>(0.0);
    }
    const Jet<6> energy =
        local_stretch_energy(local, rest[edge], constants.ea);
    scatter_local(energy, {edge, edge + 1, -1}, 2);
  }
  for (int vertex = 0; vertex < nodes - 2; ++vertex) {
    std::array<Jet<6>, 6> local;
    for (int local_node = 0; local_node < 3; ++local_node) {
      for (int coordinate = 0; coordinate < 2; ++coordinate) {
        local[2 * local_node + coordinate] = Jet<6>::variable(
            positions[vertex + local_node](coordinate),
            2 * local_node + coordinate);
      }
    }
    const Jet<6> energy = local_bending_energy(
        local, rest[vertex], rest[vertex + 1], reference_curvature[vertex],
        constants.ei);
    scatter_local(energy, {vertex, vertex + 1, vertex + 2}, 3);
  }

  residual->setZero(2 * free_count);
  for (int node = 0; node < nodes; ++node) {
    std::array<Jet<2>, 2> point = {
        Jet<2>::variable(positions[node](0), 0),
        Jet<2>::variable(positions[node](1), 1)};
    const ArcContactValue<2> contact = evaluate_arc_contact(
        point, integrals[node], previous_errors[node], dt, constants.arc_radius,
        constants.tube_half_width, constants.safe_distance, kp_active, ki_active,
        constants.max_pi_force, constants.max_integral_state,
        constants.leak_rate, constants.contact_smooth_length,
        constants.direction_epsilon);
    const int block = free_block(node);
    if (block < 0) continue;
    const Vec2 old_q = old_positions[node];
    const Vec2 current_q = positions[node];
    (*residual).segment<2>(2 * block) =
        masses[node] * (current_q - old_q - dt * old_velocities[node]) /
            (dt * dt) +
        gradient[node] + constants.damping * (current_q - old_q) / dt -
        Vec2(contact.force[0].value, contact.force[1].value) -
        pulse_forces[node];
    if (with_jacobian) {
      Mat2 contact_jacobian;
      for (int row = 0; row < 2; ++row) {
        contact_jacobian(row, 0) = contact.force[row].gradient(0);
        contact_jacobian(row, 1) = contact.force[row].gradient(1);
      }
      system->diagonal[block] +=
          (masses[node] / (dt * dt) + constants.damping / dt) *
              Mat2::Identity() -
          contact_jacobian;
    }
  }
}

py::dict contact_result(const std::vector<Vec2>& positions,
                        const std::vector<double>& integrals,
                        const std::vector<double>& previous_errors, double dt,
                        double kp_active, double ki_active,
                        const ArcConstants& constants) {
  const ssize_t nodes = static_cast<ssize_t>(positions.size());
  py::array_t<double> force(
      py::array::ShapeContainer(std::vector<py::ssize_t>{nodes, 2}));
  py::array_t<double> new_integrals(nodes);
  py::array_t<bool> active(nodes);
  py::array_t<double> weights(nodes);
  py::array_t<double> errors(nodes);
  py::array_t<double> clearance(nodes);
  auto* force_data = static_cast<double*>(force.request().ptr);
  auto* integral_data = static_cast<double*>(new_integrals.request().ptr);
  auto* active_data = static_cast<bool*>(active.request().ptr);
  auto* weight_data = static_cast<double*>(weights.request().ptr);
  auto* error_data = static_cast<double*>(errors.request().ptr);
  auto* clearance_data = static_cast<double*>(clearance.request().ptr);
  for (ssize_t node = 0; node < nodes; ++node) {
    const std::array<Jet<1>, 2> point = {
        Jet<1>(positions[node](0)), Jet<1>(positions[node](1))};
    const ArcContactValue<1> contact = evaluate_arc_contact(
        point, integrals[node], previous_errors[node], dt, constants.arc_radius,
        constants.tube_half_width, constants.safe_distance, kp_active, ki_active,
        constants.max_pi_force, constants.max_integral_state,
        constants.leak_rate, constants.contact_smooth_length,
        constants.direction_epsilon);
    force_data[2 * node] = contact.force[0].value;
    force_data[2 * node + 1] = contact.force[1].value;
    integral_data[node] = contact.integral.value;
    active_data[node] = contact.active;
    weight_data[node] = contact.weight.value;
    error_data[node] = contact.error.value;
    clearance_data[node] = contact.clearance.value;
  }
  py::dict result;
  result["force"] = force;
  result["new_integrals"] = new_integrals;
  result["active"] = active;
  result["weight"] = weights;
  result["weights"] = weights;
  result["errors"] = errors;
  result["clearance"] = clearance;
  return result;
}

}  // namespace

Arc2DCore::Arc2DCore(const py::array& rest_lengths, const py::array& masses,
                     double ea, double ei, double damping, double arc_radius,
                     double tube_half_width, double safe_distance, double kp,
                     double ki, double max_pi_force,
                     double max_integral_state, double leak_rate,
                     double contact_smooth_length, double direction_epsilon)
    : ea_(ea),
      ei_(ei),
      damping_(damping),
      arc_radius_(arc_radius),
      tube_half_width_(tube_half_width),
      safe_distance_(safe_distance),
      kp_(kp),
      ki_(ki),
      max_pi_force_(max_pi_force),
      max_integral_state_(max_integral_state),
      leak_rate_(leak_rate),
      contact_smooth_length_(contact_smooth_length),
      direction_epsilon_(direction_epsilon) {
  const auto rest = checked_array(rest_lengths, "rest_lengths", 1);
  const auto mass = checked_array(masses, "masses", 1);
  const auto* rest_data = static_cast<const double*>(rest.ptr);
  const auto* mass_data = static_cast<const double*>(mass.ptr);
  for (const auto [value, name] : {
           std::pair<double, const char*>{ea, "ea"},
           {ei, "ei"},
           {damping, "damping"},
           {arc_radius, "arc_radius"},
           {tube_half_width, "tube_half_width"},
           {safe_distance, "safe_distance"},
           {kp, "kp"},
           {ki, "ki"},
           {max_pi_force, "max_pi_force"},
           {max_integral_state, "max_integral_state"},
           {leak_rate, "leak_rate"},
           {contact_smooth_length, "contact_smooth_length"},
           {direction_epsilon, "direction_epsilon"},
       }) {
    require_finite(value, name);
  }
  if (!(contact_smooth_length > 0.0))
    throw py::value_error("contact_smooth_length must be positive");
  if (!(direction_epsilon > 0.0))
    throw py::value_error("direction_epsilon must be positive");
  require_positive_values(rest_data, rest.shape[0], "rest_lengths");
  require_positive_values(mass_data, mass.shape[0], "masses");
  rest_lengths_.assign(rest_data, rest_data + rest.shape[0]);
  masses_.assign(mass_data, mass_data + mass.shape[0]);
  if (rest_lengths_.size() < 2 || masses_.size() != rest_lengths_.size() + 1) {
    throw py::value_error("masses must contain exactly num_nodes entries and N must be at least 3");
  }
}

py::dict Arc2DCore::contact_kernel(const py::array& positions,
                                   const py::array& integrals,
                                   const py::array& previous_errors, double dt,
                                   double kp_active, double ki_active) const {
  const auto position_info = checked_array(positions, "positions", 2);
  const auto integral_info = checked_array(integrals, "integrals", 1);
  const auto previous_info =
      checked_array(previous_errors, "previous_errors", 1);
  const ssize_t nodes = position_info.shape[0];
  if (position_info.shape[1] != 2 || integral_info.shape[0] != nodes ||
      previous_info.shape[0] != nodes ||
      nodes != static_cast<ssize_t>(masses_.size())) {
    throw py::value_error("Arc2D contact kernel shape mismatch");
  }
  const auto* position_data = static_cast<const double*>(position_info.ptr);
  const auto* integral_data = static_cast<const double*>(integral_info.ptr);
  const auto* previous_data = static_cast<const double*>(previous_info.ptr);
  require_finite(dt, "dt");
  require_finite(kp_active, "kp_active");
  require_finite(ki_active, "ki_active");
  require_finite_values(position_data, 2 * nodes, "positions");
  require_finite_values(integral_data, nodes, "integrals");
  require_finite_values(previous_data, nodes, "previous_errors");
  std::vector<Vec2> position_values(nodes);
  std::vector<double> integral_values(nodes), previous_values(nodes);
  for (ssize_t node = 0; node < nodes; ++node) {
    position_values[node] = Vec2(position_data[2 * node], position_data[2 * node + 1]);
    integral_values[node] = integral_data[node];
    previous_values[node] = previous_data[node];
  }
  const ArcConstants constants{ea_, ei_, damping_, arc_radius_, tube_half_width_,
                               safe_distance_, max_pi_force_,
                               max_integral_state_, leak_rate_,
                               contact_smooth_length_, direction_epsilon_};
  return contact_result(position_values, integral_values, previous_values, dt,
                        kp_active, ki_active, constants);
}

py::dict Arc2DCore::implicit_euler_step(
    const py::array& positions, const py::array& velocities,
    const py::array& integrals, const py::array& previous_errors,
    const py::array& prescribed, const py::array& pulse_forces,
    const py::array& reference_curvature, bool tip_constrained, double dt,
    int max_newton_iters, double residual_tol, double step_tol,
    double kp_active, double ki_active) const {
  const auto position_info = checked_array(positions, "positions", 2);
  const auto velocity_info = checked_array(velocities, "velocities", 2);
  const auto integral_info = checked_array(integrals, "integrals", 1);
  const auto previous_info =
      checked_array(previous_errors, "previous_errors", 1);
  const auto prescribed_info = checked_array(prescribed, "prescribed", 2);
  const auto pulse_info = checked_array(pulse_forces, "pulse_forces", 2);
  const auto curvature_info =
      checked_array(reference_curvature, "reference_curvature", 1);
  const ssize_t nodes = position_info.shape[0];
  if (nodes != static_cast<ssize_t>(masses_.size()) || nodes < 3 ||
      position_info.shape[1] != 2 || velocity_info.shape[0] != nodes ||
      velocity_info.shape[1] != 2 || integral_info.shape[0] != nodes ||
      previous_info.shape[0] != nodes || prescribed_info.shape[0] != nodes ||
      prescribed_info.shape[1] != 2 || pulse_info.shape[0] != nodes ||
      pulse_info.shape[1] != 2 || curvature_info.shape[0] != nodes - 2) {
    throw py::value_error("Arc2D implicit step shape mismatch");
  }
  require_finite(dt, "dt");
  require_finite(kp_active, "kp_active");
  require_finite(ki_active, "ki_active");
  if (!(dt > 0.0) || max_newton_iters <= 0) {
    throw py::value_error("dt must be positive and max_newton_iters must be positive");
  }
  const int first_free = 2;
  const int last_free = static_cast<int>(nodes) - (tip_constrained ? 1 : 0);
  const int free_count = last_free - first_free;
  if (free_count < 0) throw py::value_error("invalid Arc2D prescribed topology");

  const auto* position_data = static_cast<const double*>(position_info.ptr);
  const auto* velocity_data = static_cast<const double*>(velocity_info.ptr);
  const auto* integral_data = static_cast<const double*>(integral_info.ptr);
  const auto* previous_data = static_cast<const double*>(previous_info.ptr);
  const auto* prescribed_data = static_cast<const double*>(prescribed_info.ptr);
  const auto* pulse_data = static_cast<const double*>(pulse_info.ptr);
  const auto* curvature_data = static_cast<const double*>(curvature_info.ptr);
  require_finite_values(position_data, 2 * nodes, "positions");
  require_finite_values(velocity_data, 2 * nodes, "velocities");
  require_finite_values(integral_data, nodes, "integrals");
  require_finite_values(previous_data, nodes, "previous_errors");
  require_finite_values(prescribed_data, 2 * nodes, "prescribed");
  require_finite_values(pulse_data, 2 * nodes, "pulse_forces");
  require_finite_values(curvature_data, nodes - 2, "reference_curvature");
  std::vector<Vec2> old_positions(nodes), old_velocities(nodes),
      prescribed_positions(nodes), pulse_values(nodes);
  std::vector<double> integral_values(nodes), previous_values(nodes),
      curvature_values(nodes - 2);
  Eigen::VectorXd q(2 * free_count);
  for (int node = 0; node < nodes; ++node) {
    old_positions[node] = Vec2(position_data[2 * node], position_data[2 * node + 1]);
    old_velocities[node] = Vec2(velocity_data[2 * node], velocity_data[2 * node + 1]);
    prescribed_positions[node] =
        Vec2(prescribed_data[2 * node], prescribed_data[2 * node + 1]);
    pulse_values[node] = Vec2(pulse_data[2 * node], pulse_data[2 * node + 1]);
    integral_values[node] = integral_data[node];
    previous_values[node] = previous_data[node];
    if (node >= first_free && node < last_free) {
      q.segment<2>(2 * (node - first_free)) =
          old_positions[node] + dt * old_velocities[node];
    }
  }
  for (int vertex = 0; vertex < nodes - 2; ++vertex)
    curvature_values[vertex] = curvature_data[vertex];

  const ArcConstants constants{ea_, ei_, damping_, arc_radius_, tube_half_width_,
                               safe_distance_, max_pi_force_,
                               max_integral_state_, leak_rate_,
                               contact_smooth_length_, direction_epsilon_};
  int failures = 0;
  int dense_fallback_count = 0;
  int iteration = 0;
  double residual_norm = std::numeric_limits<double>::infinity();
  double step_norm = std::numeric_limits<double>::infinity();
  bool converged = false;
  for (iteration = 1; iteration <= max_newton_iters; ++iteration) {
    Eigen::VectorXd current;
    BlockPenta2D ignored_system(free_count);
    assemble_residual(q, old_positions, old_velocities, prescribed_positions,
                      pulse_values, rest_lengths_, masses_, integral_values,
                      previous_values, curvature_values, first_free, last_free,
                      dt, kp_active, ki_active, constants, false, &current,
                      &ignored_system);
    residual_norm = current.norm();
    if (residual_norm < residual_tol) {
      converged = true;
      step_norm = 0.0;
      break;
    }
    Eigen::VectorXd residual;
    BlockPenta2D system(free_count);
    assemble_residual(q, old_positions, old_velocities, prescribed_positions,
                      pulse_values, rest_lengths_, masses_, integral_values,
                      previous_values, curvature_values, first_free, last_free,
                      dt, kp_active, ki_active, constants, true, &residual,
                      &system);
    Eigen::VectorXd delta;
    double relative_residual = std::numeric_limits<double>::infinity();
    if (!system.solve(-residual, &delta, &relative_residual)) {
      ++dense_fallback_count;
      const Eigen::MatrixXd dense = system.dense();
      delta = dense.partialPivLu().solve(-residual);
      const Eigen::VectorXd dense_difference = dense * delta + residual;
      const double dense_relative = dense_difference.norm() /
                                    std::max(residual.norm(), 1.0e-30);
      if (!delta.allFinite() || !std::isfinite(dense_relative) ||
          dense_relative > 1.0e-8) {
        throw std::runtime_error("Arc2D Newton linear solve failed");
      }
    }
    step_norm = delta.norm();
    const double base_norm = residual_norm;
    bool accepted = false;
    for (int exponent = 0; exponent < 10; ++exponent) {
      const Eigen::VectorXd candidate = q + std::ldexp(1.0, -exponent) * delta;
      Eigen::VectorXd candidate_residual;
      BlockPenta2D ignored_candidate_system(free_count);
      assemble_residual(candidate, old_positions, old_velocities,
                        prescribed_positions, pulse_values, rest_lengths_,
                        masses_, integral_values, previous_values,
                        curvature_values, first_free, last_free, dt, kp_active,
                        ki_active, constants, false, &candidate_residual,
                        &ignored_candidate_system);
      const double candidate_norm = candidate_residual.norm();
      if (std::isfinite(candidate_norm) && candidate_norm < base_norm) {
        q = candidate;
        residual_norm = candidate_norm;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      ++failures;
      q += 0.05 * delta;
    }
    if (step_norm < step_tol) {
      converged = residual_norm < 10.0 * residual_tol;
      break;
    }
  }
  if (iteration > max_newton_iters) iteration = max_newton_iters;

  std::vector<Vec2> output_positions = prescribed_positions;
  std::vector<Vec2> output_velocities(nodes, Vec2::Zero());
  for (int node = first_free; node < last_free; ++node) {
    output_positions[node] = q.segment<2>(2 * (node - first_free));
    output_velocities[node] = (output_positions[node] - old_positions[node]) / dt;
  }
  py::dict contact = contact_result(output_positions, integral_values,
                                    previous_values, dt, kp_active, ki_active,
                                    constants);
  py::array_t<double> output_position_array(
      py::array::ShapeContainer(std::vector<py::ssize_t>{nodes, 2}));
  py::array_t<double> output_velocity_array(
      py::array::ShapeContainer(std::vector<py::ssize_t>{nodes, 2}));
  auto* output_position_data =
      static_cast<double*>(output_position_array.request().ptr);
  auto* output_velocity_data =
      static_cast<double*>(output_velocity_array.request().ptr);
  for (ssize_t node = 0; node < nodes; ++node) {
    output_position_data[2 * node] = output_positions[node](0);
    output_position_data[2 * node + 1] = output_positions[node](1);
    output_velocity_data[2 * node] = output_velocities[node](0);
    output_velocity_data[2 * node + 1] = output_velocities[node](1);
  }
  py::dict result;
  result["positions"] = output_position_array;
  result["velocities"] = output_velocity_array;
  result["integrals"] = contact["new_integrals"];
  result["contact_force"] = contact["force"];
  result["contact_active"] = contact["active"];
  result["contact_weight"] = contact["weights"];
  result["contact_error"] = contact["errors"];
  result["contact_clearance"] = contact["clearance"];
  result["clearance"] = contact["clearance"];
  result["previous_errors"] = contact["errors"];
  result["converged"] = converged;
  result["iterations"] = iteration;
  result["residual_norm"] = residual_norm;
  result["step_norm"] = step_norm;
  result["line_search_failures"] = failures;
  result["dense_fallback_count"] = dense_fallback_count;
  result["dense_build_count"] = dense_fallback_count;
  return result;
}

}  // namespace der
