#include "der_core.h"
#include "der_bending.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace der {
namespace {

template <int D>
struct Jet {
  double value = 0.0;
  Eigen::Matrix<double, D, 1> gradient = Eigen::Matrix<double, D, 1>::Zero();
  Eigen::Matrix<double, D, D> hessian = Eigen::Matrix<double, D, D>::Zero();

  Jet() = default;
  explicit Jet(double x) : value(x) {}

  static Jet variable(double x, int index) {
    Jet result(x);
    result.gradient(index) = 1.0;
    return result;
  }
};

template <int D>
Jet<D> operator+(const Jet<D>& a, const Jet<D>& b) {
  Jet<D> r;
  r.value = a.value + b.value;
  r.gradient = a.gradient + b.gradient;
  r.hessian = a.hessian + b.hessian;
  return r;
}

template <int D>
Jet<D> operator-(const Jet<D>& a, const Jet<D>& b) {
  Jet<D> r;
  r.value = a.value - b.value;
  r.gradient = a.gradient - b.gradient;
  r.hessian = a.hessian - b.hessian;
  return r;
}

template <int D>
Jet<D> operator-(const Jet<D>& a) {
  Jet<D> r;
  r.value = -a.value;
  r.gradient = -a.gradient;
  r.hessian = -a.hessian;
  return r;
}

template <int D>
Jet<D> operator*(const Jet<D>& a, const Jet<D>& b) {
  Jet<D> r;
  r.value = a.value * b.value;
  r.gradient = b.value * a.gradient + a.value * b.gradient;
  r.hessian = b.value * a.hessian + a.value * b.hessian +
              a.gradient * b.gradient.transpose() +
              b.gradient * a.gradient.transpose();
  return r;
}

template <int D>
Jet<D> operator/(const Jet<D>& a, const Jet<D>& b);

template <int D>
Jet<D> unary_compose(const Jet<D>& x, double value, double first,
                    double second) {
  Jet<D> r;
  r.value = value;
  r.gradient = first * x.gradient;
  r.hessian = first * x.hessian + second * x.gradient * x.gradient.transpose();
  return r;
}

template <int D>
Jet<D> jet_sqrt(const Jet<D>& x) {
  const double safe = std::max(x.value, 1.0e-24);
  const double root = std::sqrt(safe);
  if (x.value <= 1.0e-24) return Jet<D>(root);
  return unary_compose(x, root, 0.5 / root, -0.25 / (root * root * root));
}

template <int D>
Jet<D> operator/(const Jet<D>& a, const Jet<D>& b) {
  const double safe = std::max(std::abs(b.value), 1.0e-24);
  const double signed_value = b.value >= 0.0 ? safe : -safe;
  Jet<D> reciprocal = unary_compose(
      b, 1.0 / signed_value, -1.0 / (signed_value * signed_value),
      2.0 / (signed_value * signed_value * signed_value));
  return a * reciprocal;
}

template <int D>
Jet<D> operator+(double a, const Jet<D>& b) { return Jet<D>(a) + b; }
template <int D>
Jet<D> operator+(const Jet<D>& a, double b) { return a + Jet<D>(b); }
template <int D>
Jet<D> operator-(double a, const Jet<D>& b) { return Jet<D>(a) - b; }
template <int D>
Jet<D> operator-(const Jet<D>& a, double b) { return a - Jet<D>(b); }
template <int D>
Jet<D> operator*(double a, const Jet<D>& b) { return Jet<D>(a) * b; }
template <int D>
Jet<D> operator*(const Jet<D>& a, double b) { return a * Jet<D>(b); }
template <int D>
Jet<D> operator/(const Jet<D>& a, double b) { return a / Jet<D>(b); }

template <int D>
Jet<D> clamp_min(const Jet<D>& x, double minimum) {
  return x.value < minimum ? Jet<D>(minimum) : x;
}

template <int D>
struct FirstJet {
  double value = 0.0;
  Eigen::Matrix<double, D, 1> gradient = Eigen::Matrix<double, D, 1>::Zero();

  FirstJet() = default;
  explicit FirstJet(double x) : value(x) {}
  static FirstJet variable(double x, int index) {
    FirstJet result(x);
    result.gradient(index) = 1.0;
    return result;
  }
};

template <int D>
FirstJet<D> operator+(const FirstJet<D>& a, const FirstJet<D>& b) {
  FirstJet<D> r;
  r.value = a.value + b.value;
  r.gradient = a.gradient + b.gradient;
  return r;
}
template <int D>
FirstJet<D> operator-(const FirstJet<D>& a, const FirstJet<D>& b) {
  FirstJet<D> r;
  r.value = a.value - b.value;
  r.gradient = a.gradient - b.gradient;
  return r;
}
template <int D>
FirstJet<D> operator-(const FirstJet<D>& a) {
  FirstJet<D> r;
  r.value = -a.value;
  r.gradient = -a.gradient;
  return r;
}
template <int D>
FirstJet<D> operator*(const FirstJet<D>& a, const FirstJet<D>& b) {
  FirstJet<D> r;
  r.value = a.value * b.value;
  r.gradient = b.value * a.gradient + a.value * b.gradient;
  return r;
}
template <int D>
FirstJet<D> first_compose(const FirstJet<D>& x, double value, double first) {
  FirstJet<D> r;
  r.value = value;
  r.gradient = first * x.gradient;
  return r;
}
template <int D>
FirstJet<D> first_sqrt(const FirstJet<D>& x) {
  const double safe = std::max(x.value, 1.0e-24);
  const double root = std::sqrt(safe);
  if (x.value <= 1.0e-24) return FirstJet<D>(root);
  return first_compose(x, root, 0.5 / root);
}
template <int D>
FirstJet<D> operator/(const FirstJet<D>& a, const FirstJet<D>& b) {
  const double safe = std::max(std::abs(b.value), 1.0e-24);
  const double signed_value = b.value >= 0.0 ? safe : -safe;
  return a * first_compose(b, 1.0 / signed_value,
                           -1.0 / (signed_value * signed_value));
}
template <int D>
FirstJet<D> operator+(double a, const FirstJet<D>& b) { return FirstJet<D>(a) + b; }
template <int D>
FirstJet<D> operator+(const FirstJet<D>& a, double b) { return a + FirstJet<D>(b); }
template <int D>
FirstJet<D> operator-(double a, const FirstJet<D>& b) { return FirstJet<D>(a) - b; }
template <int D>
FirstJet<D> operator-(const FirstJet<D>& a, double b) { return a - FirstJet<D>(b); }
template <int D>
FirstJet<D> operator*(double a, const FirstJet<D>& b) { return FirstJet<D>(a) * b; }
template <int D>
FirstJet<D> operator*(const FirstJet<D>& a, double b) { return a * FirstJet<D>(b); }
template <int D>
FirstJet<D> operator/(const FirstJet<D>& a, double b) { return a / FirstJet<D>(b); }
template <int D>
FirstJet<D> first_clamp_min(const FirstJet<D>& x, double minimum) {
  return x.value < minimum ? FirstJet<D>(minimum) : x;
}
template <int D, typename A, typename B>
FirstJet<D> first_dot3(const Eigen::MatrixBase<A>& a,
                       const Eigen::MatrixBase<B>& b) {
  return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}
template <int D, typename A, typename B>
Eigen::Matrix<FirstJet<D>, 3, 1> first_cross3(
    const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
  Eigen::Matrix<FirstJet<D>, 3, 1> r;
  r(0) = a(1) * b(2) - a(2) * b(1);
  r(1) = a(2) * b(0) - a(0) * b(2);
  r(2) = a(0) * b(1) - a(1) * b(0);
  return r;
}

template <int D>
FirstJet<D> first_bending_energy(const Eigen::Matrix<FirstJet<D>, 9, 1>& x,
                                 double left_rest, double right_rest,
                                 double ei) {
  Eigen::Matrix<FirstJet<D>, 3, 1> p0, p1, p2;
  for (int k = 0; k < 3; ++k) {
    p0(k) = x(k); p1(k) = x(3 + k); p2(k) = x(6 + k);
  }
  const auto left = p1 - p0;
  const auto right = p2 - p1;
  const auto t0 = left / first_sqrt(first_clamp_min(first_dot3<D>(left, left), 1.0e-24));
  const auto t1 = right / first_sqrt(first_clamp_min(first_dot3<D>(right, right), 1.0e-24));
  const auto cross = first_cross3<D>(t0, t1);
  const auto denominator = first_clamp_min(1.0 + first_dot3<D>(t0, t1), 1.0e-12);
  Eigen::Matrix<FirstJet<D>, 3, 1> curvature;
  for (int k = 0; k < 3; ++k) curvature(k) = 2.0 * cross(k) / denominator;
  const double voronoi = 0.5 * (left_rest + right_rest);
  return 0.5 * ei * first_dot3<D>(curvature, curvature) / voronoi;
}

template <int D, typename A, typename B>
Jet<D> dot3(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
  return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}

template <int D, typename A, typename B>
Eigen::Matrix<Jet<D>, 3, 1> cross3(const Eigen::MatrixBase<A>& a,
                                   const Eigen::MatrixBase<B>& b) {
  Eigen::Matrix<Jet<D>, 3, 1> r;
  r(0) = a(1) * b(2) - a(2) * b(1);
  r(1) = a(2) * b(0) - a(0) * b(2);
  r(2) = a(0) * b(1) - a(1) * b(0);
  return r;
}

template <int D>
Jet<D> bending_energy(const Eigen::Matrix<Jet<D>, 9, 1>& x, double left_rest,
                      double right_rest, double ei) {
  Eigen::Matrix<Jet<D>, 3, 1> p0, p1, p2;
  for (int k = 0; k < 3; ++k) {
    p0(k) = x(k);
    p1(k) = x(3 + k);
    p2(k) = x(6 + k);
  }
  const auto left = p1 - p0;
  const auto right = p2 - p1;
  const auto t0 = left / jet_sqrt(clamp_min(dot3<D>(left, left), 1.0e-24));
  const auto t1 = right / jet_sqrt(clamp_min(dot3<D>(right, right), 1.0e-24));
  const auto cross = cross3<D>(t0, t1);
  const auto denominator = clamp_min(1.0 + dot3<D>(t0, t1), 1.0e-12);
  Eigen::Matrix<Jet<D>, 3, 1> curvature;
  for (int k = 0; k < 3; ++k) curvature(k) = 2.0 * cross(k) / denominator;
  const double voronoi = 0.5 * (left_rest + right_rest);
  return 0.5 * ei * dot3<D>(curvature, curvature) / voronoi;
}

struct ArrayInfo {
  py::buffer_info info;
};

py::buffer_info checked_array(const py::array& array, const std::string& name,
                              int ndim) {
  auto info = array.request();
  if (info.ndim != ndim || info.format != py::format_descriptor<double>::format() ||
      info.itemsize != static_cast<ssize_t>(sizeof(double))) {
    throw py::type_error(name + " must be a contiguous float64 array");
  }
  if (!(array.flags() & py::array::c_style)) {
    throw py::value_error(name + " must be C-contiguous");
  }
  return std::move(info);
}

double softplus(double x, double beta) {
  const double bx = beta * x;
  if (bx > 40.0) return x;
  if (bx < -40.0) return std::exp(bx) / beta;
  return std::log1p(std::exp(bx)) / beta;
}

double sigmoid(double x) {
  if (x >= 0.0) {
    const double e = std::exp(-x);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(x);
  return e / (1.0 + e);
}

struct ContactValue {
  Vec3 force = Vec3::Zero();
  Mat3 jacobian = Mat3::Zero();
  double new_integral = 0.0;
  double error = 0.0;
  double weight = 0.0;
  bool active = false;
  double distance = 0.0;
};

struct MeshContactValue {
  ContactValue value;
  std::int64_t triangle_id = -1;
};

int nearest_mesh_face(const Vec3& point, const MeshBVH& mesh_bvh) {
  return mesh_bvh.nearest_face(point);
}

MeshContactValue evaluate_mesh_contact(
    const Vec3& point, double old_integral, double previous_error, double dt,
    double kp, double ki_active, double max_pi_force, double leak_rate,
    double max_integral_state, double smooth_length,
    const std::vector<Vec3>& triangles, const std::vector<Vec3>& normals,
    const std::vector<std::int64_t>& triangle_ids, const MeshBVH& mesh_bvh,
    int fixed_face = -1) {
  MeshContactValue output;
  if (triangles.empty()) return output;
  // Newton iterations deliberately retain the predicted face.  Only initial
  // and standalone contact searches use the immutable BVH.
  const int best = fixed_face >= 0 ? fixed_face : nearest_mesh_face(point, mesh_bvh);
  if (best < 0) return output;
  const Vec3 normal = normals[best];
  const double clearance = normal.dot(point - triangles[3 * best]);
  const double signed_penetration = -clearance;
  const bool active = signed_penetration > 0.0;
  const double beta = 1.0 / smooth_length;
  const double shifted = softplus(signed_penetration, beta) - std::log(2.0) / beta;
  const double error = active ? shifted : 0.0;
  const double error_prime = active ? sigmoid(beta * signed_penetration) : 0.0;
  const double penetration = std::max(signed_penetration, 0.0);
  const double activation = active ? -std::expm1(-penetration / smooth_length) : 0.0;
  const double activation_prime = active ? std::exp(-penetration / smooth_length) / smooth_length : 0.0;
  const double decay = std::exp(-leak_rate * dt);
  const double accumulated = old_integral + error * dt;
  const double leaked = old_integral * decay;
  const double candidate_raw = activation * accumulated + (1.0 - activation) * leaked;
  const double candidate = std::min(std::max(candidate_raw, 0.0), max_integral_state);
  const bool candidate_inside = candidate_raw >= 0.0 && candidate_raw <= max_integral_state;
  const double candidate_prime_raw = activation_prime * (accumulated - leaked) + activation * error_prime * dt;
  const double candidate_prime = candidate_inside ? candidate_prime_raw : 0.0;
  const double unsaturated = kp * error + ki_active * candidate;
  const bool stop_growth =
      (ki_active > 0.0 && unsaturated >= max_pi_force && candidate > old_integral) ||
      (ki_active > 0.0 && error < previous_error && ki_active * old_integral > kp * error && candidate > old_integral);
  const double integral = stop_growth ? old_integral : candidate;
  const double integral_prime = stop_growth ? 0.0 : candidate_prime;
  const double raw_force = kp * error + ki_active * integral;
  const double clamped_force = std::min(std::max(raw_force, 0.0), max_pi_force);
  const bool force_inside = raw_force >= 0.0 && raw_force <= max_pi_force;
  const double raw_prime = kp * error_prime + ki_active * integral_prime;
  const double magnitude_prime = activation_prime * clamped_force +
      activation * (force_inside ? raw_prime : 0.0);
  output.value.force = Vec3::Zero();
  if (active) output.value.force = normal * (activation * clamped_force);
  output.value.jacobian = -magnitude_prime * (normal * normal.transpose());
  output.value.new_integral = integral;
  output.value.error = error;
  output.value.weight = activation;
  output.value.active = active;
  output.value.distance = clearance;
  output.triangle_id = triangle_ids[best];
  return output;
}

struct BlockPentaSystem {
  explicit BlockPentaSystem(int count)
      : diagonal(count, Mat3::Zero()),
        upper1(std::max(count - 1, 0), Mat3::Zero()),
        lower1(std::max(count - 1, 0), Mat3::Zero()),
        upper2(std::max(count - 2, 0), Mat3::Zero()),
        lower2(std::max(count - 2, 0), Mat3::Zero()) {}

  int size() const { return static_cast<int>(diagonal.size()); }

  void reset() {
    std::fill(diagonal.begin(), diagonal.end(), Mat3::Zero());
    std::fill(upper1.begin(), upper1.end(), Mat3::Zero());
    std::fill(lower1.begin(), lower1.end(), Mat3::Zero());
    std::fill(upper2.begin(), upper2.end(), Mat3::Zero());
    std::fill(lower2.begin(), lower2.end(), Mat3::Zero());
  }

  void copy_values_from(const BlockPentaSystem& other) {
    if (size() != other.size())
      throw std::logic_error("block-pentadiagonal scratch size mismatch");
    std::copy(other.diagonal.begin(), other.diagonal.end(), diagonal.begin());
    std::copy(other.upper1.begin(), other.upper1.end(), upper1.begin());
    std::copy(other.lower1.begin(), other.lower1.end(), lower1.begin());
    std::copy(other.upper2.begin(), other.upper2.end(), upper2.begin());
    std::copy(other.lower2.begin(), other.lower2.end(), lower2.begin());
  }

  void add_block(int row, int col, const Mat3& value) {
    const int offset = col - row;
    if (offset == 0) diagonal[row] += value;
    else if (offset == 1) upper1[row] += value;
    else if (offset == -1) lower1[col] += value;
    else if (offset == 2) upper2[row] += value;
    else if (offset == -2) lower2[col] += value;
    else if (value.cwiseAbs().maxCoeff() > 1.0e-14)
      throw std::logic_error("DER Jacobian exceeded block-pentadiagonal bandwidth");
  }

  void matvec(const Eigen::VectorXd& x, Eigen::VectorXd* y) const {
    y->setZero();
    for (int i = 0; i < size(); ++i) {
      y->segment<3>(3 * i) += diagonal[i] * x.segment<3>(3 * i);
      if (i + 1 < size()) {
        y->segment<3>(3 * i) += upper1[i] * x.segment<3>(3 * (i + 1));
        y->segment<3>(3 * (i + 1)) += lower1[i] * x.segment<3>(3 * i);
      }
      if (i + 2 < size()) {
        y->segment<3>(3 * i) += upper2[i] * x.segment<3>(3 * (i + 2));
        y->segment<3>(3 * (i + 2)) += lower2[i] * x.segment<3>(3 * i);
      }
    }
  }

  Eigen::MatrixXd dense_reference() const {
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(3 * size(), 3 * size());
    for (int i = 0; i < size(); ++i) {
      dense.block<3, 3>(3 * i, 3 * i) = diagonal[i];
      if (i + 1 < size()) {
        dense.block<3, 3>(3 * i, 3 * (i + 1)) = upper1[i];
        dense.block<3, 3>(3 * (i + 1), 3 * i) = lower1[i];
      }
      if (i + 2 < size()) {
        dense.block<3, 3>(3 * i, 3 * (i + 2)) = upper2[i];
        dense.block<3, 3>(3 * (i + 2), 3 * i) = lower2[i];
      }
    }
    return dense;
  }

  std::vector<Mat3> diagonal, upper1, lower1, upper2, lower2;
};

struct SolveResult {
  bool used_dense_fallback = false;
  double relative_residual = std::numeric_limits<double>::infinity();
};

struct BlockPentaSolveScratch {
  explicit BlockPentaSolveScratch(int blocks)
      : factor_system(blocks), rhs_blocks(blocks, 3), matvec(3 * blocks) {}

  BlockPentaSystem factor_system;
  Eigen::Matrix<double, Eigen::Dynamic, 3> rhs_blocks;
  Eigen::VectorXd matvec;
};

SolveResult block_penta_solve(const BlockPentaSystem& system,
                               const Eigen::VectorXd& rhs,
                               bool force_dense,
                               BlockPentaSolveScratch* scratch,
                               Eigen::VectorXd* solution) {
  const int blocks = system.size();
  scratch->factor_system.copy_values_from(system);
  auto& diagonal = scratch->factor_system.diagonal;
  auto& upper1 = scratch->factor_system.upper1;
  auto& upper2 = scratch->factor_system.upper2;
  auto& lower1 = scratch->factor_system.lower1;
  auto& lower2 = scratch->factor_system.lower2;
  auto& vector = scratch->rhs_blocks;
  for (int i = 0; i < blocks; ++i)
    vector.row(i) = rhs.segment<3>(3 * i).transpose();
  bool failed = force_dense;
  for (int i = 0; i < blocks && !failed; ++i) {
    if (!diagonal[i].allFinite()) {
      failed = true;
      break;
    }
    const Eigen::PartialPivLU<Mat3> factorization(diagonal[i]);
    if (std::abs(diagonal[i].determinant()) < 1.0e-24) {
      failed = true;
      break;
    }
    if (i + 1 < blocks) {
      const Mat3 factor = lower1[i] * factorization.solve(Mat3::Identity());
      diagonal[i + 1] -= factor * upper1[i];
      vector.row(i + 1) -= (factor * vector.row(i).transpose()).transpose();
      if (i + 2 < blocks) upper1[i + 1] -= factor * upper2[i];
    }
    if (i + 2 < blocks) {
      const Mat3 factor = lower2[i] * factorization.solve(Mat3::Identity());
      lower1[i + 1] -= factor * upper1[i];
      diagonal[i + 2] -= factor * upper2[i];
      vector.row(i + 2) -= (factor * vector.row(i).transpose()).transpose();
    }
  }
  solution->setZero();
  if (!failed) {
    for (int i = blocks - 1; i >= 0; --i) {
      Eigen::Vector3d value = vector.row(i).transpose();
      if (i + 1 < blocks) value -= upper1[i] * solution->segment<3>(3 * (i + 1));
      if (i + 2 < blocks) value -= upper2[i] * solution->segment<3>(3 * (i + 2));
      solution->segment<3>(3 * i) = diagonal[i].partialPivLu().solve(value);
      if (!solution->segment<3>(3 * i).allFinite()) {
        failed = true;
        break;
      }
    }
  }
  SolveResult result;
  if (!failed) {
    system.matvec(*solution, &scratch->matvec);
    scratch->matvec -= rhs;
    result.relative_residual = scratch->matvec.norm() /
                               std::max(rhs.norm(), 1.0e-30);
  }
  if (!failed && result.relative_residual <= 1.0e-8) return result;
  result.used_dense_fallback = true;
  const Eigen::MatrixXd dense = system.dense_reference();
  *solution = dense.partialPivLu().solve(rhs);
  result.relative_residual =
      (dense * *solution - rhs).norm() / std::max(rhs.norm(), 1.0e-30);
  return result;
}

struct SolverWorkspace {
  SolverWorkspace(int node_count, int free_node_count)
      : old_q(3 * free_node_count),
        q(3 * free_node_count),
        prescribed(3 * node_count),
        old_full(3 * node_count),
        velocity_full(3 * node_count),
        pulse_full(3 * node_count),
        previous_error(node_count),
        full(3 * node_count),
        full_result(3 * node_count),
        gradient(3 * node_count),
        contact_force(3 * node_count),
        contact_jacobian(node_count, Mat3::Zero()),
        residual(3 * free_node_count),
        trial_residual(3 * free_node_count),
        trial(3 * free_node_count),
        delta(3 * free_node_count),
        solve_rhs(3 * free_node_count),
        new_velocity(3 * node_count),
        jacobian(free_node_count),
        solve_scratch(free_node_count),
        mesh_frozen_faces(node_count, -1) {}

  void reset_assembly(bool with_jacobian) {
    gradient.setZero();
    contact_force.setZero();
    if (with_jacobian)
      std::fill(contact_jacobian.begin(), contact_jacobian.end(), Mat3::Zero());
  }

  Eigen::VectorXd old_q, q, prescribed;
  Eigen::VectorXd old_full, velocity_full, pulse_full, previous_error;
  Eigen::VectorXd full, full_result, gradient, contact_force;
  std::vector<Mat3> contact_jacobian;
  Eigen::VectorXd residual, trial_residual, trial, delta, solve_rhs;
  Eigen::VectorXd new_velocity;
  BlockPentaSystem jacobian;
  BlockPentaSolveScratch solve_scratch;
  std::vector<int> mesh_frozen_faces;
};

}  // namespace

DERCore::DERCore(const py::array& rest_lengths, const py::array& masses,
                 double ea, double ei, double damping, double tube_radius,
                 double safe_distance, double kp, double ki,
                 double max_pi_force, double leak_rate,
                 double max_integral_state, double contact_smooth_length,
                 double direction_epsilon)
    : ea_(ea),
      ei_(ei),
      damping_(damping),
      tube_radius_(tube_radius),
      safe_distance_(safe_distance),
      kp_(kp),
      ki_(ki),
      max_pi_force_(max_pi_force),
      leak_rate_(leak_rate),
      max_integral_state_(max_integral_state),
      contact_smooth_length_(contact_smooth_length),
      direction_epsilon_(direction_epsilon) {
  const auto rest = checked_array(rest_lengths, "rest_lengths", 1);
  const auto mass = checked_array(masses, "masses", 1);
  rest_lengths_.assign(static_cast<double*>(rest.ptr),
                       static_cast<double*>(rest.ptr) + rest.shape[0]);
  masses_.assign(static_cast<double*>(mass.ptr),
                 static_cast<double*>(mass.ptr) + mass.shape[0]);
  if (rest_lengths_.empty() || masses_.size() != rest_lengths_.size() + 1) {
    throw py::value_error("masses must contain exactly num_nodes entries");
  }
}

void DERCore::configure_mesh(const py::array& triangles, const py::array& normals,
                             const py::array& triangle_ids) {
  const auto tri = triangles.request();
  const auto nor = normals.request();
  py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ids_array(triangle_ids);
  const auto ids = ids_array.request();
  if (tri.ndim != 3 || tri.shape[1] != 3 || tri.shape[2] != 3 ||
      tri.format != py::format_descriptor<double>::format() ||
      tri.itemsize != static_cast<ssize_t>(sizeof(double)) ||
      !(triangles.flags() & py::array::c_style)) {
    throw py::type_error("triangles must be a contiguous float64 array of shape (M,3,3)");
  }
  if (nor.ndim != 2 || nor.shape[0] != tri.shape[0] || nor.shape[1] != 3 ||
      nor.format != py::format_descriptor<double>::format() ||
      nor.itemsize != static_cast<ssize_t>(sizeof(double)) ||
      !(normals.flags() & py::array::c_style)) {
    throw py::type_error("normals must be a contiguous float64 array of shape (M,3)");
  }
  if (ids.ndim != 1 || ids.shape[0] != tri.shape[0]) {
    throw py::type_error("triangle_ids must be an int64 array of shape (M,)");
  }
  const auto* tri_ptr = static_cast<const double*>(tri.ptr);
  const auto* nor_ptr = static_cast<const double*>(nor.ptr);
  const auto* id_ptr = static_cast<const std::int64_t*>(ids.ptr);
  mesh_triangles_.resize(static_cast<std::size_t>(tri.shape[0]) * 3);
  mesh_normals_.resize(static_cast<std::size_t>(tri.shape[0]));
  mesh_triangle_ids_.resize(static_cast<std::size_t>(tri.shape[0]));
  for (ssize_t face = 0; face < tri.shape[0]; ++face) {
    for (int vertex = 0; vertex < 3; ++vertex) {
      mesh_triangles_[3 * face + vertex] = Vec3(
          tri_ptr[face * 9 + vertex * 3], tri_ptr[face * 9 + vertex * 3 + 1],
          tri_ptr[face * 9 + vertex * 3 + 2]);
      if (!mesh_triangles_[3 * face + vertex].allFinite())
        throw py::value_error("triangles must contain only finite values");
    }
    mesh_normals_[face] = Vec3(nor_ptr[face * 3], nor_ptr[face * 3 + 1], nor_ptr[face * 3 + 2]);
    const double normal_length = mesh_normals_[face].norm();
    if (!std::isfinite(normal_length) || normal_length < 1.0e-12)
      throw py::value_error("normals must be finite and non-degenerate");
    mesh_normals_[face] /= normal_length;
    mesh_triangle_ids_[face] = id_ptr[face];
  }
  mesh_bvh_.build(mesh_triangles_);
  mesh_enabled_ = !mesh_normals_.empty();
}

py::tuple DERCore::stretch_kernel(const py::array& positions,
                                  const py::array& rest_lengths) const {
  const auto p = checked_array(positions, "positions", 2);
  const auto r = checked_array(rest_lengths, "rest_lengths", 1);
  if (p.shape[1] != 3 || r.shape[0] != p.shape[0] - 1) {
    throw py::value_error("stretch kernel shape mismatch");
  }
  const auto* x = static_cast<const double*>(p.ptr);
  const auto* l0 = static_cast<const double*>(r.ptr);
  const ssize_t elements = p.shape[0] - 1;
  py::array_t<double> gradient(py::array::ShapeContainer(
      std::vector<py::ssize_t>{elements, 6}));
  py::array_t<double> hessian(py::array::ShapeContainer(
      std::vector<py::ssize_t>{elements, 6, 6}));
  auto* g = static_cast<double*>(gradient.request().ptr);
  auto* h = static_cast<double*>(hessian.request().ptr);
  for (ssize_t i = 0; i < elements; ++i) {
    Vec3 d;
    for (int k = 0; k < 3; ++k) d(k) = x[3 * (i + 1) + k] - x[3 * i + k];
    const double length = std::max(d.norm(), 1.0e-12);
    const double kspring = ea_ / l0[i];
    const double coeff = 1.0 - l0[i] / length;
    const Vec3 gd = kspring * coeff * d;
    const Mat3 k = kspring *
                   (coeff * Mat3::Identity() +
                    (l0[i] / std::pow(length, 3)) * (d * d.transpose()));
    for (int a = 0; a < 3; ++a) {
      g[6 * i + a] = -gd(a);
      g[6 * i + 3 + a] = gd(a);
      for (int b = 0; b < 3; ++b) {
        h[(i * 36) + a * 6 + b] = k(a, b);
        h[(i * 36) + a * 6 + 3 + b] = -k(a, b);
        h[(i * 36) + (3 + a) * 6 + b] = -k(a, b);
        h[(i * 36) + (3 + a) * 6 + 3 + b] = k(a, b);
      }
    }
  }
  return py::make_tuple(gradient, hessian);
}

py::tuple DERCore::bending_kernel(const py::array& positions,
                                  const py::array& rest_lengths) const {
  const auto p = checked_array(positions, "positions", 2);
  const auto r = checked_array(rest_lengths, "rest_lengths", 1);
  if (p.shape[1] != 3 || r.shape[0] != p.shape[0] - 1 || p.shape[0] < 3) {
    throw py::value_error("bending kernel shape mismatch");
  }
  const auto* x = static_cast<const double*>(p.ptr);
  const auto* l0 = static_cast<const double*>(r.ptr);
  const ssize_t elements = p.shape[0] - 2;
  py::array_t<double> gradient(py::array::ShapeContainer(
      std::vector<py::ssize_t>{elements, 9}));
  py::array_t<double> hessian(py::array::ShapeContainer(
      std::vector<py::ssize_t>{elements, 9, 9}));
  auto* g = static_cast<double*>(gradient.request().ptr);
  auto* h = static_cast<double*>(hessian.request().ptr);
  for (ssize_t i = 0; i < elements; ++i) {
    BendingVector9 local_positions;
    for (int k = 0; k < 9; ++k) local_positions(k) = x[3 * i + k];
    BendingVector9 local_gradient;
    BendingMatrix9 local_hessian;
    if (!analytic_bending_derivatives(local_positions, l0[i], l0[i + 1], ei_,
                                      &local_gradient, &local_hessian)) {
      Eigen::Matrix<Jet<9>, 9, 1> vars;
      for (int k = 0; k < 9; ++k)
        vars(k) = Jet<9>::variable(local_positions(k), k);
      const Jet<9> energy = bending_energy(vars, l0[i], l0[i + 1], ei_);
      local_gradient = energy.gradient;
      local_hessian = energy.hessian;
    }
    for (int a = 0; a < 9; ++a) {
      g[i * 9 + a] = local_gradient(a);
      for (int b = 0; b < 9; ++b)
        h[i * 81 + a * 9 + b] = local_hessian(a, b);
    }
  }
  return py::make_tuple(gradient, hessian);
}

py::dict DERCore::contact_kernel(const py::array& positions,
                                 const py::array& integrals,
                                 const py::array& previous_errors,
                                 double dt, double ki_active) const {
  const auto p = checked_array(positions, "positions", 2);
  const auto old_i = checked_array(integrals, "integrals", 1);
  const auto previous = checked_array(previous_errors, "previous_errors", 1);
  const ssize_t nodes = p.shape[0];
  if (p.shape[1] != 3 || old_i.shape[0] != nodes || previous.shape[0] != nodes) {
    throw py::value_error("contact kernel shape mismatch");
  }
  const auto* x = static_cast<const double*>(p.ptr);
  const auto* oi = static_cast<const double*>(old_i.ptr);
  const auto* pe = static_cast<const double*>(previous.ptr);
  if (mesh_enabled_) {
    py::array_t<double> force(py::array::ShapeContainer(
        std::vector<py::ssize_t>{nodes, 3}));
    py::array_t<double> jacobian(py::array::ShapeContainer(
        std::vector<py::ssize_t>{nodes, 3, 3}));
    py::array_t<double> new_integrals(nodes), errors(nodes), weights(nodes), distances(nodes);
    py::array_t<bool> active(nodes);
    py::array_t<std::int64_t> triangle_ids(nodes);
    auto* f = static_cast<double*>(force.request().ptr);
    auto* j = static_cast<double*>(jacobian.request().ptr);
    auto* ni = static_cast<double*>(new_integrals.request().ptr);
    auto* er = static_cast<double*>(errors.request().ptr);
    auto* we = static_cast<double*>(weights.request().ptr);
    auto* ac = static_cast<bool*>(active.request().ptr);
    auto* di = static_cast<double*>(distances.request().ptr);
    auto* ti = static_cast<std::int64_t*>(triangle_ids.request().ptr);
    for (ssize_t n = 0; n < nodes; ++n) {
      const MeshContactValue value = evaluate_mesh_contact(
          Vec3(x[3 * n], x[3 * n + 1], x[3 * n + 2]), oi[n], pe[n], dt,
          kp_, ki_active, max_pi_force_, leak_rate_, max_integral_state_,
          contact_smooth_length_, mesh_triangles_, mesh_normals_, mesh_triangle_ids_,
          mesh_bvh_);
      for (int k = 0; k < 3; ++k) {
        f[3 * n + k] = value.value.force(k);
        for (int l = 0; l < 3; ++l) j[9 * n + 3 * k + l] = value.value.jacobian(k, l);
      }
      ni[n] = value.value.new_integral;
      er[n] = value.value.error;
      we[n] = value.value.weight;
      ac[n] = value.value.active;
      di[n] = value.value.distance;
      ti[n] = value.triangle_id;
    }
    py::dict result;
    result["force"] = force;
    result["jacobian"] = jacobian;
    result["new_integrals"] = new_integrals;
    result["errors"] = errors;
    result["weights"] = weights;
    result["active"] = active;
    result["distances"] = distances;
    result["triangle_ids"] = triangle_ids;
    result["mesh_triangle_ids"] = triangle_ids;
    return result;
  }
  py::array_t<double> force(py::array::ShapeContainer(
      std::vector<py::ssize_t>{nodes, 3}));
  py::array_t<double> jacobian(py::array::ShapeContainer(
      std::vector<py::ssize_t>{nodes, 2, 2}));
  py::array_t<double> new_integrals(nodes);
  py::array_t<double> errors(nodes);
  py::array_t<double> weights(nodes);
  py::array_t<bool> active(nodes);
  py::array_t<double> distances(nodes);
  auto* f = static_cast<double*>(force.request().ptr);
  auto* j = static_cast<double*>(jacobian.request().ptr);
  auto* ni = static_cast<double*>(new_integrals.request().ptr);
  auto* er = static_cast<double*>(errors.request().ptr);
  auto* we = static_cast<double*>(weights.request().ptr);
  auto* ac = static_cast<bool*>(active.request().ptr);
  auto* di = static_cast<double*>(distances.request().ptr);
  const double beta = 1.0 / contact_smooth_length_;
  const double decay = std::exp(-leak_rate_ * dt);
  for (ssize_t n = 0; n < nodes; ++n) {
    const double y = x[3 * n + 1];
    const double z = x[3 * n + 2];
    const double radial = std::hypot(y, z);
    const double protection = radial - (tube_radius_ - safe_distance_);
    const double error = softplus(protection, beta);
    const double error_prime = sigmoid(beta * protection);
    const double weight_prime = beta * error_prime * (1.0 - error_prime);
    const double weight = error_prime;
    const double accumulated = oi[n] + error * dt;
    const double leaked = oi[n] * decay;
    const double raw_candidate = leaked + weight * (accumulated - leaked);
    const double candidate = std::min(std::max(raw_candidate, 0.0), max_integral_state_);
    const bool candidate_inside = raw_candidate >= 0.0 && raw_candidate <= max_integral_state_;
    const double candidate_prime = candidate_inside
        ? weight_prime * (accumulated - leaked) + dt * error_prime * error_prime
        : 0.0;
    const double unsaturated = kp_ * error + ki_active * candidate;
    const bool stop_growth =
        (ki_active > 0.0 && unsaturated >= max_pi_force_ && candidate > oi[n]) ||
        (ki_active > 0.0 && error < pe[n] && ki_active * oi[n] > kp_ * error && candidate > oi[n]);
    const double integral = stop_growth ? oi[n] : candidate;
    const double integral_prime = stop_growth ? 0.0 : candidate_prime;
    const double raw_magnitude = kp_ * error + ki_active * integral;
    const double magnitude = std::min(std::max(raw_magnitude, 0.0), max_pi_force_);
    const bool magnitude_inside = raw_magnitude >= 0.0 && raw_magnitude <= max_pi_force_;
    const double magnitude_prime = magnitude_inside
        ? kp_ * error_prime + ki_active * integral_prime
        : 0.0;
    const double scale = std::sqrt(radial * radial + direction_epsilon_ * direction_epsilon_);
    const double inverse_radial = radial > 0.0 ? 1.0 / radial : 0.0;
    const double c1 = -magnitude / scale;
    const double c2 = magnitude / std::pow(scale, 3) -
                      magnitude_prime * inverse_radial / scale;
    f[3 * n] = 0.0;
    f[3 * n + 1] = c1 * y;
    f[3 * n + 2] = c1 * z;
    j[4 * n] = c1 + c2 * y * y;
    j[4 * n + 1] = c2 * y * z;
    j[4 * n + 2] = c2 * y * z;
    j[4 * n + 3] = c1 + c2 * z * z;
    ni[n] = integral;
    er[n] = error;
    we[n] = weight;
    ac[n] = protection > 0.0;
    di[n] = tube_radius_ - radial;
  }
  py::dict result;
  result["force"] = force;
  result["jacobian"] = jacobian;
  result["new_integrals"] = new_integrals;
  result["errors"] = errors;
  result["weights"] = weights;
  result["active"] = active;
  result["distances"] = distances;
  return result;
}

py::dict DERCore::implicit_euler_step(
    const py::array& positions, const py::array& velocities,
    const py::array& rest_lengths, const py::array& masses,
    const py::array& integrals, const py::array& initial_positions,
    const py::array& pulse_forces,
    bool tip_constrained, double tip_displacement, double dt,
    int max_newton_iters, double residual_tol, double step_tol,
    double ki_active) const {
  const auto p = checked_array(positions, "positions", 2);
  const auto v = checked_array(velocities, "velocities", 2);
  const auto r = checked_array(rest_lengths, "rest_lengths", 1);
  const auto m = checked_array(masses, "masses", 1);
  const auto ii = checked_array(integrals, "integrals", 1);
  const auto initial = checked_array(initial_positions, "initial_positions", 2);
  const auto pulse = checked_array(pulse_forces, "pulse_forces", 2);
  const ssize_t nodes = p.shape[0];
  if (nodes < 4 || p.shape[1] != 3 || v.shape[0] != nodes || v.shape[1] != 3 ||
      r.shape[0] != nodes - 1 || m.shape[0] != nodes || ii.shape[0] != nodes ||
      initial.shape[0] != nodes || initial.shape[1] != 3 ||
      pulse.shape[0] != nodes || pulse.shape[1] != 3) {
    throw py::value_error("implicit step shape mismatch");
  }
  const auto* old_p = static_cast<const double*>(p.ptr);
  const auto* old_v = static_cast<const double*>(v.ptr);
  const auto* rest = static_cast<const double*>(r.ptr);
  const auto* mass = static_cast<const double*>(m.ptr);
  const auto* old_integral = static_cast<const double*>(ii.ptr);
  const auto* initial_p = static_cast<const double*>(initial.ptr);
  const auto* pulse_ptr = static_cast<const double*>(pulse.ptr);
  const int first_free = 2;
  const int last_free = static_cast<int>(nodes) - (tip_constrained ? 1 : 0);
  const int free_nodes = last_free - first_free;
  if (free_nodes <= 0) throw py::value_error("no free nodes in implicit step");
  const double beta = 1.0 / contact_smooth_length_;
  const auto started = std::chrono::steady_clock::now();
  SolverWorkspace workspace(static_cast<int>(nodes), free_nodes);
  auto& old_q = workspace.old_q;
  auto& q = workspace.q;
  auto& prescribed = workspace.prescribed;
  auto& old_full = workspace.old_full;
  auto& velocity_full = workspace.velocity_full;
  auto& pulse_full = workspace.pulse_full;
  auto& previous_error = workspace.previous_error;
  for (int n = 0; n < nodes; ++n) {
    for (int k = 0; k < 3; ++k) {
      old_full(3 * n + k) = old_p[3 * n + k];
      velocity_full(3 * n + k) = old_v[3 * n + k];
      pulse_full(3 * n + k) = pulse_ptr[3 * n + k];
      prescribed(3 * n + k) = initial_p[3 * n + k];
    }
  }
  if (tip_constrained) prescribed(3 * (nodes - 1)) -= tip_displacement;
  for (int n = first_free; n < last_free; ++n) {
    for (int k = 0; k < 3; ++k) {
      const int j = 3 * (n - first_free) + k;
      old_q(j) = old_full(3 * n + k);
      q(j) = old_full(3 * n + k) + dt * velocity_full(3 * n + k);
    }
  }

  std::fill(workspace.mesh_frozen_faces.begin(),
            workspace.mesh_frozen_faces.end(), -1);
  if (mesh_enabled_) {
    for (int n = 0; n < nodes; ++n) {
      Vec3 predictor = prescribed.segment<3>(3 * n);
      if (n >= first_free && n < last_free)
        predictor = q.segment<3>(3 * (n - first_free));
      workspace.mesh_frozen_faces[n] = nearest_mesh_face(predictor, mesh_bvh_);
    }
  }

  for (int n = 0; n < nodes; ++n) {
    if (mesh_enabled_) {
      const MeshContactValue value = evaluate_mesh_contact(
          old_full.segment<3>(3 * n), old_integral[n], 0.0, dt, kp_, ki_active,
          max_pi_force_, leak_rate_, max_integral_state_, contact_smooth_length_,
          mesh_triangles_, mesh_normals_, mesh_triangle_ids_, mesh_bvh_);
      previous_error(n) = value.value.error;
    } else {
      const double radial = std::hypot(old_full(3 * n + 1), old_full(3 * n + 2));
      previous_error(n) = softplus(radial - (tube_radius_ - safe_distance_), beta);
    }
  }
  // All subsequent work until the output buffers are created is pure Eigen/
  // STL arithmetic.  Let other Python threads run while Newton is active.
  auto gil_release = std::make_unique<py::gil_scoped_release>();

  auto node_block = [&](int node) {
    return (node >= first_free && node < last_free) ? node - first_free : -1;
  };
  auto add_local_block = [&](BlockPentaSystem* system, int row_node, int col_node,
                             const Mat3& block) {
    const int row = node_block(row_node), col = node_block(col_node);
    if (row >= 0 && col >= 0) system->add_block(row, col, block);
  };
  auto assemble = [&](const Eigen::VectorXd& current, bool with_jacobian,
                      Eigen::VectorXd* residual, BlockPentaSystem* system,
                      Eigen::VectorXd* full_out) {
    auto& full = workspace.full;
    auto& gradient = workspace.gradient;
    auto& contact_force = workspace.contact_force;
    auto& contact_jac = workspace.contact_jacobian;
    full = prescribed;
    for (int n = first_free; n < last_free; ++n)
      full.segment<3>(3 * n) = current.segment<3>(3 * (n - first_free));
    workspace.reset_assembly(with_jacobian);
    if (with_jacobian) workspace.jacobian.reset();

    for (int n = 0; n < nodes - 1; ++n) {
      const Vec3 d = full.segment<3>(3 * (n + 1)) - full.segment<3>(3 * n);
      const double length = std::max(d.norm(), 1.0e-12);
      const double kspring = ea_ / rest[n];
      const double coeff = 1.0 - rest[n] / length;
      const Vec3 gd = kspring * coeff * d;
      const Mat3 k = kspring *
          (coeff * Mat3::Identity() + rest[n] / std::pow(length, 3) * d * d.transpose());
      gradient.segment<3>(3 * n) -= gd;
      gradient.segment<3>(3 * (n + 1)) += gd;
      if (with_jacobian) {
        add_local_block(system, n, n, k);
        add_local_block(system, n, n + 1, -k);
        add_local_block(system, n + 1, n, -k);
        add_local_block(system, n + 1, n + 1, k);
      }
    }

    for (int n = 0; n < nodes - 2; ++n) {
      const int ns[3] = {n, n + 1, n + 2};
      BendingVector9 local_positions;
      for (int k = 0; k < 3; ++k) {
        local_positions(k) = full(3 * n + k);
        local_positions(3 + k) = full(3 * (n + 1) + k);
        local_positions(6 + k) = full(3 * (n + 2) + k);
      }
      if (with_jacobian) {
        BendingVector9 local_gradient;
        BendingMatrix9 local_hessian;
        if (!analytic_bending_derivatives(local_positions, rest[n], rest[n + 1],
                                          ei_, &local_gradient, &local_hessian)) {
          Eigen::Matrix<Jet<9>, 9, 1> vars;
          for (int k = 0; k < 9; ++k)
            vars(k) = Jet<9>::variable(local_positions(k), k);
          const auto energy = bending_energy(vars, rest[n], rest[n + 1], ei_);
          local_gradient = energy.gradient;
          local_hessian = energy.hessian;
        }
        for (int a = 0; a < 3; ++a)
          for (int ra = 0; ra < 3; ++ra)
            gradient.segment<3>(3 * ns[ra])(a) += local_gradient(3 * ra + a);
        for (int ra = 0; ra < 3; ++ra) {
          for (int cb = 0; cb < 3; ++cb) {
            Mat3 block = Mat3::Zero();
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b)
                block(a, b) = local_hessian(3 * ra + a, 3 * cb + b);
            add_local_block(system, ns[ra], ns[cb], block);
          }
        }
      } else {
        BendingVector9 local_gradient;
        if (!analytic_bending_gradient(local_positions, rest[n], rest[n + 1], ei_,
                                       &local_gradient)) {
          Eigen::Matrix<FirstJet<9>, 9, 1> vars;
          for (int k = 0; k < 9; ++k)
            vars(k) = FirstJet<9>::variable(local_positions(k), k);
          const auto energy = first_bending_energy(vars, rest[n], rest[n + 1], ei_);
          local_gradient = energy.gradient;
        }
        for (int a = 0; a < 3; ++a)
          for (int ra = 0; ra < 3; ++ra)
            gradient.segment<3>(3 * ns[ra])(a) += local_gradient(3 * ra + a);
      }
    }

    const double decay = std::exp(-leak_rate_ * dt);
    for (int n = 0; n < nodes; ++n) {
      if (mesh_enabled_) {
        const MeshContactValue value = evaluate_mesh_contact(
            full.segment<3>(3 * n), old_integral[n], previous_error(n), dt, kp_,
            ki_active, max_pi_force_, leak_rate_, max_integral_state_,
            contact_smooth_length_, mesh_triangles_, mesh_normals_, mesh_triangle_ids_,
            mesh_bvh_, workspace.mesh_frozen_faces[n]);
        contact_force.segment<3>(3 * n) = value.value.force;
        if (with_jacobian) contact_jac[n] = value.value.jacobian;
        continue;
      }
      const double y = full(3 * n + 1), z = full(3 * n + 2);
      const double radial = std::hypot(y, z);
      const double protection = radial - (tube_radius_ - safe_distance_);
      const double error = softplus(protection, beta);
      const double ep = sigmoid(beta * protection);
      const double wp = beta * ep * (1.0 - ep);
      const double accumulated = old_integral[n] + error * dt;
      const double leaked = old_integral[n] * decay;
      const double raw_candidate = leaked + ep * (accumulated - leaked);
      const double candidate = std::min(std::max(raw_candidate, 0.0), max_integral_state_);
      const bool inside = raw_candidate >= 0.0 && raw_candidate <= max_integral_state_;
      const double candidate_prime = inside
          ? wp * (accumulated - leaked) + dt * ep * ep : 0.0;
      const double unsat = kp_ * error + ki_active * candidate;
      const bool stop = (ki_active > 0.0 && unsat >= max_pi_force_ && candidate > old_integral[n]) ||
                        (ki_active > 0.0 && error < previous_error(n) &&
                         ki_active * old_integral[n] > kp_ * error && candidate > old_integral[n]);
      const double integral = stop ? old_integral[n] : candidate;
      const double integral_prime = stop ? 0.0 : candidate_prime;
      const double raw_mag = kp_ * error + ki_active * integral;
      const double magnitude = std::min(std::max(raw_mag, 0.0), max_pi_force_);
      const bool mag_inside = raw_mag >= 0.0 && raw_mag <= max_pi_force_;
      const double magnitude_prime = mag_inside ? kp_ * ep + ki_active * integral_prime : 0.0;
      const double scale = std::sqrt(radial * radial + direction_epsilon_ * direction_epsilon_);
      const double inv_r = radial > 0.0 ? 1.0 / radial : 0.0;
      const double c1 = -magnitude / scale;
      const double c2 = magnitude / std::pow(scale, 3) - magnitude_prime * inv_r / scale;
      contact_force(3 * n + 1) = c1 * y;
      contact_force(3 * n + 2) = c1 * z;
      if (with_jacobian) {
        Mat3 cj = Mat3::Zero();
        cj(1, 1) = c1 + c2 * y * y;
        cj(1, 2) = c2 * y * z;
        cj(2, 1) = c2 * y * z;
        cj(2, 2) = c1 + c2 * z * z;
        contact_jac[n] = cj;
      }
    }
    residual->setZero();
    for (int n = first_free; n < last_free; ++n) {
      const int block = n - first_free;
      for (int k = 0; k < 3; ++k) {
        const int src = 3 * n + k;
        (*residual)(3 * block + k) =
            mass[n] * (current(3 * block + k) - old_q(3 * block + k) -
                       dt * velocity_full(src)) / (dt * dt) +
            gradient(src) + damping_ * (current(3 * block + k) - old_q(3 * block + k)) / dt -
            contact_force(src) - pulse_full(src);
      }
      if (with_jacobian) {
        Mat3 diagonal = (mass[n] / (dt * dt) + damping_ / dt) * Mat3::Identity();
        diagonal -= contact_jac[n];
        system->add_block(block, block, diagonal);
      }
    }
    if (full_out) *full_out = full;
  };

  auto& residual = workspace.residual;
  auto& full_result = workspace.full_result;
  auto& jacobian = workspace.jacobian;
  double residual_norm = 0.0, step_norm = 0.0;
  bool converged = false;
  int iterations = 0, line_search_failures = 0, dense_fallback_count = 0;
  int full_assembly_count = 0, residual_only_count = 0, line_search_evaluations = 0;
  int banded_solve_count = 0, stagnation_count = 0;
  double assembly_seconds = 0.0, solve_seconds = 0.0, line_search_seconds = 0.0;
  auto timed_assemble = [&](const Eigen::VectorXd& current, bool full_jacobian,
                            Eigen::VectorXd* out_residual, BlockPentaSystem* out_system,
                            Eigen::VectorXd* out_full) {
    const auto begin = std::chrono::steady_clock::now();
    assemble(current, full_jacobian, out_residual, out_system, out_full);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    assembly_seconds += elapsed;
    if (full_jacobian) ++full_assembly_count;
    else ++residual_only_count;
  };

  timed_assemble(q, true, &residual, &jacobian, &full_result);
  residual_norm = residual.norm();
  for (int iteration = 0; iteration < max_newton_iters; ++iteration) {
    iterations = iteration + 1;
    if (residual_norm < residual_tol) {
      converged = true;
      break;
    }
    const auto solve_begin = std::chrono::steady_clock::now();
    const bool force_dense = stagnation_count >= 3;
    workspace.solve_rhs = -residual;
    const SolveResult solved = block_penta_solve(
        jacobian, workspace.solve_rhs, force_dense, &workspace.solve_scratch,
        &workspace.delta);
    solve_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_begin).count();
    ++banded_solve_count;
    if (solved.used_dense_fallback) ++dense_fallback_count;
    const auto& delta = workspace.delta;
    if (!delta.allFinite()) throw std::runtime_error("C++ Newton solve produced NaN/Inf");
    step_norm = delta.norm();
    if (step_norm < step_tol) {
      converged = true;
      break;
    }
    const double base = residual_norm;
    double alpha = 1.0;
    bool accepted = false;
    const auto line_begin = std::chrono::steady_clock::now();
    for (int trial_index = 0; trial_index < 12; ++trial_index) {
      workspace.trial = q + alpha * delta;
      timed_assemble(workspace.trial, false, &workspace.trial_residual,
                     nullptr, nullptr);
      ++line_search_evaluations;
      const double trial_norm = workspace.trial_residual.norm();
      if (std::isfinite(trial_norm) && trial_norm < base) {
        q = workspace.trial;
        residual = workspace.trial_residual;
        residual_norm = trial_norm;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    line_search_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - line_begin).count();
    if (!accepted) {
      ++line_search_failures;
      q += 0.05 * delta;
      timed_assemble(q, false, &residual, nullptr, nullptr);
      residual_norm = residual.norm();
    }
    if (accepted && residual_norm < base * (1.0 - 1.0e-8)) stagnation_count = 0;
    else ++stagnation_count;
    timed_assemble(q, true, &residual, &jacobian, &full_result);
    residual_norm = residual.norm();
  }
  if (!converged && residual_norm < residual_tol) converged = true;
  auto& new_velocity = workspace.new_velocity;
  new_velocity = (full_result - old_full) / dt;
  for (int n = 0; n < 2; ++n) new_velocity.segment<3>(3 * n).setZero();
  if (tip_constrained) new_velocity.segment<3>(3 * (nodes - 1)) =
      (prescribed.segment<3>(3 * (nodes - 1)) - old_full.segment<3>(3 * (nodes - 1))) / dt;

  gil_release.reset();
  py::array_t<double> out_positions(py::array::ShapeContainer(
      std::vector<py::ssize_t>{nodes, 3}));
  py::array_t<double> out_velocities(py::array::ShapeContainer(
      std::vector<py::ssize_t>{nodes, 3}));
  py::array_t<double> previous_error_array(nodes);
  auto* op = static_cast<double*>(out_positions.request().ptr);
  auto* ov = static_cast<double*>(out_velocities.request().ptr);
  auto* pea = static_cast<double*>(previous_error_array.request().ptr);
  for (int n = 0; n < nodes; ++n) pea[n] = previous_error(n);
  for (int i = 0; i < 3 * nodes; ++i) {
    op[i] = full_result(i);
    ov[i] = new_velocity(i);
  }
  py::dict contact = contact_kernel(out_positions, integrals, previous_error_array, dt,
                                    ki_active);
  const double wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  py::dict result;
  result["positions"] = out_positions;
  result["velocities"] = out_velocities;
  result["integrals"] = contact["new_integrals"];
  result["contact_force"] = contact["force"];
  result["contact_active"] = contact["active"];
  result["contact_weight"] = contact["weights"];
  result["contact_error"] = contact["errors"];
  result["contact_distance"] = contact["distances"];
  result["converged"] = converged;
  result["iterations"] = iterations;
  result["residual_norm"] = residual_norm;
  result["step_norm"] = step_norm;
  result["line_search_failures"] = line_search_failures;
  result["dense_fallback_count"] = dense_fallback_count;
  result["dense_build_count"] = dense_fallback_count;
  result["full_assembly_count"] = full_assembly_count;
  result["residual_only_count"] = residual_only_count;
  result["line_search_evaluation_count"] = line_search_evaluations;
  result["banded_solve_count"] = banded_solve_count;
  result["assembly_seconds"] = assembly_seconds;
  result["solve_seconds"] = solve_seconds;
  result["line_search_seconds"] = line_search_seconds;
  result["wall_seconds"] = wall_seconds;
  return result;
}

py::dict DERCore::implicit_euler_step_cached(
    const py::array& positions, const py::array& velocities,
    const py::array& integrals, const py::array& initial_positions,
    const py::array& pulse_forces, bool tip_constrained,
    double tip_displacement, double dt, int max_newton_iters,
    double residual_tol, double step_tol, double ki_active) const {
  // The constructor-owned rest/mass buffers are exposed as temporary views;
  // no per-step NumPy allocation or copy is needed for the production path.
  py::array_t<double> rest(static_cast<py::ssize_t>(rest_lengths_.size()),
                           rest_lengths_.data());
  py::array_t<double> mass(static_cast<py::ssize_t>(masses_.size()),
                           masses_.data());
  return implicit_euler_step(
      positions, velocities, rest, mass, integrals, initial_positions,
      pulse_forces, tip_constrained, tip_displacement, dt, max_newton_iters,
      residual_tol, step_tol, ki_active);
}

}  // namespace der
