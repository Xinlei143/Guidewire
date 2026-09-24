#pragma once

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <vector>

namespace der {
namespace planar2d {

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;

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
Jet<D> jet_clamp_min(const Jet<D>& value, double minimum) {
  return value.value < minimum ? Jet<D>(minimum) : value;
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

struct BlockPenta2D {
  explicit BlockPenta2D(int block_count);

  void reset();
  void add_block(int row, int column, const Mat2& value);
  Eigen::VectorXd matvec(const Eigen::VectorXd& value) const;
  Eigen::MatrixXd dense() const;
  bool solve(const Eigen::VectorXd& right_hand_side, Eigen::VectorXd* result,
             double* relative_residual) const;

  int count;
  std::vector<Mat2> diagonal;
  std::vector<Mat2> upper1;
  std::vector<Mat2> lower1;
  std::vector<Mat2> upper2;
  std::vector<Mat2> lower2;
};

void assemble_elastic_gradient_hessian(
    const std::vector<Vec2>& positions,
    const std::vector<double>& rest_lengths,
    const std::vector<double>& reference_curvature, int first_free,
    int last_free, double ea, double ei, bool with_hessian,
    std::vector<Vec2>* gradient, BlockPenta2D* system);

}  // namespace planar2d
}  // namespace der
