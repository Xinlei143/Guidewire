#include "der_bending.h"

#include <cmath>

namespace der {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using EdgeMap = Eigen::Matrix<double, 6, 9>;

constexpr double kMinimumEdgeSquared = 1.0e-24;
constexpr double kMinimumBendingDenominator = 1.0e-12;

struct BendingGeometry {
  double left_length;
  double right_length;
  double tangent_dot;
  double denominator;
  Vec3 left_tangent;
  Vec3 right_tangent;
  Vec3 left_dot_gradient;
  Vec3 right_dot_gradient;
};

const EdgeMap& node_to_edge_map() {
  static const EdgeMap map = [] {
    EdgeMap result = EdgeMap::Zero();
    result.block<3, 3>(0, 0) = -Mat3::Identity();
    result.block<3, 3>(0, 3) = Mat3::Identity();
    result.block<3, 3>(3, 3) = -Mat3::Identity();
    result.block<3, 3>(3, 6) = Mat3::Identity();
    return result;
  }();
  return map;
}

bool bending_geometry(const BendingVector9& positions, BendingGeometry* geometry) {
  const Vec3 left = positions.segment<3>(3) - positions.segment<3>(0);
  const Vec3 right = positions.segment<3>(6) - positions.segment<3>(3);
  const double left_squared = left.squaredNorm();
  const double right_squared = right.squaredNorm();

  // The old Jet kernels make sqrt(clamp(edge^2, 1e-24)) constant at and below
  // this boundary.  The normalized-dot identity is not valid in that branch.
  if (!(left_squared > kMinimumEdgeSquared &&
        right_squared > kMinimumEdgeSquared)) {
    return false;
  }

  geometry->left_length = std::sqrt(left_squared);
  geometry->right_length = std::sqrt(right_squared);
  geometry->left_tangent = left / geometry->left_length;
  geometry->right_tangent = right / geometry->right_length;
  geometry->tangent_dot =
      geometry->left_tangent.dot(geometry->right_tangent);
  geometry->denominator = 1.0 + geometry->tangent_dot;

  // Keep the complete clamped cross-product energy as the exact fallback at
  // and below the antiparallel denominator threshold.
  if (!(geometry->denominator > kMinimumBendingDenominator)) return false;

  const Mat3 left_projector =
      Mat3::Identity() - geometry->left_tangent * geometry->left_tangent.transpose();
  const Mat3 right_projector =
      Mat3::Identity() - geometry->right_tangent * geometry->right_tangent.transpose();
  geometry->left_dot_gradient =
      left_projector * geometry->right_tangent / geometry->left_length;
  geometry->right_dot_gradient =
      right_projector * geometry->left_tangent / geometry->right_length;
  return geometry->left_dot_gradient.allFinite() &&
         geometry->right_dot_gradient.allFinite();
}

Vector6 edge_dot_gradient(const BendingGeometry& geometry) {
  Vector6 gradient;
  gradient.segment<3>(0) = geometry.left_dot_gradient;
  gradient.segment<3>(3) = geometry.right_dot_gradient;
  return gradient;
}

Matrix6 edge_dot_hessian(const BendingGeometry& geometry) {
  const Vec3& u = geometry.left_tangent;
  const Vec3& v = geometry.right_tangent;
  const Vec3& ga = geometry.left_dot_gradient;
  const Vec3& gb = geometry.right_dot_gradient;
  const double r = geometry.left_length;
  const double s = geometry.right_length;
  const double c = geometry.tangent_dot;
  const Mat3 pu = Mat3::Identity() - u * u.transpose();
  const Mat3 pv = Mat3::Identity() - v * v.transpose();

  const Mat3 haa =
      -(c / (r * r)) * pu - (u * ga.transpose() + ga * u.transpose()) / r;
  const Mat3 hbb =
      -(c / (s * s)) * pv - (v * gb.transpose() + gb * v.transpose()) / s;
  const Mat3 hab = (pv / s - u * gb.transpose()) / r;

  Matrix6 hessian;
  hessian.block<3, 3>(0, 0) = haa;
  hessian.block<3, 3>(0, 3) = hab;
  hessian.block<3, 3>(3, 0) = hab.transpose();
  hessian.block<3, 3>(3, 3) = hbb;
  return hessian;
}

double energy_first_derivative(const BendingGeometry& geometry, double scale) {
  return -2.0 * scale /
         (geometry.denominator * geometry.denominator);
}

}  // namespace

bool analytic_bending_gradient(const BendingVector9& positions,
                               double left_rest, double right_rest, double ei,
                               BendingVector9* gradient) {
  BendingGeometry geometry;
  if (!bending_geometry(positions, &geometry)) return false;
  const double voronoi_length = 0.5 * (left_rest + right_rest);
  const double scale = 2.0 * ei / voronoi_length;
  const Vector6 dot_gradient = edge_dot_gradient(geometry);
  *gradient = energy_first_derivative(geometry, scale) *
              node_to_edge_map().transpose() * dot_gradient;
  return gradient->allFinite();
}

bool analytic_bending_derivatives(const BendingVector9& positions,
                                  double left_rest, double right_rest, double ei,
                                  BendingVector9* gradient,
                                  BendingMatrix9* hessian) {
  BendingGeometry geometry;
  if (!bending_geometry(positions, &geometry)) return false;
  const double voronoi_length = 0.5 * (left_rest + right_rest);
  const double scale = 2.0 * ei / voronoi_length;
  const double first = energy_first_derivative(geometry, scale);
  const double second = 4.0 * scale /
                        (geometry.denominator * geometry.denominator *
                         geometry.denominator);
  const Vector6 dot_gradient = edge_dot_gradient(geometry);
  const BendingVector9 node_dot_gradient =
      node_to_edge_map().transpose() * dot_gradient;
  const BendingMatrix9 node_dot_hessian =
      node_to_edge_map().transpose() * edge_dot_hessian(geometry) *
      node_to_edge_map();
  *gradient = first * node_dot_gradient;
  *hessian = second * node_dot_gradient * node_dot_gradient.transpose() +
             first * node_dot_hessian;
  return gradient->allFinite() && hessian->allFinite();
}

}  // namespace der
