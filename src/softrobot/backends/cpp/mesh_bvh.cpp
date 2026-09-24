#include "mesh_bvh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace der {
namespace {

constexpr int kLeafFaceCount = 8;
// The shipped protection OBJ has 64 faces; at this size tree setup/traversal
// is not a performance claim, so retain the previous direct scan.
constexpr int kBruteForceFaceThreshold = 64;

Eigen::Vector3d face_centroid(const std::vector<Eigen::Vector3d>& triangles,
                              int face) {
  return (triangles[3 * face] + triangles[3 * face + 1] +
          triangles[3 * face + 2]) /
         3.0;
}

}  // namespace

Eigen::Vector3d MeshBVH::closest_point_triangle(
    const Eigen::Vector3d& point, const Eigen::Vector3d& a,
    const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
  const Eigen::Vector3d ab = b - a;
  const Eigen::Vector3d ac = c - a;
  const Eigen::Vector3d ap = point - a;
  const double d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;
  const Eigen::Vector3d bp = point - b;
  const double d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    return a + (d1 / std::max(d1 - d3, 1.0e-30)) * ab;
  const Eigen::Vector3d cp = point - c;
  const double d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    return a + (d2 / std::max(d2 - d6, 1.0e-30)) * ac;
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    return b + ((d4 - d3) / std::max((d4 - d3) + (d5 - d6), 1.0e-30)) *
                   (c - b);
  const double denominator = std::max(va + vb + vc, 1.0e-30);
  return a + (vb / denominator) * ab + (vc / denominator) * ac;
}

void MeshBVH::build(const std::vector<Eigen::Vector3d>& triangles) {
  triangles_ = triangles;
  const int faces = static_cast<int>(triangles_.size() / 3);
  face_indices_.resize(faces);
  std::iota(face_indices_.begin(), face_indices_.end(), 0);
  nodes_.clear();
  root_ = faces > kBruteForceFaceThreshold ? build_node(0, faces) : -1;
}

int MeshBVH::build_node(int begin, int end) {
  Node node;
  node.begin = begin;
  node.count = end - begin;
  node.minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  node.maximum = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  Eigen::Vector3d centroid_min = node.minimum;
  Eigen::Vector3d centroid_max = node.maximum;
  for (int offset = begin; offset < end; ++offset) {
    const int face = face_indices_[offset];
    for (int vertex = 0; vertex < 3; ++vertex) {
      node.minimum = node.minimum.cwiseMin(triangles_[3 * face + vertex]);
      node.maximum = node.maximum.cwiseMax(triangles_[3 * face + vertex]);
    }
    const Eigen::Vector3d centroid = face_centroid(triangles_, face);
    centroid_min = centroid_min.cwiseMin(centroid);
    centroid_max = centroid_max.cwiseMax(centroid);
  }
  const int index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  if (node.count <= kLeafFaceCount) return index;

  const Eigen::Vector3d extent = centroid_max - centroid_min;
  int axis = 0;
  if (extent(1) > extent(axis)) axis = 1;
  if (extent(2) > extent(axis)) axis = 2;
  std::stable_sort(face_indices_.begin() + begin, face_indices_.begin() + end,
                   [&](int left, int right) {
                     const double left_value = face_centroid(triangles_, left)(axis);
                     const double right_value = face_centroid(triangles_, right)(axis);
                     return left_value == right_value ? left < right : left_value < right_value;
                   });
  const int middle = begin + node.count / 2;
  nodes_[index].left = build_node(begin, middle);
  nodes_[index].right = build_node(middle, end);
  return index;
}

double MeshBVH::bounds_distance2(const Eigen::Vector3d& point,
                                 const Node& node) const {
  double distance2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = point(axis) < node.minimum(axis)
        ? node.minimum(axis) - point(axis)
        : point(axis) > node.maximum(axis) ? point(axis) - node.maximum(axis) : 0.0;
    distance2 += delta * delta;
  }
  return distance2;
}

void MeshBVH::consider_face(const Eigen::Vector3d& point, int face,
                            double* best_distance2, int* best_face) const {
  const Eigen::Vector3d closest = closest_point_triangle(
      point, triangles_[3 * face], triangles_[3 * face + 1], triangles_[3 * face + 2]);
  const double distance2 = (point - closest).squaredNorm();
  // The old linear loop retained its first result on equality.  Original face
  // indices restore that contract independent of deterministic BVH ordering.
  if (distance2 < *best_distance2 ||
      (std::isfinite(distance2) && distance2 == *best_distance2 &&
       (*best_face < 0 || face < *best_face))) {
    *best_distance2 = distance2;
    *best_face = face;
  }
}

void MeshBVH::nearest_node(const Eigen::Vector3d& point, int node_index,
                           double* best_distance2, int* best_face) const {
  const Node& node = nodes_[node_index];
  // Strict comparison retains equal-distance candidates for first-face ties.
  if (bounds_distance2(point, node) > *best_distance2) return;
  if (node.leaf()) {
    for (int offset = node.begin; offset < node.begin + node.count; ++offset)
      consider_face(point, face_indices_[offset], best_distance2, best_face);
    return;
  }
  const double left_distance = bounds_distance2(point, nodes_[node.left]);
  const double right_distance = bounds_distance2(point, nodes_[node.right]);
  if (left_distance <= right_distance) {
    nearest_node(point, node.left, best_distance2, best_face);
    nearest_node(point, node.right, best_distance2, best_face);
  } else {
    nearest_node(point, node.right, best_distance2, best_face);
    nearest_node(point, node.left, best_distance2, best_face);
  }
}

int MeshBVH::nearest_face(const Eigen::Vector3d& point) const {
  if (triangles_.empty()) return -1;
  double best_distance2 = std::numeric_limits<double>::infinity();
  int best_face = -1;
  if (root_ < 0) {
    for (int face = 0; face < static_cast<int>(triangles_.size() / 3); ++face)
      consider_face(point, face, &best_distance2, &best_face);
    return best_face;
  }
  nearest_node(point, root_, &best_distance2, &best_face);
  return best_face;
}

}  // namespace der
