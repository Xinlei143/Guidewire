#pragma once

#include <Eigen/Core>

#include <vector>

namespace der {

// Immutable closest-triangle index.  Face indices always refer to the original
// configure_mesh order so equal-distance selection remains first-face wins.
class MeshBVH {
 public:
  void build(const std::vector<Eigen::Vector3d>& triangles);
  int nearest_face(const Eigen::Vector3d& point) const;

  static Eigen::Vector3d closest_point_triangle(
      const Eigen::Vector3d& point, const Eigen::Vector3d& a,
      const Eigen::Vector3d& b, const Eigen::Vector3d& c);

 private:
  struct Node {
    Eigen::Vector3d minimum = Eigen::Vector3d::Zero();
    Eigen::Vector3d maximum = Eigen::Vector3d::Zero();
    int begin = 0;
    int count = 0;
    int left = -1;
    int right = -1;
    bool leaf() const { return left < 0; }
  };

  int build_node(int begin, int end);
  double bounds_distance2(const Eigen::Vector3d& point,
                          const Node& node) const;
  void consider_face(const Eigen::Vector3d& point, int face,
                     double* best_distance2, int* best_face) const;
  void nearest_node(const Eigen::Vector3d& point, int node_index,
                    double* best_distance2, int* best_face) const;

  std::vector<Eigen::Vector3d> triangles_;
  std::vector<int> face_indices_;
  std::vector<Node> nodes_;
  int root_ = -1;
};

}  // namespace der
