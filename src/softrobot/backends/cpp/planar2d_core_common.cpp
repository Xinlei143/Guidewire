#include "planar2d_core_common.h"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace der {
namespace planar2d {

BlockPenta2D::BlockPenta2D(int block_count)
    : count(block_count),
      diagonal(block_count, Mat2::Zero()),
      upper1(std::max(block_count - 1, 0), Mat2::Zero()),
      lower1(std::max(block_count - 1, 0), Mat2::Zero()),
      upper2(std::max(block_count - 2, 0), Mat2::Zero()),
      lower2(std::max(block_count - 2, 0), Mat2::Zero()) {}

void BlockPenta2D::reset() {
  std::fill(diagonal.begin(), diagonal.end(), Mat2::Zero());
  std::fill(upper1.begin(), upper1.end(), Mat2::Zero());
  std::fill(lower1.begin(), lower1.end(), Mat2::Zero());
  std::fill(upper2.begin(), upper2.end(), Mat2::Zero());
  std::fill(lower2.begin(), lower2.end(), Mat2::Zero());
}

void BlockPenta2D::add_block(int row, int column, const Mat2& value) {
  const int offset = column - row;
  if (offset == 0)
    diagonal[row] += value;
  else if (offset == 1)
    upper1[row] += value;
  else if (offset == -1)
    lower1[column] += value;
  else if (offset == 2)
    upper2[row] += value;
  else if (offset == -2)
    lower2[column] += value;
  else if (value.cwiseAbs().maxCoeff() > 1.0e-14)
    throw std::logic_error(
        "Arc2D Jacobian exceeded block-pentadiagonal bandwidth");
}

Eigen::VectorXd BlockPenta2D::matvec(const Eigen::VectorXd& value) const {
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

Eigen::MatrixXd BlockPenta2D::dense() const {
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

bool BlockPenta2D::solve(const Eigen::VectorXd& right_hand_side,
                         Eigen::VectorXd* result,
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
  *relative_residual =
      difference.norm() / std::max(right_hand_side.norm(), 1.0e-30);
  return result->allFinite() && std::isfinite(*relative_residual) &&
         *relative_residual <= 1.0e-8;
}

void assemble_elastic_gradient_hessian(
    const std::vector<Vec2>& positions,
    const std::vector<double>& rest_lengths,
    const std::vector<double>& reference_curvature, int first_free,
    int last_free, double ea, double ei, bool with_hessian,
    std::vector<Vec2>* gradient, BlockPenta2D* system) {
  const int nodes = static_cast<int>(positions.size());
  gradient->assign(nodes, Vec2::Zero());
  if (with_hessian) system->reset();

  const auto free_block = [first_free, last_free](int node) {
    return node >= first_free && node < last_free ? node - first_free : -1;
  };
  const auto scatter_local = [&](const Jet<6>& energy,
                                 const std::array<int, 3>& local_nodes,
                                 int local_node_count) {
    for (int row = 0; row < local_node_count; ++row) {
      const int row_block = free_block(local_nodes[row]);
      if (row_block >= 0)
        (*gradient)[local_nodes[row]] +=
            energy.gradient.segment<2>(2 * row);
      if (!with_hessian || row_block < 0) continue;
      for (int column = 0; column < local_node_count; ++column) {
        const int column_block = free_block(local_nodes[column]);
        if (column_block >= 0) {
          system->add_block(
              row_block, column_block,
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
      local[coordinate] =
          Jet<6>::variable(positions[edge](coordinate), coordinate);
      local[2 + coordinate] =
          Jet<6>::variable(positions[edge + 1](coordinate), 2 + coordinate);
      local[4 + coordinate] = Jet<6>(0.0);
    }
    const Jet<6> energy =
        local_stretch_energy(local, rest_lengths[edge], ea);
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
        local, rest_lengths[vertex], rest_lengths[vertex + 1],
        reference_curvature[vertex], ei);
    scatter_local(energy, {vertex, vertex + 1, vertex + 2}, 3);
  }
}

}  // namespace planar2d
}  // namespace der
