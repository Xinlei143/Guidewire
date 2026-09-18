from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_planar2d_common_linear_algebra_and_elastic_assembly(tmp_path):
    source = tmp_path / "planar2d_common_smoke.cpp"
    executable = tmp_path / "planar2d_common_smoke"
    source.write_text(
        r"""
#include "planar2d_core_common.h"

#include <Eigen/Core>

#include <cmath>
#include <vector>

int main() {
  using der::planar2d::BlockPenta2D;
  using der::planar2d::Mat2;
  using der::planar2d::Vec2;

  BlockPenta2D matrix(3);
  matrix.diagonal[0] = 4.0 * Mat2::Identity();
  matrix.diagonal[1] = 5.0 * Mat2::Identity();
  matrix.diagonal[2] = 6.0 * Mat2::Identity();
  matrix.upper1[0] = -Mat2::Identity();
  matrix.lower1[0] = -Mat2::Identity();
  matrix.upper1[1] = -0.5 * Mat2::Identity();
  matrix.lower1[1] = -0.5 * Mat2::Identity();
  matrix.upper2[0] = 0.25 * Mat2::Identity();
  matrix.lower2[0] = 0.25 * Mat2::Identity();

  Eigen::VectorXd value(6);
  value << 0.5, -1.0, 2.0, 0.25, -0.75, 1.5;
  if (!matrix.matvec(value).isApprox(matrix.dense() * value, 1.0e-14)) return 1;
  Eigen::VectorXd solution;
  double relative_residual = 0.0;
  if (!matrix.solve(value, &solution, &relative_residual)) return 2;
  if (!(relative_residual <= 1.0e-12)) return 3;

  const std::vector<Vec2> positions = {
      Vec2(0.0, 0.0), Vec2(1.0, 0.1), Vec2(2.0, 0.0), Vec2(3.0, -0.1)};
  const std::vector<double> rest_lengths = {1.0, 1.0, 1.0};
  const std::vector<double> reference_curvature = {0.0, 0.0};
  std::vector<Vec2> gradient;
  BlockPenta2D hessian(3);
  der::planar2d::assemble_elastic_gradient_hessian(
      positions, rest_lengths, reference_curvature, 1, 4, 10.0, 2.0, true,
      &gradient, &hessian);
  if (gradient.size() != positions.size()) return 4;
  for (const Vec2& entry : gradient) {
    if (!entry.allFinite()) return 5;
  }
  const Eigen::MatrixXd dense_hessian = hessian.dense();
  if (!dense_hessian.allFinite()) return 6;
  if (!dense_hessian.isApprox(dense_hessian.transpose(), 1.0e-12)) return 7;
  return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O0",
            "-I",
            str(ROOT / "cpp"),
            "-I",
            "/usr/include/eigen3",
            str(source),
            str(ROOT / "cpp" / "planar2d_core_common.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
