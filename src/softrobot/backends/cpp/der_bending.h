#pragma once

#include <Eigen/Core>

namespace der {

using BendingVector9 = Eigen::Matrix<double, 9, 1>;
using BendingMatrix9 = Eigen::Matrix<double, 9, 9>;

// Return false when the preserved clamped DER energy must be differentiated by
// the guarded Jet fallback (zero-length edges or a clamped antiparallel
// denominator).  On true, the input is in the normal unit-tangent branch.
bool analytic_bending_gradient(const BendingVector9& positions,
                               double left_rest, double right_rest, double ei,
                               BendingVector9* gradient);

bool analytic_bending_derivatives(const BendingVector9& positions,
                                  double left_rest, double right_rest, double ei,
                                  BendingVector9* gradient,
                                  BendingMatrix9* hessian);

}  // namespace der
