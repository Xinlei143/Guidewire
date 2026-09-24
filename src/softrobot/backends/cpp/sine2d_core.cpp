#include "sine2d_core.h"

#include "planar2d_core_common.h"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace der {
namespace {

using planar2d::Vec2;

py::buffer_info checked_double_array(const py::array& array,
                                     const std::string& name, int ndim) {
  auto info = array.request();
  if (info.ndim != ndim ||
      info.format != py::format_descriptor<double>::format() ||
      info.itemsize != static_cast<ssize_t>(sizeof(double))) {
    throw py::type_error(name + " must be a contiguous float64 array");
  }
  if (!(array.flags() & py::array::c_style))
    throw py::value_error(name + " must be C-contiguous");
  return info;
}

py::buffer_info checked_bool_array(const py::array& array,
                                   const std::string& name, int ndim) {
  auto info = array.request();
  if (info.ndim != ndim || info.format != py::format_descriptor<bool>::format() ||
      info.itemsize != static_cast<ssize_t>(sizeof(bool))) {
    throw py::type_error(name + " must be a contiguous bool array");
  }
  if (!(array.flags() & py::array::c_style))
    throw py::value_error(name + " must be C-contiguous");
  return info;
}

void require_finite(double value, const char* name) {
  if (!std::isfinite(value))
    throw py::value_error(std::string(name) + " must be finite");
}

void require_positive(double value, const char* name) {
  require_finite(value, name);
  if (!(value > 0.0))
    throw py::value_error(std::string(name) + " must be positive");
}

void require_nonnegative(double value, const char* name) {
  require_finite(value, name);
  if (value < 0.0)
    throw py::value_error(std::string(name) + " must be non-negative");
}

void require_finite_values(const double* values, ssize_t count,
                           const char* name) {
  for (ssize_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index]))
      throw py::value_error(std::string(name) +
                            " must contain only finite values");
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

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

double sign(double value) {
  return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

double softplus(double value, double beta) {
  const double scaled = beta * value;
  // torch.nn.functional.softplus uses threshold=20 by default.
  if (scaled > 20.0) return value;
  return std::log1p(std::exp(scaled)) / beta;
}

double sigmoid(double value) {
  if (value >= 0.0) {
    const double exp_negative = std::exp(-value);
    return 1.0 / (1.0 + exp_negative);
  }
  const double exp_positive = std::exp(value);
  return exp_positive / (1.0 + exp_positive);
}

struct Objective {
  double value;
  double gradient;
  double hessian;
};

struct Projection {
  double q;
  Vec2 center;
  Vec2 tangent;
  Vec2 normal;
  double curvature;
  double arclength;
  double signed_distance;
  bool used_global_fallback;
};

Vec2 centerline(double q, double amplitude, double wave_number) {
  return Vec2(q, amplitude * std::cos(wave_number * q));
}

Vec2 derivative(double q, double amplitude, double wave_number) {
  return Vec2(1.0,
              -amplitude * wave_number * std::sin(wave_number * q));
}

Vec2 second_derivative(double q, double amplitude, double wave_number) {
  return Vec2(0.0, -amplitude * wave_number * wave_number *
                       std::cos(wave_number * q));
}

Objective objective(double q, const Vec2& point, double amplitude,
                    double wave_number) {
  const Vec2 delta = centerline(q, amplitude, wave_number) - point;
  const Vec2 first = derivative(q, amplitude, wave_number);
  const Vec2 second = second_derivative(q, amplitude, wave_number);
  return Objective{0.5 * delta.dot(delta), delta.dot(first),
                   first.dot(first) + delta.dot(second)};
}

double warm_refine(double q, const Vec2& point, double amplitude,
                   double length_x, double wave_number) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  q = clamp(q, 0.0, length_x);
  for (int iteration = 0; iteration < 32; ++iteration) {
    const Objective current = objective(q, point, amplitude, wave_number);
    if (std::abs(current.gradient) <= 32.0 * epsilon) break;
    double step = current.hessian > 1.0e-14
                      ? -current.gradient / current.hessian
                      : -sign(current.gradient) * length_x / 16.0;
    step = clamp(step, -length_x / 4.0, length_x / 4.0);
    double trial = clamp(q + step, 0.0, length_x);
    for (int backtrack = 0; backtrack < 32; ++backtrack) {
      if (trial == q ||
          objective(trial, point, amplitude, wave_number).value <=
              current.value)
        break;
      const double next_trial = 0.5 * (trial + q);
      if (next_trial == trial || next_trial == q) {
        trial = q;
        break;
      }
      trial = next_trial;
    }
    if (trial == q) break;
    if (std::abs(trial - q) <=
        16.0 * epsilon * std::max(1.0, length_x)) {
      q = trial;
      break;
    }
    q = trial;
  }
  return q;
}

double bracketed_stationary(double left, double right, const Vec2& point,
                            double amplitude, double length_x,
                            double wave_number) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  double g_left = objective(left, point, amplitude, wave_number).gradient;
  double q = 0.5 * (left + right);
  for (int iteration = 0; iteration < 48; ++iteration) {
    const Objective current = objective(q, point, amplitude, wave_number);
    if (std::abs(current.gradient) <= 32.0 * epsilon) return q;
    const double newton = std::abs(current.hessian) > 1.0e-14
                              ? q - current.gradient / current.hessian
                              : std::numeric_limits<double>::quiet_NaN();
    const double candidate = left < newton && newton < right
                                 ? newton
                                 : 0.5 * (left + right);
    const double g_candidate =
        objective(candidate, point, amplitude, wave_number).gradient;
    if ((g_left <= 0.0 && 0.0 <= g_candidate) ||
        (g_left >= 0.0 && 0.0 >= g_candidate)) {
      right = candidate;
    } else {
      left = candidate;
      g_left = g_candidate;
    }
    q = candidate;
    if (right - left <= 32.0 * epsilon * std::max(1.0, length_x))
      break;
  }
  return q;
}

const std::pair<std::vector<double>, std::vector<double>>& gauss_legendre_256() {
  static const auto rule = [] {
    constexpr int count = 256;
    std::vector<double> nodes(count), weights(count);
    const int half = (count + 1) / 2;
    const double pi = std::acos(-1.0);
    for (int index = 0; index < half; ++index) {
      double root =
          std::cos(pi * (static_cast<double>(index) + 0.75) /
                   (static_cast<double>(count) + 0.5));
      double previous;
      double derivative_value = 0.0;
      int iterations = 0;
      do {
        previous = root;
        double p0 = 1.0;
        double p1 = root;
        for (int order = 2; order <= count; ++order) {
          const double p2 =
              ((2.0 * order - 1.0) * root * p1 - (order - 1.0) * p0) /
              order;
          p0 = p1;
          p1 = p2;
        }
        derivative_value =
            count * (root * p1 - p0) / (root * root - 1.0);
        root = previous - p1 / derivative_value;
        ++iterations;
      } while (std::abs(root - previous) > 1.0e-16 && iterations < 64);
      const double weight =
          2.0 / ((1.0 - root * root) * derivative_value * derivative_value);
      nodes[index] = -root;
      nodes[count - 1 - index] = root;
      weights[index] = weight;
      weights[count - 1 - index] = weight;
    }
    return std::make_pair(nodes, weights);
  }();
  return rule;
}

double arclength(double q, double amplitude, double wave_number) {
  if (q == 0.0) return 0.0;
  const auto& rule = gauss_legendre_256();
  double integral = 0.0;
  for (size_t index = 0; index < rule.first.size(); ++index) {
    const double sample = 0.5 * q * (rule.first[index] + 1.0);
    integral += rule.second[index] *
                derivative(sample, amplitude, wave_number).norm();
  }
  return 0.5 * q * integral;
}

Projection project_one(const Vec2& point, double initial_q, double amplitude,
                       double length_x, double wave_number) {
  enum class Source { Endpoint, Warm, Global };
  struct Candidate {
    double q;
    Source source;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(1030);
  candidates.push_back({0.0, Source::Endpoint});
  candidates.push_back({length_x, Source::Endpoint});
  candidates.push_back(
      {warm_refine(initial_q, point, amplitude, length_x, wave_number),
       Source::Warm});

  constexpr int intervals = 512;
  std::vector<double> values(intervals + 1), gradients(intervals + 1);
  for (int index = 0; index <= intervals; ++index) {
    const double q = length_x * index / intervals;
    const Objective current = objective(q, point, amplitude, wave_number);
    values[index] = current.value;
    gradients[index] = current.gradient;
  }
  for (int index = 1; index < intervals; ++index) {
    if (values[index] <= values[index - 1] &&
        values[index] <= values[index + 1]) {
      const double seed = length_x * index / intervals;
      candidates.push_back(
          {warm_refine(seed, point, amplitude, length_x, wave_number),
           Source::Global});
    }
  }
  for (int index = 0; index < intervals; ++index) {
    if (gradients[index] * gradients[index + 1] < 0.0) {
      candidates.push_back(
          {bracketed_stationary(length_x * index / intervals,
                                length_x * (index + 1) / intervals, point,
                                amplitude, length_x, wave_number),
           Source::Global});
    }
  }

  Candidate best = candidates.front();
  double best_value = objective(best.q, point, amplitude, wave_number).value;
  for (size_t index = 1; index < candidates.size(); ++index) {
    const double value =
        objective(candidates[index].q, point, amplitude, wave_number).value;
    if (value < best_value ||
        (value == best_value && candidates[index].q < best.q)) {
      best = candidates[index];
      best_value = value;
    }
  }

  const Vec2 center = centerline(best.q, amplitude, wave_number);
  const Vec2 first = derivative(best.q, amplitude, wave_number);
  const double speed = first.norm();
  const Vec2 tangent = first / speed;
  const Vec2 normal(-first.y() / speed, first.x() / speed);
  const double curvature =
      -amplitude * wave_number * wave_number *
      std::cos(wave_number * best.q) / (speed * speed * speed);
  return Projection{best.q,
                    center,
                    tangent,
                    normal,
                    curvature,
                    arclength(best.q, amplitude, wave_number),
                    (point - center).dot(normal),
                    best.source == Source::Global};
}

std::vector<double> projection_starts(const py::object& initial_q,
                                      ssize_t count, double length_x) {
  std::vector<double> starts(count, 0.5 * length_x);
  if (initial_q.is_none()) return starts;
  const py::array array = py::cast<py::array>(initial_q);
  const auto info = checked_double_array(array, "initial_q", 1);
  if (info.shape[0] != count)
    throw py::value_error("initial_q must have shape (N,)");
  const auto* data = static_cast<const double*>(info.ptr);
  require_finite_values(data, count, "initial_q");
  std::copy(data, data + count, starts.begin());
  return starts;
}

py::array_t<double> double_vector_array(ssize_t count) {
  return py::array_t<double>(count);
}

py::array_t<double> double_point_array(ssize_t count) {
  return py::array_t<double>(
      py::array::ShapeContainer(std::vector<py::ssize_t>{count, 2}));
}

Projection tracked_project_one(const Vec2& point, double accepted_q,
                               double trust_radius, double amplitude,
                               double length_x, double wave_number) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double stationarity_tol =
      256.0 * epsilon * std::max(1.0, length_x);
  const double warm_q =
      warm_refine(accepted_q, point, amplitude, length_x, wave_number);
  const Objective current = objective(warm_q, point, amplitude, wave_number);
  const Objective start = objective(accepted_q, point, amplitude, wave_number);
  const bool endpoint_ok =
      (warm_q <= stationarity_tol &&
       current.gradient >= -stationarity_tol) ||
      (warm_q >= length_x - stationarity_tol &&
       current.gradient <= stationarity_tol);
  const bool interior_ok =
      stationarity_tol < warm_q && warm_q < length_x - stationarity_tol &&
      std::abs(current.gradient) <= stationarity_tol && current.hessian > 0.0;
  const bool objective_ok =
      current.value <= start.value +
                           128.0 * epsilon * std::max(1.0, start.value);
  bool trusted = std::abs(warm_q - accepted_q) <= trust_radius &&
                 objective_ok && (interior_ok || endpoint_ok);
  if (trusted) {
    constexpr int screen_intervals = 64;
    double screen_seed = 0.0;
    double screen_seed_value = objective(
        screen_seed, point, amplitude, wave_number).value;
    for (int index = 1; index <= screen_intervals; ++index) {
      const double q = length_x * index / screen_intervals;
      const double candidate_value =
          objective(q, point, amplitude, wave_number).value;
      if (candidate_value < screen_seed_value) {
        screen_seed = q;
        screen_seed_value = candidate_value;
      }
    }
    const double screened_q = warm_refine(
        screen_seed, point, amplitude, length_x, wave_number);
    const double screened_value =
        objective(screened_q, point, amplitude, wave_number).value;
    const double comparison_tolerance =
        128.0 * epsilon *
        std::max({1.0, std::abs(current.value), std::abs(screened_value)});
    const bool competing_lobe =
        std::abs(screened_q - warm_q) > stationarity_tol &&
        screened_value <= current.value + comparison_tolerance;
    trusted = !competing_lobe;
  }
  if (!trusted) {
    Projection result = project_one(point, accepted_q, amplitude, length_x,
                                    wave_number);
    result.used_global_fallback = true;
    return result;
  }
  const Vec2 center = centerline(warm_q, amplitude, wave_number);
  const Vec2 first = derivative(warm_q, amplitude, wave_number);
  const double speed = first.norm();
  const Vec2 tangent = first / speed;
  const Vec2 normal(-first.y() / speed, first.x() / speed);
  const double curvature =
      -amplitude * wave_number * wave_number *
      std::cos(wave_number * warm_q) / (speed * speed * speed);
  return Projection{warm_q,
                    center,
                    tangent,
                    normal,
                    curvature,
                    0.0,  // The implicit solver does not consume arclength.
                    (point - center).dot(normal),
                    false};
}

struct FrozenExternal {
  Vec2 vessel = Vec2::Zero();
  Vec2 feeder = Vec2::Zero();
  Vec2 friction = Vec2::Zero();
  planar2d::Mat2 jacobian = planar2d::Mat2::Zero();
  double integral = 0.0;
  double error = 0.0;
  bool active = false;
  double trial_slip = 0.0;
  double candidate_slip = 0.0;
  double traction = 0.0;
  bool stick = false;
  bool slip = false;
  double limit = 0.0;
};

FrozenExternal frozen_external(
    const Vec2& point, const Projection& frame, bool vessel_region,
    double old_integral, double previous_error, double nodal_length,
    double committed_slip, const Vec2& anchor, bool committed_contact,
    double dt, bool friction_enabled, double amplitude,
    double tube_half_width, double safe_distance, double feeder_half_width,
    double feeder_contact_layer, double feeder_kp_line,
    double contact_kp_line, double contact_ki_line,
    double contact_force_cap_line, double friction_mu,
    double friction_kt_line, double max_integral_state, double leak_rate,
    double contact_smooth_length, double direction_epsilon) {
  FrozenExternal result;
  const double beta = 1.0 / contact_smooth_length;
  const double signed_value = (point - frame.center).dot(frame.normal);
  const double signed_sign = sign(signed_value);
  const double z = std::abs(signed_value) -
                   (tube_half_width - safe_distance);
  const double raw_error = softplus(z, beta);
  const double raw_weight = sigmoid(beta * z);
  const double error = vessel_region ? raw_error : 0.0;
  const double weight = vessel_region ? raw_weight : 0.0;
  const double decay = std::exp(-leak_rate * dt);
  const double leaked = old_integral * decay;
  const double candidate_unclamped =
      weight * (old_integral + error * dt) + (1.0 - weight) * leaked;
  const double integral_candidate =
      clamp(candidate_unclamped, 0.0, max_integral_state);
  const double kp = contact_kp_line * nodal_length;
  const double ki = contact_ki_line * nodal_length;
  const double cap = contact_force_cap_line * nodal_length;
  const double unsaturated = kp * error + ki * integral_candidate;
  const bool blocked =
      (ki > 0.0 && unsaturated >= cap &&
       integral_candidate > old_integral) ||
      (ki > 0.0 && error < previous_error &&
       ki * old_integral > kp * error &&
       integral_candidate > old_integral);
  result.integral = blocked ? old_integral : integral_candidate;
  const double raw_magnitude = kp * error + ki * result.integral;
  const double magnitude = clamp(raw_magnitude, 0.0, cap);
  const double radius = std::sqrt(signed_value * signed_value +
                                  direction_epsilon * direction_epsilon);
  const double direction_scale = -signed_value / radius;
  if (vessel_region)
    result.vessel = magnitude * direction_scale * frame.normal;
  result.error = error;
  result.active = vessel_region && z > 0.0;

  // Frozen-frame derivative of the PI normal force.  Branch selection follows
  // torch.where/clamp exactly away from nondifferentiable branch points.
  double derror_dd = 0.0;
  double dweight_dd = 0.0;
  if (vessel_region) {
    derror_dd = raw_weight * signed_sign;
    dweight_dd = beta * raw_weight * (1.0 - raw_weight) * signed_sign;
  }
  double dcandidate_dd =
      dweight_dd * (old_integral + error * dt - leaked) +
      weight * derror_dd * dt;
  if (!(0.0 < candidate_unclamped &&
        candidate_unclamped < max_integral_state))
    dcandidate_dd = 0.0;
  const double dintegral_dd = blocked ? 0.0 : dcandidate_dd;
  double dmagnitude_dd = kp * derror_dd + ki * dintegral_dd;
  if (!(0.0 < raw_magnitude && raw_magnitude < cap)) dmagnitude_dd = 0.0;
  const double ddirection_dd =
      -direction_epsilon * direction_epsilon /
      (radius * radius * radius);
  const double dvscalar_dd =
      dmagnitude_dd * direction_scale + magnitude * ddirection_dd;
  if (vessel_region)
    result.jacobian +=
        dvscalar_dd * (frame.normal * frame.normal.transpose());

  const double feeder_signed = point.y() - amplitude;
  const double feeder_sign = sign(feeder_signed);
  const double feeder_z = std::abs(feeder_signed) -
                          (feeder_half_width - feeder_contact_layer);
  const double feeder_error = softplus(feeder_z, beta);
  const double feeder_weight = sigmoid(beta * feeder_z);
  const double feeder_magnitude =
      feeder_kp_line * nodal_length * feeder_error;
  const double feeder_radius =
      std::sqrt(feeder_signed * feeder_signed +
                direction_epsilon * direction_epsilon);
  const double feeder_direction = -feeder_signed / feeder_radius;
  if (!vessel_region) {
    result.feeder.y() = feeder_magnitude * feeder_direction;
    const double dfeeder_magnitude =
        feeder_kp_line * nodal_length * feeder_weight * feeder_sign;
    const double dfeeder_direction =
        -direction_epsilon * direction_epsilon /
        (feeder_radius * feeder_radius * feeder_radius);
    result.jacobian(1, 1) +=
        dfeeder_magnitude * feeder_direction +
        feeder_magnitude * dfeeder_direction;
  }

  const double base_slip = committed_contact ? committed_slip : 0.0;
  result.trial_slip = base_slip + (point - anchor).dot(frame.tangent);
  const double stiffness = friction_kt_line * nodal_length;
  const double uncapped_traction = -stiffness * result.trial_slip;
  result.limit = friction_mu * result.vessel.norm();
  const bool law_active =
      result.active && friction_mu > 0.0 && friction_enabled;
  result.stick = law_active &&
                 std::abs(uncapped_traction) <= result.limit;
  result.slip = law_active && !result.stick;
  result.traction =
      law_active
          ? (result.stick
                 ? uncapped_traction
                 : -result.limit * sign(result.trial_slip))
          : 0.0;
  result.candidate_slip =
      law_active ? -result.traction / stiffness : 0.0;
  result.friction = result.traction * frame.tangent;
  if (law_active) {
    Vec2 traction_gradient = Vec2::Zero();
    if (result.stick) {
      traction_gradient = -stiffness * frame.tangent;
    } else {
      const double vessel_scalar = magnitude * direction_scale;
      const double normal_sign = sign(vessel_scalar);
      const Vec2 limit_gradient =
          friction_mu * normal_sign * dvscalar_dd * frame.normal;
      traction_gradient = -sign(result.trial_slip) * limit_gradient;
    }
    result.jacobian += frame.tangent * traction_gradient.transpose();
  }
  return result;
}

py::array_t<double> vec2_array(const std::vector<Vec2>& values) {
  py::array_t<double> result = double_point_array(values.size());
  auto* data = static_cast<double*>(result.request().ptr);
  for (size_t index = 0; index < values.size(); ++index) {
    data[2 * index] = values[index].x();
    data[2 * index + 1] = values[index].y();
  }
  return result;
}

}  // namespace

Sine2DCore::Sine2DCore(
    const py::array& rest_lengths, const py::array& masses, double ea,
    double ei, double damping, double amplitude, double length_x,
    double tube_half_width, double safe_distance, double feeder_half_width,
    double feeder_contact_layer, double feeder_kp_line,
    double contact_kp_line, double contact_ki_line,
    double contact_force_cap_line, double friction_mu,
    double friction_kt_line, double max_integral_state, double leak_rate,
    double contact_smooth_length, double direction_epsilon)
    : ea_(ea),
      ei_(ei),
      damping_(damping),
      amplitude_(amplitude),
      length_x_(length_x),
      wave_number_(3.0 * std::acos(-1.0) / length_x),
      tube_half_width_(tube_half_width),
      safe_distance_(safe_distance),
      feeder_half_width_(feeder_half_width),
      feeder_contact_layer_(feeder_contact_layer),
      feeder_kp_line_(feeder_kp_line),
      contact_kp_line_(contact_kp_line),
      contact_ki_line_(contact_ki_line),
      contact_force_cap_line_(contact_force_cap_line),
      friction_mu_(friction_mu),
      friction_kt_line_(friction_kt_line),
      max_integral_state_(max_integral_state),
      leak_rate_(leak_rate),
      contact_smooth_length_(contact_smooth_length),
      direction_epsilon_(direction_epsilon) {
  const auto rest = checked_double_array(rest_lengths, "rest_lengths", 1);
  const auto mass = checked_double_array(masses, "masses", 1);
  const auto* rest_data = static_cast<const double*>(rest.ptr);
  const auto* mass_data = static_cast<const double*>(mass.ptr);

  require_positive(ea, "ea");
  require_positive(ei, "ei");
  require_nonnegative(damping, "damping");
  require_positive(amplitude, "amplitude");
  require_positive(length_x, "length_x");
  require_positive(tube_half_width, "tube_half_width");
  require_positive(safe_distance, "safe_distance");
  require_positive(feeder_half_width, "feeder_half_width");
  require_positive(feeder_contact_layer, "feeder_contact_layer");
  require_positive(feeder_kp_line, "feeder_kp_line");
  require_positive(contact_kp_line, "contact_kp_line");
  require_nonnegative(contact_ki_line, "contact_ki_line");
  require_positive(contact_force_cap_line, "contact_force_cap_line");
  require_nonnegative(friction_mu, "friction_mu");
  require_positive(friction_kt_line, "friction_kt_line");
  require_positive(max_integral_state, "max_integral_state");
  require_nonnegative(leak_rate, "leak_rate");
  require_positive(contact_smooth_length, "contact_smooth_length");
  require_positive(direction_epsilon, "direction_epsilon");
  if (!(safe_distance < tube_half_width))
    throw py::value_error("safe_distance must be smaller than tube_half_width");
  if (!(feeder_contact_layer < feeder_half_width))
    throw py::value_error(
        "feeder_contact_layer must be smaller than feeder_half_width");
  const double minimum_radius =
      1.0 / (amplitude * wave_number_ * wave_number_);
  if (!(tube_half_width < minimum_radius))
    throw py::value_error(
        "tube_half_width must be smaller than the minimum curvature radius");

  require_positive_values(rest_data, rest.shape[0], "rest_lengths");
  require_positive_values(mass_data, mass.shape[0], "masses");
  rest_lengths_.assign(rest_data, rest_data + rest.shape[0]);
  masses_.assign(mass_data, mass_data + mass.shape[0]);
  if (rest_lengths_.size() < 2 ||
      masses_.size() != rest_lengths_.size() + 1) {
    throw py::value_error(
        "masses must contain exactly num_nodes entries and N must be at least 3");
  }
}

py::dict Sine2DCore::geometry_kernel(const py::array& points,
                                     const py::object& initial_q) const {
  const auto point_info = checked_double_array(points, "points", 2);
  const ssize_t count = point_info.shape[0];
  if (point_info.shape[1] != 2)
    throw py::value_error("points must have shape (N, 2)");
  const auto* point_data = static_cast<const double*>(point_info.ptr);
  require_finite_values(point_data, 2 * count, "points");
  const std::vector<double> starts =
      projection_starts(initial_q, count, length_x_);

  py::array_t<double> center = double_point_array(count);
  py::array_t<double> tangent = double_point_array(count);
  py::array_t<double> normal = double_point_array(count);
  py::array_t<double> curvature = double_vector_array(count);
  py::array_t<double> q = double_vector_array(count);
  py::array_t<double> s = double_vector_array(count);
  py::array_t<double> signed_distance = double_vector_array(count);
  py::array_t<bool> region(count);
  py::array_t<bool> used_global_fallback(count);
  auto* center_data = static_cast<double*>(center.request().ptr);
  auto* tangent_data = static_cast<double*>(tangent.request().ptr);
  auto* normal_data = static_cast<double*>(normal.request().ptr);
  auto* curvature_data = static_cast<double*>(curvature.request().ptr);
  auto* q_data = static_cast<double*>(q.request().ptr);
  auto* s_data = static_cast<double*>(s.request().ptr);
  auto* signed_data = static_cast<double*>(signed_distance.request().ptr);
  auto* region_data = static_cast<bool*>(region.request().ptr);
  auto* fallback_data = static_cast<bool*>(used_global_fallback.request().ptr);

  for (ssize_t index = 0; index < count; ++index) {
    const Vec2 point(point_data[2 * index], point_data[2 * index + 1]);
    const Projection projection = project_one(
        point, starts[index], amplitude_, length_x_, wave_number_);
    center_data[2 * index] = projection.center.x();
    center_data[2 * index + 1] = projection.center.y();
    tangent_data[2 * index] = projection.tangent.x();
    tangent_data[2 * index + 1] = projection.tangent.y();
    normal_data[2 * index] = projection.normal.x();
    normal_data[2 * index + 1] = projection.normal.y();
    curvature_data[index] = projection.curvature;
    q_data[index] = projection.q;
    s_data[index] = projection.arclength;
    signed_data[index] = projection.signed_distance;
    region_data[index] = point.x() >= 0.0;
    fallback_data[index] = projection.used_global_fallback;
  }
  py::dict result;
  result["centerline"] = center;
  result["center"] = center;
  result["tangent"] = tangent;
  result["normal"] = normal;
  result["curvature"] = curvature;
  result["q"] = q;
  result["s"] = s;
  result["arclength"] = s;
  result["signed_distance"] = signed_distance;
  result["region"] = region;
  result["used_global_fallback"] = used_global_fallback;
  return result;
}

py::dict Sine2DCore::tracked_geometry_kernel(
    const py::array& points, const py::array& accepted_q,
    double trust_radius) const {
  const auto point_info = checked_double_array(points, "points", 2);
  const auto accepted_info =
      checked_double_array(accepted_q, "accepted_q", 1);
  const ssize_t count = point_info.shape[0];
  if (point_info.shape[1] != 2)
    throw py::value_error("points must have shape (N, 2)");
  if (accepted_info.shape[0] != count)
    throw py::value_error("accepted_q must have shape (N,)");
  if (!std::isfinite(trust_radius) || trust_radius <= 0.0)
    throw py::value_error("trust_radius must be finite and positive");
  const auto* point_data = static_cast<const double*>(point_info.ptr);
  const auto* accepted_data = static_cast<const double*>(accepted_info.ptr);
  require_finite_values(point_data, 2 * count, "points");
  require_finite_values(accepted_data, count, "accepted_q");
  for (ssize_t index = 0; index < count; ++index) {
    if (accepted_data[index] < 0.0 || accepted_data[index] > length_x_)
      throw py::value_error("accepted_q must lie in [0, x_span]");
  }

  py::array_t<double> center = double_point_array(count);
  py::array_t<double> tangent = double_point_array(count);
  py::array_t<double> normal = double_point_array(count);
  py::array_t<double> curvature = double_vector_array(count);
  py::array_t<double> q = double_vector_array(count);
  py::array_t<double> s = double_vector_array(count);
  py::array_t<double> signed_distance = double_vector_array(count);
  py::array_t<bool> region(count);
  py::array_t<bool> used_global_fallback(count);
  auto* center_data = static_cast<double*>(center.request().ptr);
  auto* tangent_data = static_cast<double*>(tangent.request().ptr);
  auto* normal_data = static_cast<double*>(normal.request().ptr);
  auto* curvature_data = static_cast<double*>(curvature.request().ptr);
  auto* q_data = static_cast<double*>(q.request().ptr);
  auto* s_data = static_cast<double*>(s.request().ptr);
  auto* signed_data = static_cast<double*>(signed_distance.request().ptr);
  auto* region_data = static_cast<bool*>(region.request().ptr);
  auto* fallback_data = static_cast<bool*>(used_global_fallback.request().ptr);

  for (ssize_t index = 0; index < count; ++index) {
    const Vec2 point(point_data[2 * index], point_data[2 * index + 1]);
    Projection projection = tracked_project_one(
        point, accepted_data[index], trust_radius, amplitude_, length_x_,
        wave_number_);
    projection.arclength = arclength(
        projection.q, amplitude_, wave_number_);
    center_data[2 * index] = projection.center.x();
    center_data[2 * index + 1] = projection.center.y();
    tangent_data[2 * index] = projection.tangent.x();
    tangent_data[2 * index + 1] = projection.tangent.y();
    normal_data[2 * index] = projection.normal.x();
    normal_data[2 * index + 1] = projection.normal.y();
    curvature_data[index] = projection.curvature;
    q_data[index] = projection.q;
    s_data[index] = projection.arclength;
    signed_data[index] = projection.signed_distance;
    region_data[index] = point.x() >= 0.0;
    fallback_data[index] = projection.used_global_fallback;
  }
  py::dict result;
  result["centerline"] = center;
  result["center"] = center;
  result["tangent"] = tangent;
  result["normal"] = normal;
  result["curvature"] = curvature;
  result["q"] = q;
  result["s"] = s;
  result["arclength"] = s;
  result["signed_distance"] = signed_distance;
  result["region"] = region;
  result["used_global_fallback"] = used_global_fallback;
  return result;
}

py::dict Sine2DCore::contact_kernel(
    const py::array& positions, const py::array& old_integrals,
    const py::array& previous_errors, const py::array& nodal_lengths,
    const py::array& committed_slip,
    const py::array& increment_start_anchors,
    const py::array& committed_contact, double dt, bool friction_enabled,
    double kp_active, double ki_active, const py::object& initial_q) const {
  const auto position_info =
      checked_double_array(positions, "positions", 2);
  const auto integral_info =
      checked_double_array(old_integrals, "old_integrals", 1);
  const auto previous_info =
      checked_double_array(previous_errors, "previous_errors", 1);
  const auto nodal_info =
      checked_double_array(nodal_lengths, "nodal_lengths", 1);
  const auto slip_info =
      checked_double_array(committed_slip, "committed_slip", 1);
  const auto anchor_info = checked_double_array(
      increment_start_anchors, "increment_start_anchors", 2);
  const auto committed_info =
      checked_bool_array(committed_contact, "committed_contact", 1);
  const ssize_t count = position_info.shape[0];
  if (position_info.shape[1] != 2 || integral_info.shape[0] != count ||
      previous_info.shape[0] != count || nodal_info.shape[0] != count ||
      slip_info.shape[0] != count || anchor_info.shape[0] != count ||
      anchor_info.shape[1] != 2 || committed_info.shape[0] != count ||
      count != static_cast<ssize_t>(masses_.size())) {
    throw py::value_error("Sine2D contact kernel shape mismatch");
  }
  require_positive(dt, "dt");
  require_positive(kp_active, "kp_active");
  require_nonnegative(ki_active, "ki_active");
  const auto* position_data = static_cast<const double*>(position_info.ptr);
  const auto* integral_data = static_cast<const double*>(integral_info.ptr);
  const auto* previous_data = static_cast<const double*>(previous_info.ptr);
  const auto* nodal_data = static_cast<const double*>(nodal_info.ptr);
  const auto* slip_data = static_cast<const double*>(slip_info.ptr);
  const auto* anchor_data = static_cast<const double*>(anchor_info.ptr);
  const auto* committed_data = static_cast<const bool*>(committed_info.ptr);
  require_finite_values(position_data, 2 * count, "positions");
  require_finite_values(integral_data, count, "old_integrals");
  require_finite_values(previous_data, count, "previous_errors");
  require_positive_values(nodal_data, count, "nodal_lengths");
  require_finite_values(slip_data, count, "committed_slip");
  require_finite_values(anchor_data, 2 * count, "increment_start_anchors");
  const std::vector<double> starts =
      projection_starts(initial_q, count, length_x_);

  py::array_t<double> vessel_force = double_point_array(count);
  py::array_t<double> feeder_force = double_point_array(count);
  py::array_t<double> total_normal = double_point_array(count);
  py::array_t<double> friction = double_point_array(count);
  py::array_t<double> total_contact = double_point_array(count);
  py::array_t<double> new_integrals = double_vector_array(count);
  py::array_t<bool> active(count);
  py::array_t<double> weight = double_vector_array(count);
  py::array_t<double> error = double_vector_array(count);
  py::array_t<double> clearance = double_vector_array(count);
  py::array_t<double> q = double_vector_array(count);
  py::array_t<double> center = double_point_array(count);
  py::array_t<double> tangent = double_point_array(count);
  py::array_t<double> normal = double_point_array(count);
  py::array_t<bool> region(count);
  py::array_t<double> signed_distance = double_vector_array(count);
  py::array_t<double> trial_slip = double_vector_array(count);
  py::array_t<double> candidate_slip = double_vector_array(count);
  py::array_t<double> traction = double_vector_array(count);
  py::array_t<bool> stick(count);
  py::array_t<bool> slip(count);
  py::array_t<double> limit = double_vector_array(count);
  py::array_t<double> force_jacobian(
      py::array::ShapeContainer(
          std::vector<py::ssize_t>{count, 2, 2}));

  auto* vessel_force_data = static_cast<double*>(vessel_force.request().ptr);
  auto* feeder_force_data = static_cast<double*>(feeder_force.request().ptr);
  auto* total_normal_data = static_cast<double*>(total_normal.request().ptr);
  auto* friction_data = static_cast<double*>(friction.request().ptr);
  auto* total_contact_data = static_cast<double*>(total_contact.request().ptr);
  auto* new_integral_data = static_cast<double*>(new_integrals.request().ptr);
  auto* active_data = static_cast<bool*>(active.request().ptr);
  auto* weight_data = static_cast<double*>(weight.request().ptr);
  auto* error_data = static_cast<double*>(error.request().ptr);
  auto* clearance_data = static_cast<double*>(clearance.request().ptr);
  auto* q_data = static_cast<double*>(q.request().ptr);
  auto* center_data = static_cast<double*>(center.request().ptr);
  auto* tangent_data = static_cast<double*>(tangent.request().ptr);
  auto* normal_data = static_cast<double*>(normal.request().ptr);
  auto* region_data = static_cast<bool*>(region.request().ptr);
  auto* signed_data = static_cast<double*>(signed_distance.request().ptr);
  auto* trial_data = static_cast<double*>(trial_slip.request().ptr);
  auto* candidate_data = static_cast<double*>(candidate_slip.request().ptr);
  auto* traction_data = static_cast<double*>(traction.request().ptr);
  auto* stick_data = static_cast<bool*>(stick.request().ptr);
  auto* slip_state_data = static_cast<bool*>(slip.request().ptr);
  auto* limit_data = static_cast<double*>(limit.request().ptr);
  auto* jacobian_data =
      static_cast<double*>(force_jacobian.request().ptr);

  const double beta = 1.0 / contact_smooth_length_;
  const double decay = std::exp(-leak_rate_ * dt);
  for (ssize_t index = 0; index < count; ++index) {
    const Vec2 point(position_data[2 * index],
                     position_data[2 * index + 1]);
    const Projection projection = project_one(
        point, starts[index], amplitude_, length_x_, wave_number_);
    const bool vessel_region = point.x() >= 0.0;
    const double signed_value = projection.signed_distance;
    const double z = std::abs(signed_value) -
                     (tube_half_width_ - safe_distance_);
    const double raw_error = softplus(z, beta);
    const double raw_weight = sigmoid(beta * z);
    const double current_error = vessel_region ? raw_error : 0.0;
    const double current_weight = vessel_region ? raw_weight : 0.0;
    double integral_candidate =
        current_weight * (integral_data[index] + current_error * dt) +
        (1.0 - current_weight) * integral_data[index] * decay;
    integral_candidate =
        clamp(integral_candidate, 0.0, max_integral_state_);
    const double kp = kp_active * nodal_data[index];
    const double ki = ki_active * nodal_data[index];
    const double cap = contact_force_cap_line_ * nodal_data[index];
    const double unsaturated = kp * current_error + ki * integral_candidate;
    const bool blocked =
        (ki > 0.0 && unsaturated >= cap &&
         integral_candidate > integral_data[index]) ||
        (ki > 0.0 && current_error < previous_data[index] &&
         ki * integral_data[index] > kp * current_error &&
         integral_candidate > integral_data[index]);
    const double next_integral =
        blocked ? integral_data[index] : integral_candidate;
    const double magnitude =
        std::min(std::max(kp * current_error + ki * next_integral, 0.0),
                 cap);
    const double direction_scale =
        -signed_value /
        std::sqrt(signed_value * signed_value +
                  direction_epsilon_ * direction_epsilon_);
    Vec2 vessel = Vec2::Zero();
    if (vessel_region)
      vessel = magnitude * direction_scale * projection.normal;

    const double feeder_signed = point.y() - amplitude_;
    const double feeder_z =
        std::abs(feeder_signed) -
        (feeder_half_width_ - feeder_contact_layer_);
    const double feeder_error = softplus(feeder_z, beta);
    const double feeder_magnitude =
        feeder_kp_line_ * nodal_data[index] * feeder_error;
    const double feeder_y =
        vessel_region
            ? 0.0
            : -feeder_magnitude * feeder_signed /
                  std::sqrt(feeder_signed * feeder_signed +
                            direction_epsilon_ * direction_epsilon_);
    const Vec2 feeder(0.0, feeder_y);
    const Vec2 normal_total = vessel + feeder;
    const bool current_active = vessel_region && z > 0.0;

    const double base_slip =
        committed_data[index] ? slip_data[index] : 0.0;
    const Vec2 anchor(anchor_data[2 * index], anchor_data[2 * index + 1]);
    const double displacement = (point - anchor).dot(projection.tangent);
    const double trial = base_slip + displacement;
    const double stiffness = friction_kt_line_ * nodal_data[index];
    const double uncapped_traction = -stiffness * trial;
    const double coulomb_limit = friction_mu_ * vessel.norm();
    const bool law_active =
        current_active && friction_mu_ > 0.0 && friction_enabled;
    const bool is_stick =
        law_active && std::abs(uncapped_traction) <= coulomb_limit;
    const bool is_slip = law_active && !is_stick;
    const double capped_traction = -coulomb_limit * sign(trial);
    const double tangent_traction =
        law_active ? (is_stick ? uncapped_traction : capped_traction) : 0.0;
    const double next_slip =
        law_active ? -tangent_traction / stiffness : 0.0;
    const Vec2 friction_value = tangent_traction * projection.tangent;
    const Vec2 contact_total = normal_total + friction_value;
    const FrozenExternal local = frozen_external(
        point, projection, vessel_region, integral_data[index],
        previous_data[index], nodal_data[index], slip_data[index], anchor,
        committed_data[index], dt, friction_enabled, amplitude_,
        tube_half_width_, safe_distance_, feeder_half_width_,
        feeder_contact_layer_, feeder_kp_line_, kp_active, ki_active,
        contact_force_cap_line_, friction_mu_, friction_kt_line_,
        max_integral_state_, leak_rate_, contact_smooth_length_,
        direction_epsilon_);

    vessel_force_data[2 * index] = vessel.x();
    vessel_force_data[2 * index + 1] = vessel.y();
    feeder_force_data[2 * index] = 0.0;
    feeder_force_data[2 * index + 1] = feeder.y();
    total_normal_data[2 * index] = normal_total.x();
    total_normal_data[2 * index + 1] = normal_total.y();
    friction_data[2 * index] = friction_value.x();
    friction_data[2 * index + 1] = friction_value.y();
    total_contact_data[2 * index] = contact_total.x();
    total_contact_data[2 * index + 1] = contact_total.y();
    new_integral_data[index] = next_integral;
    active_data[index] = current_active;
    weight_data[index] = current_weight;
    error_data[index] = current_error;
    clearance_data[index] = tube_half_width_ - std::abs(signed_value);
    q_data[index] = projection.q;
    center_data[2 * index] = projection.center.x();
    center_data[2 * index + 1] = projection.center.y();
    tangent_data[2 * index] = projection.tangent.x();
    tangent_data[2 * index + 1] = projection.tangent.y();
    normal_data[2 * index] = projection.normal.x();
    normal_data[2 * index + 1] = projection.normal.y();
    region_data[index] = vessel_region;
    signed_data[index] = signed_value;
    trial_data[index] = trial;
    candidate_data[index] = next_slip;
    traction_data[index] = tangent_traction;
    stick_data[index] = is_stick;
    slip_state_data[index] = is_slip;
    limit_data[index] = coulomb_limit;
    for (int row = 0; row < 2; ++row)
      for (int column = 0; column < 2; ++column)
        jacobian_data[4 * index + 2 * row + column] =
            local.jacobian(row, column);
  }

  py::dict result;
  result["vessel_normal_force"] = vessel_force;
  result["vessel_force"] = vessel_force;
  result["feeder_force"] = feeder_force;
  result["total_normal"] = total_normal;
  result["total_normal_force"] = total_normal;
  result["friction"] = friction;
  result["friction_force"] = friction;
  result["total_contact"] = total_contact;
  result["total_force"] = total_contact;
  result["new_integrals"] = new_integrals;
  result["integrals"] = new_integrals;
  result["active"] = active;
  result["weight"] = weight;
  result["weights"] = weight;
  result["error"] = error;
  result["errors"] = error;
  result["clearance"] = clearance;
  result["q"] = q;
  result["center"] = center;
  result["centerline"] = center;
  result["tangent"] = tangent;
  result["normal"] = normal;
  result["region"] = region;
  result["signed_distance"] = signed_distance;
  result["trial_slip"] = trial_slip;
  result["candidate_slip"] = candidate_slip;
  result["traction"] = traction;
  result["tangent_traction"] = traction;
  result["stick"] = stick;
  result["slip"] = slip;
  result["limit"] = limit;
  result["coulomb_limit"] = limit;
  result["force_jacobian"] = force_jacobian;
  return result;
}

py::dict Sine2DCore::implicit_euler_step(
    const py::array& positions, const py::array& velocities,
    const py::array& old_integrals, const py::array& previous_errors,
    const py::array& nodal_lengths, const py::array& prescribed,
    const py::array& reference_curvature,
    const py::array& committed_slip,
    const py::array& increment_start_anchors,
    const py::array& committed_contact, bool friction_enabled,
    const py::array& accepted_q, double dt, int max_newton_iters,
    double residual_tol, double step_tol, double projection_trust_radius,
    double kp_active, double ki_active) const {
  const auto position_info = checked_double_array(positions, "positions", 2);
  const auto velocity_info = checked_double_array(velocities, "velocities", 2);
  const auto integral_info =
      checked_double_array(old_integrals, "old_integrals", 1);
  const auto previous_info =
      checked_double_array(previous_errors, "previous_errors", 1);
  const auto nodal_info =
      checked_double_array(nodal_lengths, "nodal_lengths", 1);
  const auto prescribed_info =
      checked_double_array(prescribed, "prescribed", 2);
  const auto curvature_info = checked_double_array(
      reference_curvature, "reference_curvature", 1);
  const auto slip_info =
      checked_double_array(committed_slip, "committed_slip", 1);
  const auto anchor_info = checked_double_array(
      increment_start_anchors, "increment_start_anchors", 2);
  const auto contact_info =
      checked_bool_array(committed_contact, "committed_contact", 1);
  const auto accepted_info =
      checked_double_array(accepted_q, "accepted_q", 1);
  const ssize_t nodes = position_info.shape[0];
  if (nodes != static_cast<ssize_t>(masses_.size()) || nodes < 3 ||
      position_info.shape[1] != 2 || velocity_info.shape[0] != nodes ||
      velocity_info.shape[1] != 2 || integral_info.shape[0] != nodes ||
      previous_info.shape[0] != nodes || nodal_info.shape[0] != nodes ||
      prescribed_info.shape[0] != nodes || prescribed_info.shape[1] != 2 ||
      curvature_info.shape[0] != nodes - 2 || slip_info.shape[0] != nodes ||
      anchor_info.shape[0] != nodes || anchor_info.shape[1] != 2 ||
      contact_info.shape[0] != nodes || accepted_info.shape[0] != nodes) {
    throw py::value_error("Sine2D implicit step shape mismatch");
  }
  require_positive(dt, "dt");
  require_positive(residual_tol, "residual_tol");
  require_positive(step_tol, "step_tol");
  require_positive(projection_trust_radius, "projection_trust_radius");
  require_positive(kp_active, "kp_active");
  require_nonnegative(ki_active, "ki_active");
  if (max_newton_iters <= 0)
    throw py::value_error("max_newton_iters must be positive");

  const auto* position_data = static_cast<const double*>(position_info.ptr);
  const auto* velocity_data = static_cast<const double*>(velocity_info.ptr);
  const auto* integral_data = static_cast<const double*>(integral_info.ptr);
  const auto* previous_data = static_cast<const double*>(previous_info.ptr);
  const auto* nodal_data = static_cast<const double*>(nodal_info.ptr);
  const auto* prescribed_data = static_cast<const double*>(prescribed_info.ptr);
  const auto* curvature_data = static_cast<const double*>(curvature_info.ptr);
  const auto* slip_data = static_cast<const double*>(slip_info.ptr);
  const auto* anchor_data = static_cast<const double*>(anchor_info.ptr);
  const auto* contact_data = static_cast<const bool*>(contact_info.ptr);
  const auto* accepted_data = static_cast<const double*>(accepted_info.ptr);
  require_finite_values(position_data, 2 * nodes, "positions");
  require_finite_values(velocity_data, 2 * nodes, "velocities");
  require_finite_values(integral_data, nodes, "old_integrals");
  require_finite_values(previous_data, nodes, "previous_errors");
  require_positive_values(nodal_data, nodes, "nodal_lengths");
  require_finite_values(prescribed_data, 2 * nodes, "prescribed");
  require_finite_values(curvature_data, nodes - 2, "reference_curvature");
  require_finite_values(slip_data, nodes, "committed_slip");
  require_finite_values(anchor_data, 2 * nodes, "increment_start_anchors");
  require_finite_values(accepted_data, nodes, "accepted_q");
  for (ssize_t node = 0; node < nodes; ++node) {
    if (accepted_data[node] < 0.0 || accepted_data[node] > length_x_)
      throw py::value_error("accepted_q must lie in [0, x_span]");
  }

  const int first_free = 2;
  const int last_free = static_cast<int>(nodes);
  const int free_count = last_free - first_free;
  std::vector<Vec2> old_positions(nodes), old_velocities(nodes),
      prescribed_positions(nodes), anchors(nodes);
  std::vector<double> integral_values(nodes), previous_values(nodes),
      nodal_values(nodes), slip_values(nodes), curvature_values(nodes - 2),
      tracking_q(nodes);
  std::vector<bool> contact_values(nodes);
  Eigen::VectorXd free(2 * free_count);
  for (int node = 0; node < nodes; ++node) {
    old_positions[node] =
        Vec2(position_data[2 * node], position_data[2 * node + 1]);
    old_velocities[node] =
        Vec2(velocity_data[2 * node], velocity_data[2 * node + 1]);
    prescribed_positions[node] =
        Vec2(prescribed_data[2 * node], prescribed_data[2 * node + 1]);
    anchors[node] = Vec2(anchor_data[2 * node], anchor_data[2 * node + 1]);
    integral_values[node] = integral_data[node];
    previous_values[node] = previous_data[node];
    nodal_values[node] = nodal_data[node];
    slip_values[node] = slip_data[node];
    contact_values[node] = contact_data[node];
    tracking_q[node] = accepted_data[node];
    if (node >= first_free)
      free.segment<2>(2 * (node - first_free)) =
          old_positions[node] + dt * old_velocities[node];
  }
  for (int vertex = 0; vertex < nodes - 2; ++vertex)
    curvature_values[vertex] = curvature_data[vertex];

  const auto unpack = [&](const Eigen::VectorXd& current_free) {
    std::vector<Vec2> result = prescribed_positions;
    for (int node = first_free; node < last_free; ++node)
      result[node] = current_free.segment<2>(2 * (node - first_free));
    return result;
  };
  const auto tracked_frame = [&](const std::vector<Vec2>& current_positions,
                                 const std::vector<double>& starts) {
    std::vector<Projection> result(nodes);
    for (int node = 0; node < nodes; ++node)
      result[node] = tracked_project_one(
          current_positions[node], starts[node], projection_trust_radius,
          amplitude_, length_x_, wave_number_);
    return result;
  };
  const auto residual = [&](const Eigen::VectorXd& current_free,
                            const std::vector<Projection>& frame,
                            const std::vector<bool>& regions,
                            bool with_jacobian, Eigen::VectorXd* value,
                            planar2d::BlockPenta2D* system) {
    const std::vector<Vec2> current_positions = unpack(current_free);
    std::vector<Vec2> elastic_gradient;
    planar2d::assemble_elastic_gradient_hessian(
        current_positions, rest_lengths_, curvature_values, first_free,
        last_free, ea_, ei_, with_jacobian, &elastic_gradient, system);
    value->setZero(2 * free_count);
    for (int node = first_free; node < last_free; ++node) {
      const FrozenExternal external = frozen_external(
          current_positions[node], frame[node], regions[node],
          integral_values[node], previous_values[node], nodal_values[node],
          slip_values[node], anchors[node], contact_values[node], dt,
          friction_enabled, amplitude_, tube_half_width_, safe_distance_,
          feeder_half_width_, feeder_contact_layer_, feeder_kp_line_,
          kp_active, ki_active, contact_force_cap_line_, friction_mu_,
          friction_kt_line_, max_integral_state_, leak_rate_,
          contact_smooth_length_, direction_epsilon_);
      const Vec2 inertial =
          masses_[node] *
              (current_positions[node] - old_positions[node] -
               dt * old_velocities[node]) /
              (dt * dt) +
          damping_ * (current_positions[node] - old_positions[node]) / dt;
      value->segment<2>(2 * (node - first_free)) =
          inertial + elastic_gradient[node] -
          (external.vessel + external.feeder + external.friction);
      if (with_jacobian) {
        system->diagonal[node - first_free] +=
            (masses_[node] / (dt * dt) + damping_ / dt) *
                planar2d::Mat2::Identity() -
            external.jacobian;
      }
    }
  };

  int line_search_failures = 0;
  int dense_fallback_count = 0;
  int iteration = 0;
  double residual_norm = std::numeric_limits<double>::infinity();
  double step_norm = std::numeric_limits<double>::infinity();
  bool converged = false;
  std::vector<Projection> frame(nodes);
  std::vector<bool> frame_regions(nodes);
  for (iteration = 1; iteration <= max_newton_iters; ++iteration) {
    const std::vector<Vec2> current_positions = unpack(free);
    frame = tracked_frame(current_positions, tracking_q);
    for (int node = 0; node < nodes; ++node)
      frame_regions[node] = current_positions[node].x() >= 0.0;
    Eigen::VectorXd current_residual;
    planar2d::BlockPenta2D system(free_count);
    residual(free, frame, frame_regions, true, &current_residual, &system);
    residual_norm = current_residual.norm();
    if (residual_norm < residual_tol) {
      converged = true;
      step_norm = 0.0;
      for (int node = 0; node < nodes; ++node) tracking_q[node] = frame[node].q;
      break;
    }
    Eigen::VectorXd delta;
    double relative_residual = std::numeric_limits<double>::infinity();
    if (!system.solve(-current_residual, &delta, &relative_residual)) {
      ++dense_fallback_count;
      const Eigen::MatrixXd dense = system.dense();
      delta = dense.partialPivLu().solve(-current_residual);
      const Eigen::VectorXd difference = dense * delta + current_residual;
      const double dense_relative =
          difference.norm() / std::max(current_residual.norm(), 1.0e-30);
      if (!delta.allFinite() || !std::isfinite(dense_relative) ||
          dense_relative > 1.0e-8)
        throw std::runtime_error("Sine2D Newton linear solve failed");
    }
    step_norm = delta.norm();
    const double base_norm = residual_norm;
    bool accepted = false;
    for (int exponent = 0; exponent < 10; ++exponent) {
      const Eigen::VectorXd candidate =
          free + std::ldexp(1.0, -exponent) * delta;
      Eigen::VectorXd candidate_residual;
      planar2d::BlockPenta2D ignored(free_count);
      residual(candidate, frame, frame_regions, false, &candidate_residual,
               &ignored);
      const double candidate_norm = candidate_residual.norm();
      if (std::isfinite(candidate_norm) && candidate_norm < base_norm) {
        free = candidate;
        residual_norm = candidate_norm;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      ++line_search_failures;
      const Eigen::VectorXd forced = free + 0.05 * delta;
      Eigen::VectorXd forced_residual;
      planar2d::BlockPenta2D ignored(free_count);
      residual(forced, frame, frame_regions, false, &forced_residual,
               &ignored);
      const double forced_norm = forced_residual.norm();
      if (std::isfinite(forced_norm)) {
        free = forced;
        residual_norm = forced_norm;
      } else {
        // Do not move to an unevaluable iterate or use the pre-step residual
        // for the relaxed small-step convergence check below.
        residual_norm = std::numeric_limits<double>::infinity();
      }
    }
    for (int node = 0; node < nodes; ++node) tracking_q[node] = frame[node].q;
    if (step_norm < step_tol) {
      converged = residual_norm < 10.0 * residual_tol;
      break;
    }
  }
  if (iteration > max_newton_iters) iteration = max_newton_iters;

  const std::vector<Vec2> output_positions = unpack(free);
  frame = tracked_frame(output_positions, tracking_q);
  for (int node = 0; node < nodes; ++node) {
    frame_regions[node] = output_positions[node].x() >= 0.0;
    tracking_q[node] = frame[node].q;
  }
  std::vector<Vec2> output_velocities(nodes, Vec2::Zero());
  std::vector<Vec2> dynamic_force(nodes, Vec2::Zero());
  double max_contact_error = 0.0;
  for (int node = 0; node < nodes; ++node) {
    if (node >= first_free)
      output_velocities[node] =
          (output_positions[node] - old_positions[node]) / dt;
    dynamic_force[node] =
        masses_[node] *
            (output_positions[node] - old_positions[node] -
             dt * old_velocities[node]) /
            (dt * dt) +
        damping_ * (output_positions[node] - old_positions[node]) / dt;
    const double intrusion =
        frame_regions[node]
            ? std::max(std::abs((output_positions[node] - frame[node].center)
                                    .dot(frame[node].normal)) -
                           (tube_half_width_ - safe_distance_),
                       0.0)
            : std::max(std::abs(output_positions[node].y() - amplitude_) -
                           (feeder_half_width_ - feeder_contact_layer_),
                       0.0);
    max_contact_error = std::max(max_contact_error, intrusion);
  }
  py::array_t<double> q_array(nodes);
  auto* q_data = static_cast<double*>(q_array.request().ptr);
  for (int node = 0; node < nodes; ++node) q_data[node] = tracking_q[node];
  std::vector<Vec2> final_normal(nodes, Vec2::Zero());
  std::vector<Vec2> final_feeder(nodes, Vec2::Zero());
  std::vector<Vec2> final_friction(nodes, Vec2::Zero());
  std::vector<Vec2> final_center(nodes, Vec2::Zero());
  std::vector<Vec2> final_tangent(nodes, Vec2::Zero());
  std::vector<Vec2> final_frame_normal(nodes, Vec2::Zero());
  py::array_t<double> final_integrals(nodes), final_errors(nodes),
      final_trial(nodes), final_candidate(nodes), final_traction(nodes),
      final_limit(nodes);
  py::array_t<bool> final_active(nodes), final_region(nodes),
      final_stick(nodes), final_slip_state(nodes);
  auto* final_integral_data =
      static_cast<double*>(final_integrals.request().ptr);
  auto* final_error_data = static_cast<double*>(final_errors.request().ptr);
  auto* final_trial_data = static_cast<double*>(final_trial.request().ptr);
  auto* final_candidate_data =
      static_cast<double*>(final_candidate.request().ptr);
  auto* final_traction_data =
      static_cast<double*>(final_traction.request().ptr);
  auto* final_limit_data = static_cast<double*>(final_limit.request().ptr);
  auto* final_active_data = static_cast<bool*>(final_active.request().ptr);
  auto* final_region_data = static_cast<bool*>(final_region.request().ptr);
  auto* final_stick_data = static_cast<bool*>(final_stick.request().ptr);
  auto* final_slip_data =
      static_cast<bool*>(final_slip_state.request().ptr);
  for (int node = 0; node < nodes; ++node) {
    const FrozenExternal value = frozen_external(
        output_positions[node], frame[node], frame_regions[node],
        integral_values[node], previous_values[node], nodal_values[node],
        slip_values[node], anchors[node], contact_values[node], dt,
        friction_enabled, amplitude_, tube_half_width_, safe_distance_,
        feeder_half_width_, feeder_contact_layer_, feeder_kp_line_, kp_active,
        ki_active, contact_force_cap_line_, friction_mu_, friction_kt_line_,
        max_integral_state_, leak_rate_, contact_smooth_length_,
        direction_epsilon_);
    final_normal[node] = value.vessel;
    final_feeder[node] = value.feeder;
    final_friction[node] = value.friction;
    final_center[node] = frame[node].center;
    final_tangent[node] = frame[node].tangent;
    final_frame_normal[node] = frame[node].normal;
    final_integral_data[node] = value.integral;
    final_error_data[node] = value.error;
    final_trial_data[node] = value.trial_slip;
    final_candidate_data[node] = value.candidate_slip;
    final_traction_data[node] = value.traction;
    final_limit_data[node] = value.limit;
    final_active_data[node] = value.active;
    final_region_data[node] = frame_regions[node];
    final_stick_data[node] = value.stick;
    final_slip_data[node] = value.slip;
  }

  py::dict result;
  result["positions"] = vec2_array(output_positions);
  result["velocities"] = vec2_array(output_velocities);
  result["dynamic_force"] = vec2_array(dynamic_force);
  result["integrals"] = final_integrals;
  result["previous_errors"] = final_errors;
  result["normal_force"] = vec2_array(final_normal);
  result["feeder_force"] = vec2_array(final_feeder);
  result["friction_force"] = vec2_array(final_friction);
  result["active"] = final_active;
  result["q"] = q_array;
  result["center"] = vec2_array(final_center);
  result["tangent"] = vec2_array(final_tangent);
  result["normal"] = vec2_array(final_frame_normal);
  result["region"] = final_region;
  result["candidate_slip"] = final_candidate;
  result["trial_slip"] = final_trial;
  result["traction"] = final_traction;
  result["stick"] = final_stick;
  result["slip"] = final_slip_state;
  result["limit"] = final_limit;
  result["max_contact_error"] = max_contact_error;
  result["converged"] = converged;
  result["iterations"] = iteration;
  result["residual_norm"] = residual_norm;
  result["step_norm"] = step_norm;
  result["line_search_failures"] = line_search_failures;
  result["dense_fallback_count"] = dense_fallback_count;
  return result;
}

}  // namespace der
