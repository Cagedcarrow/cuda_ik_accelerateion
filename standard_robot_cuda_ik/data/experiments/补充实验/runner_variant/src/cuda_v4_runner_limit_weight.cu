#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_DEFINE_CONSTANTS
#include "cuda_utilities.cuh"
#include "standard_robot_cuda_ik/generated/ur10_model_constants.h"

namespace {

constexpr double kStrictPos = 0.005;
constexpr double kStrictRot = 0.01745;
constexpr double kMediumPos = 0.010;
constexpr double kMediumRot = 0.0873;
constexpr double kLoosePos = 0.030;
constexpr double kLooseRot = 0.1745;
constexpr double kLimitMargin = 0.087;
__constant__ double c_limit_weight;
constexpr int kCandidateStride = 16;
constexpr int kBestStride = 18;

cudaError_t upload_constants(double limit_weight) {
  cudaError_t err;
  err = cudaMemcpyToSymbol(c_segment_origins, k_origins, sizeof(k_origins));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_segment_axes, k_axes, sizeof(k_axes));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_q_index, k_q_index, sizeof(k_q_index));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_T_wrist3_to_tcp, k_T_wrist3_to_tcp, sizeof(k_T_wrist3_to_tcp));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_joint_limits, k_joint_limits, sizeof(k_joint_limits));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_weight_schedule, k_weights, sizeof(k_weights));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_lambda_params, k_lambda_params, sizeof(k_lambda_params));
  if (err != cudaSuccess) return err;
  err = cudaMemcpyToSymbol(c_limit_weight, &limit_weight, sizeof(double));
  return err;
}

bool read_raw_doubles(const char* path, std::vector<double>* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path);
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes < 0 || bytes % static_cast<long>(sizeof(double)) != 0) {
    std::fprintf(stderr, "ERROR: invalid double raw file %s\n", path);
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(bytes) / sizeof(double));
  const size_t n = std::fread(out->data(), sizeof(double), out->size(), f);
  std::fclose(f);
  if (n != out->size()) {
    std::fprintf(stderr, "ERROR: short read %s\n", path);
    return false;
  }
  return true;
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return values[lo];
  const double t = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

bool write_best_csv(const char* path, const std::vector<double>& best, int N) {
  FILE* f = std::fopen(path, "w");
  if (!f) return false;
  std::fprintf(f, "target_id,best_seed_id,q0,q1,q2,q3,q4,q5,pos_err_mm,rot_err_deg,pose_cost,limit_score,total_loss,iters,success_loose,success_medium,success_strict,near_limit,success_rank\n");
  for (int i = 0; i < N; ++i) {
    const double* b = &best[i * kBestStride];
    std::fprintf(f, "%d,%.0f", i, b[6]);
    for (int j = 0; j < 6; ++j) std::fprintf(f, ",%.17g", b[j]);
    std::fprintf(f, ",%.12g,%.12g,%.17g,%.17g,%.17g,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
                 b[7] * 1000.0, b[8] * 180.0 / M_PI, b[9], b[10], b[11],
                 b[12], b[13], b[14], b[15], b[16], b[17]);
  }
  std::fclose(f);
  return true;
}

bool write_candidates_csv(const char* path, const std::vector<double>& cand, int N, int K) {
  FILE* f = std::fopen(path, "w");
  if (!f) return false;
  std::fprintf(f, "target_id,seed_id,q0,q1,q2,q3,q4,q5,pos_err_mm,rot_err_deg,pose_cost,limit_score,total_loss,iters,success_loose,success_medium,success_strict,near_limit\n");
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      const double* c = &cand[(i * K + k) * kCandidateStride];
      std::fprintf(f, "%d,%d", i, k);
      for (int j = 0; j < 6; ++j) std::fprintf(f, ",%.17g", c[j]);
      std::fprintf(f, ",%.12g,%.12g,%.17g,%.17g,%.17g,%.0f,%.0f,%.0f,%.0f,%.0f\n",
                   c[6] * 1000.0, c[7] * 180.0 / M_PI, c[8], c[9], c[10],
                   c[11], c[12], c[13], c[14], c[15]);
    }
  }
  std::fclose(f);
  return true;
}

bool write_summary_csv(const char* path, const char* method, int N, int K,
                       int warmup, int repeat, const std::vector<double>& kernel_ms,
                       const std::vector<double>& best) {
  std::vector<double> pos, rot, pos_suc, rot_suc, iters;
  pos.reserve(N);
  rot.reserve(N);
  int loose = 0, medium = 0, strict = 0, near_limit = 0, nan_count = 0, inf_count = 0;
  for (int i = 0; i < N; ++i) {
    const double* b = &best[i * kBestStride];
    const double p = b[7] * 1000.0;
    const double r = b[8] * 180.0 / M_PI;
    if (std::isnan(p) || std::isnan(r)) nan_count++;
    if (std::isinf(p) || std::isinf(r)) inf_count++;
    pos.push_back(p);
    rot.push_back(r);
    iters.push_back(b[12]);
    if (b[13] > 0.5) loose++;
    if (b[14] > 0.5) medium++;
    if (b[15] > 0.5) {
      strict++;
      pos_suc.push_back(p);
      rot_suc.push_back(r);
    }
    if (b[16] > 0.5) near_limit++;
  }
  double mean_ms = 0.0;
  for (double x : kernel_ms) mean_ms += x;
  mean_ms /= std::max<size_t>(1, kernel_ms.size());
  double var_ms = 0.0;
  for (double x : kernel_ms) {
    const double d = x - mean_ms;
    var_ms += d * d;
  }
  const double std_ms = std::sqrt(var_ms / std::max<size_t>(1, kernel_ms.size()));
  const double throughput = mean_ms > 0.0 ? 1000.0 * static_cast<double>(N) / mean_ms : 0.0;
  FILE* f = std::fopen(path, "w");
  if (!f) return false;
  std::fprintf(f, "method,N,K,warmup,repeat,gpu_stream_ms_mean,gpu_stream_ms_std,e2e_ms_mean,e2e_ms_std,raw_throughput_mean,raw_throughput_std,valid_throughput_strict,loose_sr,medium_sr,strict_sr,pos_p50_all_mm,pos_p95_all_mm,pos_p99_all_mm,pos_max_all_mm,pos_p95_suc_mm,rot_p50_all_deg,rot_p95_all_deg,rot_p95_suc_deg,near_limit_ratio,iter_mean,iter_p95,monotonic_pass,nan_count,inf_count\n");
  std::fprintf(f, "%s,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,0,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d\n",
               method, N, K, warmup, repeat, mean_ms, std_ms, mean_ms, std_ms,
               throughput, throughput * static_cast<double>(strict) / static_cast<double>(N),
               static_cast<double>(loose) / N, static_cast<double>(medium) / N,
               static_cast<double>(strict) / N, percentile(pos, 50), percentile(pos, 95),
               percentile(pos, 99), *std::max_element(pos.begin(), pos.end()),
               pos_suc.empty() ? 0.0 : percentile(pos_suc, 95), percentile(rot, 50),
               percentile(rot, 95), rot_suc.empty() ? 0.0 : percentile(rot_suc, 95),
               static_cast<double>(near_limit) / N, percentile(iters, 50), percentile(iters, 95),
               (loose >= medium && medium >= strict) ? 1 : 0, nan_count, inf_count);
  std::fclose(f);
  return true;
}

bool write_timing_csv(const char* path,
                      const std::vector<double>& host_prepare_ms,
                      const std::vector<double>& h2d_ms,
                      const std::vector<double>& launch_overhead_ms,
                      const std::vector<double>& gpu_ms,
                      const std::vector<double>& d2h_ms,
                      const std::vector<double>& e2e_ms,
                      const std::vector<int>& fallback_count) {
  FILE* f = std::fopen(path, "w");
  if (!f) return false;
  std::fprintf(f, "repeat_id,host_prepare_ms,h2d_ms,graph_launch_or_kernel_launch_ms,gpu_stream_ms,d2h_ms,e2e_ms,fallback_count\n");
  for (size_t i = 0; i < gpu_ms.size(); ++i) {
    std::fprintf(f, "%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
                 i,
                 i < host_prepare_ms.size() ? host_prepare_ms[i] : 0.0,
                 i < h2d_ms.size() ? h2d_ms[i] : 0.0,
                 i < launch_overhead_ms.size() ? launch_overhead_ms[i] : 0.0,
                 gpu_ms[i],
                 i < d2h_ms.size() ? d2h_ms[i] : 0.0,
                 i < e2e_ms.size() ? e2e_ms[i] : gpu_ms[i],
                 i < fallback_count.size() ? fallback_count[i] : 0);
  }
  std::fclose(f);
  return true;
}

}  // namespace

namespace rtfg {
namespace cuda {

__device__ __forceinline__ void build_rotation_matrix_v4(
    double ax, double ay, double az, double angle, double* R) {
  const double c = cos(angle);
  const double s = sin(angle);
  const double t = 1.0 - c;
  R[0] = t * ax * ax + c;
  R[1] = t * ax * ay - s * az;
  R[2] = t * ax * az + s * ay;
  R[3] = 0.0;
  R[4] = t * ax * ay + s * az;
  R[5] = t * ay * ay + c;
  R[6] = t * ay * az - s * ax;
  R[7] = 0.0;
  R[8] = t * ax * az - s * ay;
  R[9] = t * ay * az + s * ax;
  R[10] = t * az * az + c;
  R[11] = 0.0;
  R[12] = 0.0;
  R[13] = 0.0;
  R[14] = 0.0;
  R[15] = 1.0;
}

__device__ __forceinline__ void fk_with_frames_v4(const double* q, double* T_tip,
                                                   double* p_joint, double* z_joint) {
  for (int i = 0; i < 16; ++i) T_tip[i] = (i % 5 == 0) ? 1.0 : 0.0;
  double T_tmp[16], R[16];
  for (int seg = 0; seg < 6; ++seg) {
    mat44_mul(T_tip, &c_segment_origins[seg * 16], T_tmp);
    for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];

    p_joint[seg * 3 + 0] = T_tip[3];
    p_joint[seg * 3 + 1] = T_tip[7];
    p_joint[seg * 3 + 2] = T_tip[11];

    const double ax = c_segment_axes[seg * 3 + 0];
    const double ay = c_segment_axes[seg * 3 + 1];
    const double az = c_segment_axes[seg * 3 + 2];
    z_joint[seg * 3 + 0] = T_tip[0] * ax + T_tip[1] * ay + T_tip[2] * az;
    z_joint[seg * 3 + 1] = T_tip[4] * ax + T_tip[5] * ay + T_tip[6] * az;
    z_joint[seg * 3 + 2] = T_tip[8] * ax + T_tip[9] * ay + T_tip[10] * az;

    build_rotation_matrix_v4(ax, ay, az, q[c_q_index[seg]], R);
    mat44_mul(T_tip, R, T_tmp);
    for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
  }
  mat44_mul(T_tip, c_T_wrist3_to_tcp, T_tmp);
  for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
}

__device__ __forceinline__ void pose_error_v4(const double* T_cur, const double* T_tgt, double* err) {
  err[0] = T_cur[3] - T_tgt[3];
  err[1] = T_cur[7] - T_tgt[7];
  err[2] = T_cur[11] - T_tgt[11];

  double r00 = T_cur[0], r01 = T_cur[1], r02 = T_cur[2];
  double r10 = T_cur[4], r11 = T_cur[5], r12 = T_cur[6];
  double r20 = T_cur[8], r21 = T_cur[9], r22 = T_cur[10];
  double t00 = T_tgt[0], t01 = T_tgt[1], t02 = T_tgt[2];
  double t10 = T_tgt[4], t11 = T_tgt[5], t12 = T_tgt[6];
  double t20 = T_tgt[8], t21 = T_tgt[9], t22 = T_tgt[10];

  double e01 = r00 * t01 + r10 * t11 + r20 * t21;
  double e02 = r00 * t02 + r10 * t12 + r20 * t22;
  double e10 = r01 * t00 + r11 * t10 + r21 * t20;
  double e12 = r01 * t02 + r11 * t12 + r21 * t22;
  double e20 = r02 * t00 + r12 * t10 + r22 * t20;
  double e21 = r02 * t01 + r12 * t11 + r22 * t21;
  err[3] = 0.5 * (e21 - e12);
  err[4] = 0.5 * (e02 - e20);
  err[5] = 0.5 * (e10 - e01);
}

__device__ __forceinline__ void analytical_jacobian_v4(const double* T_ee,
                                                        const double* p_joint,
                                                        const double* z_joint,
                                                        double* J) {
  const double pe[3] = {T_ee[3], T_ee[7], T_ee[11]};
  for (int j = 0; j < 6; ++j) {
    const double zx = z_joint[j * 3 + 0];
    const double zy = z_joint[j * 3 + 1];
    const double zz = z_joint[j * 3 + 2];
    const double rx = pe[0] - p_joint[j * 3 + 0];
    const double ry = pe[1] - p_joint[j * 3 + 1];
    const double rz = pe[2] - p_joint[j * 3 + 2];
    J[0 * 6 + j] = zy * rz - zz * ry;
    J[1 * 6 + j] = zz * rx - zx * rz;
    J[2 * 6 + j] = zx * ry - zy * rx;
    J[3 * 6 + j] = zx;
    J[4 * 6 + j] = zy;
    J[5 * 6 + j] = zz;
  }
}

__device__ __forceinline__ double limit_loss_v4(const double* q) {
  double loss = 0.0;
  for (int j = 0; j < 6; ++j) {
    const double lo = c_joint_limits[j * 2 + 0];
    const double hi = c_joint_limits[j * 2 + 1];
    const double dl = q[j] - lo;
    const double du = hi - q[j];
    if (dl < kLimitMargin) {
      const double d = kLimitMargin - dl;
      loss += d * d;
    }
    if (du < kLimitMargin) {
      const double d = kLimitMargin - du;
      loss += d * d;
    }
  }
  return loss;
}

__device__ __forceinline__ double limit_loss_grad_analytic_v4(const double* q, double* grad) {
  double loss = 0.0;
  for (int j = 0; j < 6; ++j) grad[j] = 0.0;
  for (int j = 0; j < 6; ++j) {
    const double lo = c_joint_limits[j * 2 + 0];
    const double hi = c_joint_limits[j * 2 + 1];
    const double dl = q[j] - lo;
    if (dl < kLimitMargin) {
      const double v = kLimitMargin - dl;
      loss += v * v;
      grad[j] += -2.0 * v;
    }
    const double du = hi - q[j];
    if (du < kLimitMargin) {
      const double v = kLimitMargin - du;
      loss += v * v;
      grad[j] += 2.0 * v;
    }
  }
  return loss;
}

__device__ __forceinline__ int near_limit_v4(const double* q) {
  for (int j = 0; j < 6; ++j) {
    const double lo = c_joint_limits[j * 2 + 0];
    const double hi = c_joint_limits[j * 2 + 1];
    if (q[j] - lo < kLimitMargin || hi - q[j] < kLimitMargin) return 1;
  }
  return 0;
}

__device__ __forceinline__ int success_rank_v4(double pos, double rot) {
  if (pos < kStrictPos && rot < kStrictRot) return 0;
  if (pos < kMediumPos && rot < kMediumRot) return 1;
  if (pos < kLoosePos && rot < kLooseRot) return 2;
  return 3;
}

__device__ __forceinline__ void solve_6x6_gauss_v4(double* A, double* b, double* x) {
  double M[6][7];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) M[r][c] = A[r * 6 + c];
    M[r][6] = b[r];
  }
  for (int k = 0; k < 6; ++k) {
    int piv = k;
    double best = fabs(M[k][k]);
    for (int r = k + 1; r < 6; ++r) {
      const double v = fabs(M[r][k]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != k) {
      for (int c = k; c < 7; ++c) {
        const double tmp = M[k][c];
        M[k][c] = M[piv][c];
        M[piv][c] = tmp;
      }
    }
    double diag = M[k][k];
    if (fabs(diag) < 1e-14) diag = (diag < 0.0 ? -1e-14 : 1e-14);
    for (int r = k + 1; r < 6; ++r) {
      const double f = M[r][k] / diag;
      for (int c = k; c < 7; ++c) M[r][c] -= f * M[k][c];
    }
  }
  for (int r = 5; r >= 0; --r) {
    double sum = M[r][6];
    for (int c = r + 1; c < 6; ++c) sum -= M[r][c] * x[c];
    double diag = M[r][r];
    if (fabs(diag) < 1e-14) diag = (diag < 0.0 ? -1e-14 : 1e-14);
    x[r] = sum / diag;
  }
}

__device__ __forceinline__ int candidate_rank_v4(const double* c) {
  return c[14] > 0.5 ? 0 : (c[13] > 0.5 ? 1 : (c[12] > 0.5 ? 2 : 3));
}

__device__ __forceinline__ bool candidate_better_v4(const double* c, int seed_id,
                                                     int best_rank, int best_near,
                                                     double best_pose, int best_seed) {
  const int rank = candidate_rank_v4(c);
  const int near = c[15] > 0.5 ? 1 : 0;
  const double pose = c[8];
  return rank < best_rank ||
         (rank == best_rank && near < best_near) ||
         (rank == best_rank && near == best_near && pose < best_pose) ||
         (rank == best_rank && near == best_near && pose == best_pose && seed_id < best_seed);
}

__device__ __forceinline__ void write_best_from_candidate_v4(const double* c, int best_k,
                                                              double* b) {
  const int rank = candidate_rank_v4(c);
  for (int j = 0; j < 6; ++j) b[j] = c[j];
  b[6] = static_cast<double>(best_k);
  b[7] = c[6];
  b[8] = c[7];
  b[9] = c[8];
  b[10] = c[9];
  b[11] = c[10];
  b[12] = c[11];
  b[13] = c[12];
  b[14] = c[13];
  b[15] = c[14];
  b[16] = c[15];
  b[17] = static_cast<double>(rank);
}

__device__ __forceinline__ void solve_candidate_v4(const double* T_tgt, const double* q_seed,
                                                    int max_iter, int limit_gradient_mode,
                                                    int precision_mode, double* out) {
  double q[6];
  for (int j = 0; j < 6; ++j) q[j] = q_seed[j];
  double lambda = 1e-2;
  double T[16], p[18], z[18], J[36], e[6], H[36], g[6], dq[6];
  int iters = max_iter;

  for (int iter = 0; iter < max_iter; ++iter) {
    if (precision_mode >= 3) {
      for (int j = 0; j < 6; ++j) q[j] = static_cast<double>(static_cast<float>(q[j]));
    }
    fk_with_frames_v4(q, T, p, z);
    pose_error_v4(T, T_tgt, e);
    const double pos = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    const double rot = sqrt(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    if (pos < kStrictPos && rot < kStrictRot) {
      iters = iter + 1;
      break;
    }
    analytical_jacobian_v4(T, p, z, J);
    if (precision_mode <= 0) {
      for (int c = 0; c < 6; ++c) {
        double sum = 0.0;
        for (int r = 0; r < 6; ++r) sum += J[r * 6 + c] * e[r];
        g[c] = sum;
      }
    } else {
      float Jf[36], ef[6];
      for (int i = 0; i < 36; ++i) Jf[i] = static_cast<float>(J[i]);
      for (int i = 0; i < 6; ++i) ef[i] = static_cast<float>(e[i]);
      for (int c = 0; c < 6; ++c) {
        float sum = 0.0f;
        for (int r = 0; r < 6; ++r) sum += Jf[r * 6 + c] * ef[r];
        g[c] = static_cast<double>(sum);
      }
    }
    if (c_limit_weight > 0.0 && limit_gradient_mode == 1) {
      double gl[6];
      limit_loss_grad_analytic_v4(q, gl);
      for (int j = 0; j < 6; ++j) g[j] += c_limit_weight * gl[j];
    } else if (c_limit_weight > 0.0) {
      const double eps = 1e-6;
      const double l0 = limit_loss_v4(q);
      for (int j = 0; j < 6; ++j) {
        double qp[6];
        for (int a = 0; a < 6; ++a) qp[a] = q[a];
        qp[j] += eps;
        g[j] += c_limit_weight * (limit_loss_v4(qp) - l0) / eps;
      }
    }
    if (precision_mode <= 0) {
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
          double sum = 0.0;
          for (int a = 0; a < 6; ++a) sum += J[a * 6 + r] * J[a * 6 + c];
          H[r * 6 + c] = sum + (r == c ? lambda : 0.0);
        }
      }
    } else {
      float Jf[36];
      for (int i = 0; i < 36; ++i) Jf[i] = static_cast<float>(J[i]);
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
          float sum = 0.0f;
          for (int a = 0; a < 6; ++a) sum += Jf[a * 6 + r] * Jf[a * 6 + c];
          H[r * 6 + c] = static_cast<double>(sum) + (r == c ? lambda : 0.0);
        }
      }
    }
    double rhs[6];
    for (int j = 0; j < 6; ++j) rhs[j] = -g[j];
    solve_6x6_gauss_v4(H, rhs, dq);
    double max_abs = 0.0;
    for (int j = 0; j < 6; ++j) max_abs = fmax(max_abs, fabs(dq[j]));
    if (max_abs > 0.35) {
      const double s = 0.35 / max_abs;
      for (int j = 0; j < 6; ++j) dq[j] *= s;
    }

    double q_trial[6];
    for (int j = 0; j < 6; ++j) {
      const double lo = c_joint_limits[j * 2 + 0];
      const double hi = c_joint_limits[j * 2 + 1];
      q_trial[j] = cuda_clamp(q[j] + dq[j], lo, hi);
      if (precision_mode >= 3) q_trial[j] = static_cast<double>(static_cast<float>(q_trial[j]));
    }

    double loss_old = 0.5 * (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] +
                             e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    if (c_limit_weight > 0.0) loss_old += c_limit_weight * limit_loss_v4(q);

    double T_trial[16], p_trial[18], z_trial[18], e_trial[6];
    fk_with_frames_v4(q_trial, T_trial, p_trial, z_trial);
    pose_error_v4(T_trial, T_tgt, e_trial);
    double loss_new = 0.5 * (e_trial[0] * e_trial[0] + e_trial[1] * e_trial[1] +
                             e_trial[2] * e_trial[2] + e_trial[3] * e_trial[3] +
                             e_trial[4] * e_trial[4] + e_trial[5] * e_trial[5]);
    if (c_limit_weight > 0.0) loss_new += c_limit_weight * limit_loss_v4(q_trial);

    for (int j = 0; j < 6; ++j) q[j] = q_trial[j];
    lambda *= (loss_new < loss_old) ? 0.5 : 2.0;
    lambda = cuda_clamp(lambda, 1e-6, 0.5);
  }

  fk_with_frames_v4(q, T, p, z);
  pose_error_v4(T, T_tgt, e);
  const double pos = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  const double rot = sqrt(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
  const double pose_cost = pos * pos + rot * rot;
  const double limit_score = limit_loss_v4(q);
  const double total_loss = 0.5 * pose_cost + c_limit_weight * limit_score;
  const int rank = success_rank_v4(pos, rot);
  for (int j = 0; j < 6; ++j) out[j] = q[j];
  out[6] = pos;
  out[7] = rot;
  out[8] = pose_cost;
  out[9] = limit_score;
  out[10] = total_loss;
  out[11] = static_cast<double>(iters);
  out[12] = (rank <= 2) ? 1.0 : 0.0;
  out[13] = (rank <= 1) ? 1.0 : 0.0;
  out[14] = (rank == 0) ? 1.0 : 0.0;
  out[15] = static_cast<double>(near_limit_v4(q));
}

__global__ void ik_lm_multiseed_v4_kernel(const double* targets, const double* seeds,
                                          double* candidates, int N, int K, int max_iter,
                                          int limit_gradient_mode) {
  const int target_id = blockIdx.x;
  const int seed_id = blockIdx.y;
  if (target_id >= N || seed_id >= K || threadIdx.x != 0) return;

  const double* T_tgt = targets + target_id * 16;
  const double* q_seed = seeds + (target_id * K + seed_id) * 6;
  double q[6];
  for (int j = 0; j < 6; ++j) q[j] = q_seed[j];
  double lambda = 1e-2;
  double T[16], p[18], z[18], J[36], e[6], H[36], g[6], dq[6];
  int iters = max_iter;

  for (int iter = 0; iter < max_iter; ++iter) {
    fk_with_frames_v4(q, T, p, z);
    pose_error_v4(T, T_tgt, e);
    const double pos = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    const double rot = sqrt(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    if (pos < kStrictPos && rot < kStrictRot) {
      iters = iter + 1;
      break;
    }
    analytical_jacobian_v4(T, p, z, J);
    for (int c = 0; c < 6; ++c) {
      double sum = 0.0;
      for (int r = 0; r < 6; ++r) sum += J[r * 6 + c] * e[r];
      g[c] = sum;
    }
    if (c_limit_weight > 0.0 && limit_gradient_mode == 1) {
      double gl[6];
      limit_loss_grad_analytic_v4(q, gl);
      for (int j = 0; j < 6; ++j) g[j] += c_limit_weight * gl[j];
    } else if (c_limit_weight > 0.0) {
      const double eps = 1e-6;
      const double l0 = limit_loss_v4(q);
      for (int j = 0; j < 6; ++j) {
        double qp[6];
        for (int a = 0; a < 6; ++a) qp[a] = q[a];
        qp[j] += eps;
        g[j] += c_limit_weight * (limit_loss_v4(qp) - l0) / eps;
      }
    }
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        double sum = 0.0;
        for (int a = 0; a < 6; ++a) sum += J[a * 6 + r] * J[a * 6 + c];
        H[r * 6 + c] = sum + (r == c ? lambda : 0.0);
      }
    }
    double rhs[6];
    for (int j = 0; j < 6; ++j) rhs[j] = -g[j];
    solve_6x6_gauss_v4(H, rhs, dq);
    double max_abs = 0.0;
    for (int j = 0; j < 6; ++j) max_abs = fmax(max_abs, fabs(dq[j]));
    if (max_abs > 0.35) {
      const double s = 0.35 / max_abs;
      for (int j = 0; j < 6; ++j) dq[j] *= s;
    }

    double q_trial[6];
    for (int j = 0; j < 6; ++j) {
      const double lo = c_joint_limits[j * 2 + 0];
      const double hi = c_joint_limits[j * 2 + 1];
      q_trial[j] = cuda_clamp(q[j] + dq[j], lo, hi);
    }

    double loss_old = 0.5 * (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] +
                             e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    if (c_limit_weight > 0.0) loss_old += c_limit_weight * limit_loss_v4(q);

    double T_trial[16], p_trial[18], z_trial[18], e_trial[6];
    fk_with_frames_v4(q_trial, T_trial, p_trial, z_trial);
    pose_error_v4(T_trial, T_tgt, e_trial);
    double loss_new = 0.5 * (e_trial[0] * e_trial[0] + e_trial[1] * e_trial[1] +
                             e_trial[2] * e_trial[2] + e_trial[3] * e_trial[3] +
                             e_trial[4] * e_trial[4] + e_trial[5] * e_trial[5]);
    if (c_limit_weight > 0.0) loss_new += c_limit_weight * limit_loss_v4(q_trial);

    for (int j = 0; j < 6; ++j) q[j] = q_trial[j];
    lambda *= (loss_new < loss_old) ? 0.5 : 2.0;
    lambda = cuda_clamp(lambda, 1e-6, 0.5);
  }

  fk_with_frames_v4(q, T, p, z);
  pose_error_v4(T, T_tgt, e);
  const double pos = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  const double rot = sqrt(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
  const double pose_cost = pos * pos + rot * rot;
  const double limit_score = limit_loss_v4(q);
  const double total_loss = 0.5 * pose_cost + c_limit_weight * limit_score;
  const int rank = success_rank_v4(pos, rot);
  const int out = (target_id * K + seed_id) * kCandidateStride;
  for (int j = 0; j < 6; ++j) candidates[out + j] = q[j];
  candidates[out + 6] = pos;
  candidates[out + 7] = rot;
  candidates[out + 8] = pose_cost;
  candidates[out + 9] = limit_score;
  candidates[out + 10] = total_loss;
  candidates[out + 11] = static_cast<double>(iters);
  candidates[out + 12] = (rank <= 2) ? 1.0 : 0.0;
  candidates[out + 13] = (rank <= 1) ? 1.0 : 0.0;
  candidates[out + 14] = (rank == 0) ? 1.0 : 0.0;
  candidates[out + 15] = static_cast<double>(near_limit_v4(q));
}

__global__ void select_best_per_target_v4_kernel(const double* candidates, double* best, int N, int K) {
  const int target_id = blockIdx.x;
  if (target_id >= N || threadIdx.x != 0) return;
  int best_k = 0;
  int best_rank = 4;
  int best_near = 2;
  double best_pose = 1e300;
  for (int k = 0; k < K; ++k) {
    const double* c = candidates + (target_id * K + k) * kCandidateStride;
    const int rank = c[14] > 0.5 ? 0 : (c[13] > 0.5 ? 1 : (c[12] > 0.5 ? 2 : 3));
    const int near = c[15] > 0.5 ? 1 : 0;
    const double pose = c[8];
    if (rank < best_rank ||
        (rank == best_rank && near < best_near) ||
        (rank == best_rank && near == best_near && pose < best_pose)) {
      best_rank = rank;
      best_near = near;
      best_pose = pose;
      best_k = k;
    }
  }
  const double* c = candidates + (target_id * K + best_k) * kCandidateStride;
  double* b = best + target_id * kBestStride;
  for (int j = 0; j < 6; ++j) b[j] = c[j];
  b[6] = static_cast<double>(best_k);
  b[7] = c[6];
  b[8] = c[7];
  b[9] = c[8];
  b[10] = c[9];
  b[11] = c[10];
  b[12] = c[11];
  b[13] = c[12];
  b[14] = c[13];
  b[15] = c[14];
  b[16] = c[15];
  b[17] = static_cast<double>(best_rank);
}

__global__ void ik_lm_multiseed_v4_block_target_kernel(const double* targets,
                                                       const double* seeds,
                                                       double* best, int N, int K,
                                                       int max_iter,
                                                       int limit_gradient_mode,
                                                       int precision_mode,
                                                       int fallback_mode,
                                                       int* fallback_count) {
  const int target_id = blockIdx.x;
  const int seed_id = threadIdx.x;
  if (target_id >= N) return;

  __shared__ double s_cand[16 * kCandidateStride];
  __shared__ int s_fallback;
  if (threadIdx.x == 0) s_fallback = 0;
  __syncthreads();

  if (seed_id < 16) {
    double* c = s_cand + seed_id * kCandidateStride;
    if (seed_id < K) {
      solve_candidate_v4(targets + target_id * 16,
                         seeds + (target_id * K + seed_id) * 6,
                         max_iter, limit_gradient_mode, precision_mode, c);
    } else {
      for (int j = 0; j < kCandidateStride; ++j) c[j] = 0.0;
      c[8] = 1e300;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int best_k = 0;
    int best_rank = 4;
    int best_near = 2;
    double best_pose = 1e300;
    for (int k = 0; k < K && k < 16; ++k) {
      const double* c = s_cand + k * kCandidateStride;
      if (candidate_better_v4(c, k, best_rank, best_near, best_pose, best_k)) {
        best_rank = candidate_rank_v4(c);
        best_near = c[15] > 0.5 ? 1 : 0;
        best_pose = c[8];
        best_k = k;
      }
    }
    write_best_from_candidate_v4(s_cand + best_k * kCandidateStride, best_k,
                                 best + target_id * kBestStride);
    s_fallback = (fallback_mode == 1 && best_rank != 0 && precision_mode != 0) ? 1 : 0;
  }
  __syncthreads();

  if (s_fallback) {
    if (seed_id < 16) {
      double* c = s_cand + seed_id * kCandidateStride;
      if (seed_id < K) {
        solve_candidate_v4(targets + target_id * 16,
                           seeds + (target_id * K + seed_id) * 6,
                           max_iter, limit_gradient_mode, 0, c);
      } else {
        for (int j = 0; j < kCandidateStride; ++j) c[j] = 0.0;
        c[8] = 1e300;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int best_k = 0;
      int best_rank = 4;
      int best_near = 2;
      double best_pose = 1e300;
      for (int k = 0; k < K && k < 16; ++k) {
        const double* c = s_cand + k * kCandidateStride;
        if (candidate_better_v4(c, k, best_rank, best_near, best_pose, best_k)) {
          best_rank = candidate_rank_v4(c);
          best_near = c[15] > 0.5 ? 1 : 0;
          best_pose = c[8];
          best_k = k;
        }
      }
      write_best_from_candidate_v4(s_cand + best_k * kCandidateStride, best_k,
                                   best + target_id * kBestStride);
      if (fallback_count) atomicAdd(fallback_count, 1);
    }
  }
}

__global__ void ik_lm_multiseed_v4_warp_target_kernel(const double* targets,
                                                      const double* seeds,
                                                      double* best, int N, int K,
                                                      int max_iter,
                                                      int limit_gradient_mode,
                                                      int precision_mode) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int target_id = blockIdx.x * warps_per_block + warp_in_block;
  if (target_id >= N) return;

  extern __shared__ double s_all[];
  double* s_warp = s_all + warp_in_block * 16 * kCandidateStride;
  if (lane < 16) {
    double* c = s_warp + lane * kCandidateStride;
    if (lane < K) {
      solve_candidate_v4(targets + target_id * 16,
                         seeds + (target_id * K + lane) * 6,
                         max_iter, limit_gradient_mode, precision_mode, c);
    } else {
      for (int j = 0; j < kCandidateStride; ++j) c[j] = 0.0;
      c[8] = 1e300;
    }
  }
  __syncthreads();

  if (lane == 0) {
    int best_k = 0;
    int best_rank = 4;
    int best_near = 2;
    double best_pose = 1e300;
    for (int k = 0; k < K && k < 16; ++k) {
      const double* c = s_warp + k * kCandidateStride;
      if (candidate_better_v4(c, k, best_rank, best_near, best_pose, best_k)) {
        best_rank = candidate_rank_v4(c);
        best_near = c[15] > 0.5 ? 1 : 0;
        best_pose = c[8];
        best_k = k;
      }
    }
    write_best_from_candidate_v4(s_warp + best_k * kCandidateStride, best_k,
                                 best + target_id * kBestStride);
  }
}

__global__ void fk_frames_check_kernel(const double* q, double* out, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double T[16], p[18], z[18];
  fk_with_frames_v4(q + i * 6, T, p, z);
  double* o = out + i * 52;
  for (int j = 0; j < 16; ++j) o[j] = T[j];
  for (int j = 0; j < 18; ++j) o[16 + j] = p[j];
  for (int j = 0; j < 18; ++j) o[34 + j] = z[j];
}

__global__ void jacobian_check_kernel(const double* q, double* out, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double T[16], p[18], z[18], J[36];
  fk_with_frames_v4(q + i * 6, T, p, z);
  analytical_jacobian_v4(T, p, z, J);
  for (int j = 0; j < 36; ++j) out[i * 36 + j] = J[j];
}

}  // namespace cuda
}  // namespace rtfg

int main(int argc, char** argv) {
  const char* mode = "v4_static";
  const char* variant = "baseline";
  const char* limit_gradient = "finite_diff";
  const char* graph_mode = "off";
  const char* precision_mode_name = "fp64";
  const char* fallback_mode_name = "none";
  const char* targets_path = nullptr;
  const char* seeds_path = nullptr;
  const char* best_csv = nullptr;
  const char* candidates_csv = nullptr;
  const char* summary_csv = nullptr;
  const char* timing_csv = nullptr;
  int N = 0, K = 16, repeat = 1, warmup = 0, max_iter = 60;
  double limit_weight = 0.03;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = argv[++i];
    else if (std::strcmp(argv[i], "--limit-gradient") == 0 && i + 1 < argc) limit_gradient = argv[++i];
    else if (std::strcmp(argv[i], "--graph-mode") == 0 && i + 1 < argc) graph_mode = argv[++i];
    else if (std::strcmp(argv[i], "--precision-mode") == 0 && i + 1 < argc) precision_mode_name = argv[++i];
    else if (std::strcmp(argv[i], "--fallback-mode") == 0 && i + 1 < argc) fallback_mode_name = argv[++i];
    else if (std::strcmp(argv[i], "--targets") == 0 && i + 1 < argc) targets_path = argv[++i];
    else if (std::strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) seeds_path = argv[++i];
    else if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--K") == 0 && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) max_iter = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--limit-weight") == 0 && i + 1 < argc) limit_weight = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--best-csv") == 0 && i + 1 < argc) best_csv = argv[++i];
    else if (std::strcmp(argv[i], "--candidates-csv") == 0 && i + 1 < argc) candidates_csv = argv[++i];
    else if (std::strcmp(argv[i], "--summary-csv") == 0 && i + 1 < argc) summary_csv = argv[++i];
    else if (std::strcmp(argv[i], "--timing-csv") == 0 && i + 1 < argc) timing_csv = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: %s --mode v4_static --variant baseline|opt4c_block_target --limit-gradient finite_diff|analytic --graph-mode off|capture_replay|persistent_replay --precision-mode fp64|mixed_safe|mixed_mid|mixed_aggressive|fp32_risky --fallback-mode none|strict_fail_to_fp64 --targets <raw> --seeds <raw> --N <N> --K 16 [--repeat 30 --warmup 10] [--limit-weight 0.03]\n", argv[0]);
      return 0;
    }
  }

  if (upload_constants(limit_weight) != cudaSuccess) {
    std::fprintf(stderr, "ERROR: failed to upload constants\n");
    return 1;
  }

  std::vector<double> targets, seeds;
  if (targets_path && !read_raw_doubles(targets_path, &targets)) return 1;
  if (seeds_path && !read_raw_doubles(seeds_path, &seeds)) return 1;
  if (N <= 0 && !targets.empty()) N = static_cast<int>(targets.size() / 16);

  if (std::strcmp(mode, "fk_check") == 0 || std::strcmp(mode, "jacobian_check") == 0) {
    if (seeds.empty()) {
      std::fprintf(stderr, "ERROR: --seeds raw q array required for %s\n", mode);
      return 1;
    }
    N = static_cast<int>(seeds.size() / 6);
    double *d_q = nullptr, *d_out = nullptr;
    const int out_cols = std::strcmp(mode, "fk_check") == 0 ? 52 : 36;
    std::vector<double> out(static_cast<size_t>(N) * out_cols);
    cudaMalloc(&d_q, seeds.size() * sizeof(double));
    cudaMalloc(&d_out, out.size() * sizeof(double));
    cudaMemcpy(d_q, seeds.data(), seeds.size() * sizeof(double), cudaMemcpyHostToDevice);
    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    if (std::strcmp(mode, "fk_check") == 0) {
      rtfg::cuda::fk_frames_check_kernel<<<grid, block>>>(d_q, d_out, N);
    } else {
      rtfg::cuda::jacobian_check_kernel<<<grid, block>>>(d_q, d_out, N);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_out);
    if (best_csv) {
      FILE* f = std::fopen(best_csv, "w");
      if (!f) return 1;
      if (std::strcmp(mode, "fk_check") == 0) {
        std::fprintf(f, "sample_id,kind,index,value\n");
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < 16; ++j) std::fprintf(f, "%d,T,%d,%.17g\n", i, j, out[i * out_cols + j]);
          for (int j = 0; j < 18; ++j) std::fprintf(f, "%d,p,%d,%.17g\n", i, j, out[i * out_cols + 16 + j]);
          for (int j = 0; j < 18; ++j) std::fprintf(f, "%d,z,%d,%.17g\n", i, j, out[i * out_cols + 34 + j]);
        }
      } else {
        std::fprintf(f, "sample_id,row,col,value\n");
        for (int i = 0; i < N; ++i)
          for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
              std::fprintf(f, "%d,%d,%d,%.17g\n", i, r, c, out[i * out_cols + r * 6 + c]);
      }
      std::fclose(f);
    }
    std::printf("mode=%s\nN=%d\noutput_csv=%s\n", mode, N, best_csv ? best_csv : "");
    return 0;
  }

  if (targets.empty() || seeds.empty() || N <= 0 || K <= 0) {
    std::fprintf(stderr, "ERROR: v4_static needs --targets --seeds --N --K\n");
    return 1;
  }
  if (std::strcmp(variant, "opt4_warp_per_seed") == 0) {
    std::fprintf(stderr, "ERROR: variant opt4_warp_per_seed is intentionally not implemented; use follow-up postmortem outputs for this failed branch\n");
    return 2;
  }
  const int limit_gradient_mode = std::strcmp(limit_gradient, "analytic") == 0 ? 1 : 0;
  if (static_cast<int>(targets.size()) < N * 16 || static_cast<int>(seeds.size()) < N * K * 6) {
    std::fprintf(stderr, "ERROR: input size mismatch targets=%zu seeds=%zu N=%d K=%d\n",
                 targets.size(), seeds.size(), N, K);
    return 1;
  }

  int precision_mode = 0;
  if (std::strcmp(precision_mode_name, "fp64") == 0) precision_mode = 0;
  else if (std::strcmp(precision_mode_name, "mixed_safe") == 0) precision_mode = 1;
  else if (std::strcmp(precision_mode_name, "mixed_mid") == 0) precision_mode = 2;
  else if (std::strcmp(precision_mode_name, "mixed_aggressive") == 0) precision_mode = 3;
  else if (std::strcmp(precision_mode_name, "fp32_risky") == 0) precision_mode = 4;
  else {
    std::fprintf(stderr, "ERROR: unknown --precision-mode %s\n", precision_mode_name);
    return 1;
  }
  const int fallback_mode = std::strcmp(fallback_mode_name, "strict_fail_to_fp64") == 0 ? 1 : 0;
  const bool use_graph = std::strcmp(graph_mode, "off") != 0;
  const bool use_opt4c = std::strcmp(variant, "opt4c_block_target") == 0;
  const bool use_opt4b = std::strcmp(variant, "opt4b_warp_target") == 0;
  if (use_graph && !use_opt4c) {
    std::fprintf(stderr, "ERROR: graph-mode currently supports --variant opt4c_block_target only\n");
    return 1;
  }

  double *d_targets = nullptr, *d_seeds = nullptr, *d_candidates = nullptr, *d_best = nullptr;
  double *h_targets_pinned = nullptr, *h_seeds_pinned = nullptr, *h_best_pinned = nullptr;
  int* d_fallback_count = nullptr;
  std::vector<double> h_candidates(static_cast<size_t>(N) * K * kCandidateStride);
  std::vector<double> h_best(static_cast<size_t>(N) * kBestStride);
  cudaMalloc(&d_targets, static_cast<size_t>(N) * 16 * sizeof(double));
  cudaMalloc(&d_seeds, static_cast<size_t>(N) * K * 6 * sizeof(double));
  cudaMalloc(&d_candidates, h_candidates.size() * sizeof(double));
  cudaMalloc(&d_best, h_best.size() * sizeof(double));
  cudaMalloc(&d_fallback_count, sizeof(int));
  cudaMallocHost(&h_targets_pinned, static_cast<size_t>(N) * 16 * sizeof(double));
  cudaMallocHost(&h_seeds_pinned, static_cast<size_t>(N) * K * 6 * sizeof(double));
  cudaMallocHost(&h_best_pinned, h_best.size() * sizeof(double));
  std::memcpy(h_targets_pinned, targets.data(), static_cast<size_t>(N) * 16 * sizeof(double));
  std::memcpy(h_seeds_pinned, seeds.data(), static_cast<size_t>(N) * K * 6 * sizeof(double));

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEvent_t h2d0, h2d1, d2h0, d2h1;
  cudaEventCreate(&h2d0);
  cudaEventCreate(&h2d1);
  cudaEventCreate(&d2h0);
  cudaEventCreate(&d2h1);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::vector<double> times;
  std::vector<double> host_prepare_ms;
  std::vector<double> h2d_times;
  std::vector<double> launch_overhead_ms;
  std::vector<double> d2h_times;
  std::vector<double> e2e_times;
  std::vector<int> fallback_counts;

  auto launch_solver = [&](cudaStream_t s) -> cudaError_t {
    cudaMemsetAsync(d_fallback_count, 0, sizeof(int), s);
    if (use_opt4c) {
      rtfg::cuda::ik_lm_multiseed_v4_block_target_kernel<<<dim3(N, 1, 1), dim3(32, 1, 1), 0, s>>>(
          d_targets, d_seeds, d_best, N, K, max_iter, limit_gradient_mode,
          precision_mode, fallback_mode, d_fallback_count);
    } else if (use_opt4b) {
      constexpr int kBlockThreads = 128;
      constexpr int kWarpsPerBlock = kBlockThreads / 32;
      const int blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
      const size_t shmem = static_cast<size_t>(kWarpsPerBlock) * 16 * kCandidateStride * sizeof(double);
      rtfg::cuda::ik_lm_multiseed_v4_warp_target_kernel<<<dim3(blocks, 1, 1), dim3(kBlockThreads, 1, 1), shmem, s>>>(
          d_targets, d_seeds, d_best, N, K, max_iter, limit_gradient_mode, precision_mode);
    } else {
      rtfg::cuda::ik_lm_multiseed_v4_kernel<<<dim3(N, K, 1), dim3(1, 1, 1), 0, s>>>(d_targets, d_seeds, d_candidates, N, K, max_iter, limit_gradient_mode);
      rtfg::cuda::select_best_per_target_v4_kernel<<<dim3(N, 1, 1), dim3(1, 1, 1), 0, s>>>(d_candidates, d_best, N, K);
    }
    return cudaGetLastError();
  };

  auto copy_inputs = [&]() -> float {
    cudaEventRecord(h2d0, stream);
    cudaMemcpyAsync(d_targets, h_targets_pinned, static_cast<size_t>(N) * 16 * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seeds, h_seeds_pinned, static_cast<size_t>(N) * K * 6 * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaEventRecord(h2d1, stream);
    cudaEventSynchronize(h2d1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, h2d0, h2d1);
    return ms;
  };

  auto copy_outputs = [&]() -> float {
    cudaEventRecord(d2h0, stream);
    cudaMemcpyAsync(h_best_pinned, d_best, h_best.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(d2h1, stream);
    cudaEventSynchronize(d2h1);
    std::memcpy(h_best.data(), h_best_pinned, h_best.size() * sizeof(double));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, d2h0, d2h1);
    return ms;
  };

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  if (use_graph) {
    for (int rep = 0; rep < warmup; ++rep) {
      copy_inputs();
      launch_solver(stream);
      cudaStreamSynchronize(stream);
      copy_outputs();
    }
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(d_targets, h_targets_pinned, static_cast<size_t>(N) * 16 * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seeds, h_seeds_pinned, static_cast<size_t>(N) * K * 6 * sizeof(double), cudaMemcpyHostToDevice, stream);
    launch_solver(stream);
    cudaMemcpyAsync(h_best_pinned, d_best, h_best.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  }

  const int total_repeat = use_graph ? repeat : warmup + repeat;
  for (int rep = 0; rep < total_repeat; ++rep) {
    const bool collect = use_graph || rep >= warmup;
    auto e2e_start = std::chrono::high_resolution_clock::now();
    auto host0 = std::chrono::high_resolution_clock::now();
    auto host1 = std::chrono::high_resolution_clock::now();
    const double host_ms = std::chrono::duration<double, std::milli>(host1 - host0).count();
    const float h2d_ms = use_graph ? 0.0f : copy_inputs();

    cudaEventRecord(ev0, stream);
    const auto launch0 = std::chrono::high_resolution_clock::now();
    if (use_graph) {
      cudaGraphLaunch(graph_exec, stream);
    } else {
      launch_solver(stream);
    }
    const auto launch1 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);

    const float d2h_ms = use_graph ? 0.0f : copy_outputs();
    if (use_graph) std::memcpy(h_best.data(), h_best_pinned, h_best.size() * sizeof(double));
    int h_fallback_count = 0;
    cudaMemcpyAsync(&h_fallback_count, d_fallback_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const auto e2e_end = std::chrono::high_resolution_clock::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();
    const double launch_ms = std::chrono::duration<double, std::milli>(launch1 - launch0).count();

    if (collect) {
      host_prepare_ms.push_back(host_ms);
      h2d_times.push_back(h2d_ms);
      launch_overhead_ms.push_back(launch_ms);
      times.push_back(gpu_ms);
      d2h_times.push_back(d2h_ms);
      e2e_times.push_back(e2e_ms);
      fallback_counts.push_back(h_fallback_count);
    }
  }
  if (!use_opt4c) cudaMemcpy(h_candidates.data(), d_candidates, h_candidates.size() * sizeof(double), cudaMemcpyDeviceToHost);

  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph) cudaGraphDestroy(graph);
  cudaFree(d_targets);
  cudaFree(d_seeds);
  cudaFree(d_candidates);
  cudaFree(d_best);
  cudaFree(d_fallback_count);
  cudaFreeHost(h_targets_pinned);
  cudaFreeHost(h_seeds_pinned);
  cudaFreeHost(h_best_pinned);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaEventDestroy(h2d0);
  cudaEventDestroy(h2d1);
  cudaEventDestroy(d2h0);
  cudaEventDestroy(d2h1);
  cudaStreamDestroy(stream);

  if (best_csv && !write_best_csv(best_csv, h_best, N)) return 1;
  if (candidates_csv && !write_candidates_csv(candidates_csv, h_candidates, N, K)) return 1;
  std::string method = std::string("CUDA-V4-") + variant + "-" + limit_gradient +
                       "-graph_" + graph_mode + "-precision_" + precision_mode_name +
                       "-fallback_" + fallback_mode_name;
  if (summary_csv && !write_summary_csv(summary_csv, method.c_str(), N, K, warmup, repeat, times, h_best)) return 1;
  if (timing_csv && !write_timing_csv(timing_csv, host_prepare_ms, h2d_times, launch_overhead_ms,
                                      times, d2h_times, e2e_times, fallback_counts)) return 1;

  double mean = 0.0;
  for (double x : times) mean += x;
  mean /= std::max<size_t>(1, times.size());
  std::printf("mode=%s\nvariant=%s\nlimit_gradient=%s\ngraph_mode=%s\nprecision_mode=%s\nfallback_mode=%s\nlimit_weight=%.9g\nN=%d\nK=%d\nrepeat=%d\nwarmup=%d\ngpu_stream_ms_mean=%.9g\nthroughput_targets_per_s=%.9g\nbest_csv=%s\nsummary_csv=%s\ntiming_csv=%s\n",
              mode, variant, limit_gradient, graph_mode, precision_mode_name, fallback_mode_name,
              limit_weight, N, K, repeat, warmup, mean, mean > 0.0 ? 1000.0 * N / mean : 0.0,
              best_csv ? best_csv : "", summary_csv ? summary_csv : "", timing_csv ? timing_csv : "");
  return 0;
}
