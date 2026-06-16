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
#include "cuda_ik_6dof.cu"
#include "standard_robot_cuda_ik/generated/ur10_model_constants.h"

using namespace rtfg::cuda;

namespace {

// ---------------------------------------------------------------------------
// A0 ablation: store constant data in global device memory for the A0 kernel,
// which does not use __constant__ memory at all.
// ---------------------------------------------------------------------------
#if ABLATION_LEVEL == 0
struct A0DeviceData {
  double *origins    = nullptr;  // size 96  (6 seg × 16)
  double *axes       = nullptr;  // size 18  (6 seg × 3)
  int    *q_index    = nullptr;  // size 6
  double *T_tcp      = nullptr;  // size 16
  double *joint_limits = nullptr; // size 12 (6 joints × 2)
  double *weights    = nullptr;  // size 24 (4 levels × 6)
};

cudaError_t upload_a0_globals(A0DeviceData& a0) {
  auto upload = [](auto*& d_ptr, const auto* h_data, size_t bytes) -> cudaError_t {
    cudaError_t err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) return err;
    return cudaMemcpy(d_ptr, h_data, bytes, cudaMemcpyHostToDevice);
  };
  CUDA_CHECK(upload(a0.origins,    k_origins,       sizeof(k_origins)));
  CUDA_CHECK(upload(a0.axes,       k_axes,          sizeof(k_axes)));
  CUDA_CHECK(upload(a0.q_index,    k_q_index,       sizeof(k_q_index)));
  CUDA_CHECK(upload(a0.T_tcp,      k_T_wrist3_to_tcp, sizeof(k_T_wrist3_to_tcp)));
  CUDA_CHECK(upload(a0.joint_limits, k_joint_limits, sizeof(k_joint_limits)));
  CUDA_CHECK(upload(a0.weights,    k_weights,       sizeof(k_weights)));
  return cudaSuccess;
}

void free_a0_globals(A0DeviceData& a0) {
  if (a0.origins)       cudaFree(a0.origins);
  if (a0.axes)          cudaFree(a0.axes);
  if (a0.q_index)       cudaFree(a0.q_index);
  if (a0.T_tcp)         cudaFree(a0.T_tcp);
  if (a0.joint_limits)  cudaFree(a0.joint_limits);
  if (a0.weights)       cudaFree(a0.weights);
}
#endif  // ABLATION_LEVEL == 0

#if ABLATION_LEVEL == 0
// File-level A0 device data (accessible from run_batch_solver and main)
static A0DeviceData g_a0;
#endif

struct RunSummary {
  std::vector<double> kernel_ms;
  std::vector<double> gpu_ms;
  std::vector<double> host_ms;
  std::vector<double> d2h_ms;        // per-repeat D2H transfer time
  double h2d_ms = 0.0;             // H2D transfer time (measured once before repeat loop)
  int converged = 0;               // medium-threshold convergence count
  int converged_loose = 0;         // loose-threshold (30mm/10deg)
  int converged_strict = 0;        // strict-threshold (5mm/1deg)
  double avg_pos_err = 0.0;
  double avg_rot_err = 0.0;
  double avg_iters = 0.0;
  // Per-target error arrays (populated on first repeat for CSV output)
  std::vector<double> per_target_pos_err;
  std::vector<double> per_target_rot_err;
  std::vector<double> per_target_iters;
  std::vector<int>    per_target_converged;
};

bool read_binary_file(const char* path, std::vector<double>* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "ERROR: failed to open %s\n", path);
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes % static_cast<long>(sizeof(double)) != 0) {
    std::fprintf(stderr, "ERROR: invalid file size %ld for %s\n", bytes, path);
    std::fclose(f);
    return false;
  }
  out->resize(bytes / static_cast<long>(sizeof(double)));
  if (std::fread(out->data(), sizeof(double), out->size(), f) != out->size()) {
    std::fprintf(stderr, "ERROR: short read for %s\n", path);
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

cudaError_t upload_constants() {
  CUDA_CHECK(cudaMemcpyToSymbol(c_segment_origins, k_origins, sizeof(k_origins)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_segment_axes, k_axes, sizeof(k_axes)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_q_index, k_q_index, sizeof(k_q_index)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_T_wrist3_to_tcp, k_T_wrist3_to_tcp, sizeof(k_T_wrist3_to_tcp)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_joint_limits, k_joint_limits, sizeof(k_joint_limits)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_weight_schedule, k_weights, sizeof(k_weights)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_lambda_params, k_lambda_params, sizeof(k_lambda_params)));
  return cudaSuccess;
}

void cpu_rotation_matrix(double ax, double ay, double az, double angle, double* R) {
  double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
  R[0] = t * ax * ax + c;      R[1] = t * ax * ay + s * az; R[2] = t * ax * az - s * ay; R[3] = 0.0;
  R[4] = t * ax * ay - s * az; R[5] = t * ay * ay + c;      R[6] = t * ay * az + s * ax; R[7] = 0.0;
  R[8] = t * ax * az + s * ay; R[9] = t * ay * az - s * ax; R[10] = t * az * az + c;     R[11] = 0.0;
  R[12] = 0.0;                 R[13] = 0.0;                 R[14] = 0.0;                  R[15] = 1.0;
}

void cpu_mat44_mul(const double* A, const double* B, double* C) {
  for (int r = 0; r < 4; ++r) {
    double a0 = A[r * 4 + 0], a1 = A[r * 4 + 1], a2 = A[r * 4 + 2], a3 = A[r * 4 + 3];
    C[r * 4 + 0] = a0 * B[0] + a1 * B[4] + a2 * B[8] + a3 * B[12];
    C[r * 4 + 1] = a0 * B[1] + a1 * B[5] + a2 * B[9] + a3 * B[13];
    C[r * 4 + 2] = a0 * B[2] + a1 * B[6] + a2 * B[10] + a3 * B[14];
    C[r * 4 + 3] = a0 * B[3] + a1 * B[7] + a2 * B[11] + a3 * B[15];
  }
}

void cpu_forward_kinematics(const double* q, double* T_tip) {
  for (int i = 0; i < 16; ++i) T_tip[i] = (i % 5 == 0) ? 1.0 : 0.0;
  double T_tmp[16], R[16];
  for (int seg = 0; seg < 6; ++seg) {
    cpu_mat44_mul(T_tip, &k_origins[seg * 16], T_tmp);
    std::memcpy(T_tip, T_tmp, 16 * sizeof(double));
    cpu_rotation_matrix(k_axes[seg * 3 + 0], k_axes[seg * 3 + 1], k_axes[seg * 3 + 2], q[k_q_index[seg]], R);
    cpu_mat44_mul(T_tip, R, T_tmp);
    std::memcpy(T_tip, T_tmp, 16 * sizeof(double));
  }
  cpu_mat44_mul(T_tip, k_T_wrist3_to_tcp, T_tmp);
  std::memcpy(T_tip, T_tmp, 16 * sizeof(double));
}

__global__ void test_fk_kernel(const double* d_q, double* d_T, int count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  forward_kinematics(&d_q[tid * 6], &d_T[tid * 16]);
}

int run_fk_verification() {
  constexpr int N = 20;
  std::vector<double> h_q(N * 6), h_gpu(N * 16);
  std::vector<double> h_cpu(N * 16);
  unsigned seed = 42;
  for (int i = 0; i < N * 6; ++i) {
    seed = seed * 1103515245 + 12345;
    double u = static_cast<double>(seed & 0x7fffffff) / 2147483648.0;
    h_q[i] = -M_PI + 2.0 * M_PI * u;
  }
  double *d_q = nullptr, *d_T = nullptr;
  cudaMalloc(&d_q, h_q.size() * sizeof(double));
  cudaMalloc(&d_T, h_gpu.size() * sizeof(double));
  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(double), cudaMemcpyHostToDevice);
  test_fk_kernel<<<1, 32>>>(d_q, d_T, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu.data(), d_T, h_gpu.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_T);

  double max_abs = 0.0;
  for (int i = 0; i < N; ++i) {
    cpu_forward_kinematics(&h_q[i * 6], &h_cpu[i * 16]);
    for (int j = 0; j < 16; ++j) {
      max_abs = std::max(max_abs, std::abs(h_cpu[i * 16 + j] - h_gpu[i * 16 + j]));
    }
  }
  std::printf("fk_max_abs_error=%0.12e\n", max_abs);
  return 0;
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return values[lo];
  double t = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

// ============================================================================
// run_batch_solver — timed IK batch solver loop with optional CUDA Graph replay
// ============================================================================
RunSummary run_batch_solver(
    const std::vector<double>& targets,
    const std::vector<double>& seeds,
    int repeat_count,
    int max_iter,
    int weight_level,
    double pos_tol,
    double rot_tol,
    const char* error_log_path = nullptr) {
  int N = static_cast<int>(targets.size() / 16);
  RunSummary summary;

  // Multi-threshold tolerance values (revision instruction §2.4)
  constexpr double kPosLoose  = 0.030;       // 30 mm
  constexpr double kRotLoose  = 0.1745329252; // 10 deg
  constexpr double kPosStrict = 0.005;       //  5 mm
  constexpr double kRotStrict = 0.0174532925; //  1 deg

  double *d_targets = nullptr, *d_seeds = nullptr, *d_results = nullptr, *d_errors = nullptr, *d_iters = nullptr;
  cudaMalloc(&d_targets, targets.size() * sizeof(double));
  cudaMalloc(&d_seeds, seeds.size() * sizeof(double));
  cudaMalloc(&d_results, N * 6 * sizeof(double));
  cudaMalloc(&d_errors, N * 2 * sizeof(double));
  cudaMalloc(&d_iters, N * sizeof(double));

  std::vector<double> h_results(N * 6), h_errors(N * 2), h_iters(N);
  dim3 grid(N, 1, 1);
  dim3 block(128, 1, 1);

  // === H2D timing (measured once, shared across all repeats) ===
  auto h2d_begin = std::chrono::steady_clock::now();
  cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seeds, seeds.data(), seeds.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto h2d_end = std::chrono::steady_clock::now();
  summary.h2d_ms = std::chrono::duration<double, std::milli>(h2d_end - h2d_begin).count();

  // Persistent CUDA events (created once, used across iterations)
  cudaEvent_t gpu_begin, kernel_begin, kernel_end, gpu_end;
  cudaEventCreate(&gpu_begin);
  cudaEventCreate(&kernel_begin);
  cudaEventCreate(&kernel_end);
  cudaEventCreate(&gpu_end);

#ifdef USE_CUDA_GRAPH
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool graph_captured = false;
  cudaStream_t graph_stream = nullptr;
  cudaStreamCreate(&graph_stream);
#endif

  for (int rep = 0; rep < repeat_count; ++rep) {

#ifdef USE_CUDA_GRAPH
    // === CUDA Graph path ===
    auto host_begin = std::chrono::steady_clock::now();

    if (!graph_captured) {
      // Iteration 0: capture the kernel into a graph
      cudaError_t err = cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeThreadLocal);
      if (err != cudaSuccess) { std::fprintf(stderr, "BeginCapture failed: %s\n", cudaGetErrorString(err)); std::exit(1); }
    }

    // Kernel launch (captured on rep=0, replayed on rep=1+)
    rtfg::cuda::ik_batch_solve_mixed<<<grid, block, 0, graph_stream>>>(
        d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
        max_iter, pos_tol, rot_tol, weight_level, N);

    if (!graph_captured) {
      cudaError_t err = cudaStreamEndCapture(graph_stream, &graph);
      if (err != cudaSuccess) { std::fprintf(stderr, "EndCapture failed: %s\n", cudaGetErrorString(err)); std::abort(); }
      cudaGraphNode_t err_node = nullptr;
      char log_buf[4096] = {};
      err = cudaGraphInstantiate(&graph_exec, graph, &err_node, log_buf, sizeof(log_buf));
      if (err != cudaSuccess) { std::fprintf(stderr, "Instantiate failed: %s\n", cudaGetErrorString(err)); std::exit(1); }
      graph_captured = true;
    }

    // Events placed ONLY around the graph launch (on graph_stream), measuring pure graph execution time.
    // This matches the proven test pattern where events + graph launch on the same stream give accurate timing.
    cudaEventRecord(kernel_begin, graph_stream);
    cudaEventRecord(gpu_begin, graph_stream);
    cudaGraphLaunch(graph_exec, graph_stream);
    cudaEventRecord(kernel_end, graph_stream);
    cudaEventRecord(gpu_end, graph_stream);

    cudaStreamSynchronize(graph_stream);

    // D2H timing (host chrono, measured per-repeat)
    auto d2h_begin = std::chrono::steady_clock::now();
    cudaMemcpy(h_results.data(), d_results, h_results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_errors.data(), d_errors, h_errors.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iters.data(), d_iters, h_iters.size() * sizeof(double), cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::steady_clock::now();
    summary.d2h_ms.push_back(std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count());

    auto host_end = std::chrono::steady_clock::now();

    float kernel_time = 0.0f, gpu_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, kernel_begin, kernel_end);
    cudaEventElapsedTime(&gpu_time, gpu_begin, gpu_end);
    double host_time = std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    summary.kernel_ms.push_back(static_cast<double>(kernel_time));
    summary.gpu_ms.push_back(static_cast<double>(gpu_time));
    summary.host_ms.push_back(host_time);

    // Convergence data only on first iteration (Graph path)
    if (rep == 0) {
      summary.per_target_pos_err.resize(N);
      summary.per_target_rot_err.resize(N);
      summary.per_target_iters.resize(N);
      summary.per_target_converged.resize(N);
      double sum_pos = 0.0, sum_rot = 0.0, sum_iters = 0.0;
      for (int i = 0; i < N; ++i) {
        double pos_err = h_errors[i * 2 + 0];
        double rot_err = h_errors[i * 2 + 1];
        double iters = h_iters[i];
        sum_pos += pos_err; sum_rot += rot_err; sum_iters += iters;
        summary.per_target_pos_err[i] = pos_err;
        summary.per_target_rot_err[i] = rot_err;
        summary.per_target_iters[i] = iters;
        // Medium threshold (primary)
        if (pos_err < pos_tol && rot_err < rot_tol) {
          summary.converged += 1;
          summary.per_target_converged[i] = 1;
        }
        // Loose threshold
        if (pos_err < kPosLoose && rot_err < kRotLoose) summary.converged_loose += 1;
        // Strict threshold
        if (pos_err < kPosStrict && rot_err < kRotStrict) summary.converged_strict += 1;
      }
      summary.avg_pos_err = sum_pos / static_cast<double>(N);
      summary.avg_rot_err = sum_rot / static_cast<double>(N);
      summary.avg_iters = sum_iters / static_cast<double>(N);
    }

#else
    // === Non-graph path (A0-A7): event-based timing ===
    auto host_begin = std::chrono::steady_clock::now();

    cudaEventRecord(kernel_begin);
    cudaEventRecord(gpu_begin);

#if ABLATION_LEVEL >= 7
    rtfg::cuda::ik_batch_solve_mixed<<<grid, block>>>(
        d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
        max_iter, pos_tol, rot_tol, weight_level, N);
#elif ABLATION_LEVEL == 0
    rtfg::cuda::ik_batch_solve_ablation_A0<<<grid, block>>>(
        d_targets, d_seeds, d_results, d_errors, d_iters,
        g_a0.origins, g_a0.axes, g_a0.q_index, g_a0.T_tcp,
        g_a0.joint_limits, g_a0.weights,
        max_iter, pos_tol, rot_tol, 0.05, weight_level, N);
#else
    rtfg::cuda::ik_batch_solve<<<grid, block>>>(
        d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
        max_iter, pos_tol, rot_tol, weight_level, N);
#endif

    cudaEventRecord(kernel_end);
    // D2H timing (host chrono, measured per-repeat)
    auto d2h_begin = std::chrono::steady_clock::now();
    cudaMemcpy(h_results.data(), d_results, h_results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_errors.data(), d_errors, h_errors.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iters.data(), d_iters, h_iters.size() * sizeof(double), cudaMemcpyDeviceToHost);
    auto d2h_end = std::chrono::steady_clock::now();
    summary.d2h_ms.push_back(std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count());
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    auto host_end = std::chrono::steady_clock::now();

    float kernel_time = 0.0f, gpu_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, kernel_begin, kernel_end);
    cudaEventElapsedTime(&gpu_time, gpu_begin, gpu_end);
    double host_time = std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    summary.kernel_ms.push_back(static_cast<double>(kernel_time));
    summary.gpu_ms.push_back(static_cast<double>(gpu_time));
    summary.host_ms.push_back(host_time);

    // Convergence data only on first iteration (non-Graph path)
    if (rep == 0) {
      summary.per_target_pos_err.resize(N);
      summary.per_target_rot_err.resize(N);
      summary.per_target_iters.resize(N);
      summary.per_target_converged.resize(N);
      double sum_pos = 0.0, sum_rot = 0.0, sum_iters = 0.0;
      for (int i = 0; i < N; ++i) {
        double pos_err = h_errors[i * 2 + 0];
        double rot_err = h_errors[i * 2 + 1];
        double iters = h_iters[i];
        sum_pos += pos_err; sum_rot += rot_err; sum_iters += iters;
        summary.per_target_pos_err[i] = pos_err;
        summary.per_target_rot_err[i] = rot_err;
        summary.per_target_iters[i] = iters;
        // Medium threshold (primary)
        if (pos_err < pos_tol && rot_err < rot_tol) {
          summary.converged += 1;
          summary.per_target_converged[i] = 1;
        }
        // Loose threshold
        if (pos_err < kPosLoose && rot_err < kRotLoose) summary.converged_loose += 1;
        // Strict threshold
        if (pos_err < kPosStrict && rot_err < kRotStrict) summary.converged_strict += 1;
      }
      summary.avg_pos_err = sum_pos / static_cast<double>(N);
      summary.avg_rot_err = sum_rot / static_cast<double>(N);
      summary.avg_iters = sum_iters / static_cast<double>(N);
    }
#endif

  }

  // === Write per-target error CSV (if requested) ===
  if (error_log_path && !summary.per_target_pos_err.empty()) {
    FILE* f = std::fopen(error_log_path, "w");
    if (f) {
      std::fprintf(f, "target_id,pos_error_m,rot_error_rad,num_iterations,converged\n");
      for (int i = 0; i < N; ++i) {
        std::fprintf(f, "%d,%.12e,%.12e,%.6f,%d\n",
                     i,
                     summary.per_target_pos_err[i],
                     summary.per_target_rot_err[i],
                     summary.per_target_iters[i],
                     summary.per_target_converged[i]);
      }
      std::fclose(f);
    }
  }

  cudaFree(d_targets);
  cudaFree(d_seeds);
  cudaFree(d_results);
  cudaFree(d_errors);
  cudaFree(d_iters);

#ifdef USE_CUDA_GRAPH
  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph)      cudaGraphDestroy(graph);
  if (graph_stream) cudaStreamDestroy(graph_stream);
#endif

  cudaEventDestroy(gpu_begin);
  cudaEventDestroy(kernel_begin);
  cudaEventDestroy(kernel_end);
  cudaEventDestroy(gpu_end);

  return summary;
}

void print_summary(const RunSummary& summary, int N) {
  auto stats = [](const std::vector<double>& v, double& mean, double& stdv, double& p50, double& p95, double& p99, double& minv, double& maxv) {
    if (v.empty()) { mean=stdv=p50=p95=p99=minv=maxv=0.0; return; }
    mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (double x : v) { double d = x - mean; var += d * d; }
    stdv = std::sqrt(var / static_cast<double>(v.size()));
    p50 = percentile(v, 50.0);
    p95 = percentile(v, 95.0);
    p99 = percentile(v, 99.0);
    minv = *std::min_element(v.begin(), v.end());
    maxv = *std::max_element(v.begin(), v.end());
  };

  double ak, sk, p50k, p95k, p99k, mink, maxk;
  double ag, sg, p50g, p95g, p99g, ming, maxg;
  double ah, sh, p50h, p95h, p99h, minh, maxh;
  double ad2h, sd2h, p50d2h, p95d2h, p99d2h, mind2h, maxd2h;
  stats(summary.kernel_ms, ak, sk, p50k, p95k, p99k, mink, maxk);
  stats(summary.gpu_ms,    ag, sg, p50g, p95g, p99g, ming, maxg);
  stats(summary.host_ms,   ah, sh, p50h, p95h, p99h, minh, maxh);
  stats(summary.d2h_ms,    ad2h, sd2h, p50d2h, p95d2h, p99d2h, mind2h, maxd2h);

  // E2E = H2D + avg_kernel_time + avg_D2H (per revision instruction §2.1 Level 1)
  double e2e_ms = summary.h2d_ms + ak + ad2h;

  std::printf("num_targets=%d\n", N);
  std::printf("repeat_count=%zu\n", summary.kernel_ms.size());

  // Timing: kernel-only (GPU events)
  std::printf("kernel_time_only_ms_mean=%0.6f\n", ak);
  std::printf("kernel_time_only_ms_std=%0.6f\n", sk);
  std::printf("kernel_time_only_ms_min=%0.6f\n", mink);
  std::printf("kernel_time_only_ms_max=%0.6f\n", maxk);
  std::printf("kernel_time_only_ms_p50=%0.6f\n", p50k);
  std::printf("kernel_time_only_ms_p95=%0.6f\n", p95k);
  std::printf("kernel_time_only_ms_p99=%0.6f\n", p99k);

  // Timing: GPU end-to-end (kernel + D2H, GPU events)
  std::printf("gpu_end_to_end_ms_mean=%0.6f\n", ag);
  std::printf("gpu_end_to_end_ms_std=%0.6f\n", sg);
  std::printf("gpu_end_to_end_ms_min=%0.6f\n", ming);
  std::printf("gpu_end_to_end_ms_max=%0.6f\n", maxg);

  // Timing: Host API total (chrono wall clock)
  std::printf("host_api_total_ms_mean=%0.6f\n", ah);
  std::printf("host_api_total_ms_std=%0.6f\n", sh);
  std::printf("host_api_total_ms_min=%0.6f\n", minh);
  std::printf("host_api_total_ms_max=%0.6f\n", maxh);
  std::printf("host_api_total_ms_p50=%0.6f\n", p50h);
  std::printf("host_api_total_ms_p95=%0.6f\n", p95h);
  std::printf("host_api_total_ms_p99=%0.6f\n", p99h);

  // Timing: Data transfer
  std::printf("h2d_time_ms=%0.6f\n", summary.h2d_ms);
  std::printf("d2h_time_ms_mean=%0.6f\n", ad2h);
  std::printf("d2h_time_ms_std=%0.6f\n", sd2h);
  std::printf("e2e_time_ms_mean=%0.6f\n", e2e_ms);

  // Throughput (based on kernel-only GPU time for fair GPU-to-GPU comparison)
  std::printf("throughput_targets_per_s=%0.6f\n", (1000.0 * N) / ak);
  // Valid throughput = raw throughput × success rate (medium)
  double sr_medium = static_cast<double>(summary.converged) / static_cast<double>(N);
  std::printf("valid_throughput_targets_per_s=%0.6f\n", (1000.0 * N) / ak * sr_medium);

  // Convergence: multi-threshold
  std::printf("converged_medium=%d\n", summary.converged);
  std::printf("converged_loose=%d\n", summary.converged_loose);
  std::printf("converged_strict=%d\n", summary.converged_strict);
  std::printf("convergence_rate_medium=%0.6f\n", sr_medium);
  std::printf("convergence_rate_loose=%0.6f\n", static_cast<double>(summary.converged_loose) / static_cast<double>(N));
  std::printf("convergence_rate_strict=%0.6f\n", static_cast<double>(summary.converged_strict) / static_cast<double>(N));

  // Error and iteration averages
  std::printf("avg_pos_error_m=%0.12f\n", summary.avg_pos_err);
  std::printf("avg_rot_error_rad=%0.12f\n", summary.avg_rot_err);
  std::printf("avg_iterations=%0.6f\n", summary.avg_iters);

  // Per-target error percentiles (for error distribution analysis)
  if (!summary.per_target_pos_err.empty()) {
    std::printf("pos_error_p50_m=%0.12f\n", percentile(summary.per_target_pos_err, 50.0));
    std::printf("pos_error_p95_m=%0.12f\n", percentile(summary.per_target_pos_err, 95.0));
    std::printf("pos_error_max_m=%0.12f\n", *std::max_element(summary.per_target_pos_err.begin(), summary.per_target_pos_err.end()));
    std::printf("rot_error_p50_rad=%0.12f\n", percentile(summary.per_target_rot_err, 50.0));
    std::printf("rot_error_p95_rad=%0.12f\n", percentile(summary.per_target_rot_err, 95.0));
    std::printf("rot_error_max_rad=%0.12f\n", *std::max_element(summary.per_target_rot_err.begin(), summary.per_target_rot_err.end()));
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* targets_path = nullptr;
  const char* seeds_path = nullptr;
  const char* error_log_path = nullptr;
  int repeat_count = 30;
  int warmup_count = 3;
  int max_iter = 100;
  int weight_level = 0;
  double pos_tol = 0.03;
  double rot_tol = M_PI / 6.0;
  bool verify_fk = false;

  int ablation_level = ABLATION_LEVEL;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--targets") == 0 && i + 1 < argc) {
      targets_path = argv[++i];
    } else if (std::strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
      seeds_path = argv[++i];
    } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat_count = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup_count = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
      max_iter = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--weight-level") == 0 && i + 1 < argc) {
      weight_level = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--pos-tol") == 0 && i + 1 < argc) {
      pos_tol = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--rot-tol") == 0 && i + 1 < argc) {
      rot_tol = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--error-log") == 0 && i + 1 < argc) {
      error_log_path = argv[++i];
    } else if (std::strcmp(argv[i], "--verify-fk") == 0) {
      verify_fk = true;
    } else if (std::strcmp(argv[i], "--ablation-level") == 0 && i + 1 < argc) {
      ablation_level = std::atoi(argv[++i]);
      if (ablation_level != ABLATION_LEVEL) {
        std::fprintf(stderr, "ERROR: binary compiled for ABLATION_LEVEL=%d but --ablation-level=%d\n",
                     ABLATION_LEVEL, ablation_level);
        return 1;
      }
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: %s --targets <bin> --seeds <bin> [--ablation-level N] [--repeat 30] [--warmup 3] [--error-log <csv>] [--verify-fk]\n", argv[0]);
      return 0;
    }
  }

  if (upload_constants() != cudaSuccess) {
    std::fprintf(stderr, "ERROR: failed to upload constants\n");
    return 1;
  }
  cudaDeviceSynchronize();
#if ABLATION_LEVEL == 0
  // Upload constant data to global device memory for A0 kernel
  if (upload_a0_globals(g_a0) != cudaSuccess) {
    std::fprintf(stderr, "ERROR: failed to upload A0 globals\n");
    return 1;
  }
#endif

  if (verify_fk) {
    return run_fk_verification();
  }
  if (!targets_path || !seeds_path) {
    std::fprintf(stderr, "ERROR: --targets and --seeds are required unless --verify-fk is set\n");
    return 1;
  }
  if (weight_level < 0 || weight_level > 3) {
    std::fprintf(stderr, "ERROR: --weight-level must be in [0, 3]\n");
    return 1;
  }

  std::vector<double> targets;
  std::vector<double> seeds;
  if (!read_binary_file(targets_path, &targets)) return 1;
  if (!read_binary_file(seeds_path, &seeds)) return 1;

  if (targets.size() % 16 != 0) {
    std::fprintf(stderr, "ERROR: targets size must be divisible by 16 doubles\n");
    return 1;
  }
  int N = static_cast<int>(targets.size() / 16);
  if (static_cast<int>(seeds.size()) != N * 6) {
    std::fprintf(stderr, "ERROR: seeds contain %zu doubles, expected %d\n", seeds.size(), N * 6);
    return 1;
  }

  RunSummary summary = run_batch_solver(targets, seeds, warmup_count + repeat_count, max_iter, weight_level, pos_tol, rot_tol, error_log_path);

  // Discard warmup iterations from timing vectors
  if (warmup_count > 0 && static_cast<int>(summary.kernel_ms.size()) >= warmup_count) {
    summary.kernel_ms.erase(summary.kernel_ms.begin(), summary.kernel_ms.begin() + warmup_count);
    summary.gpu_ms.erase(summary.gpu_ms.begin(), summary.gpu_ms.begin() + warmup_count);
    summary.host_ms.erase(summary.host_ms.begin(), summary.host_ms.begin() + warmup_count);
    summary.d2h_ms.erase(summary.d2h_ms.begin(), summary.d2h_ms.begin() + warmup_count);
  }

  print_summary(summary, N);
  std::printf("weight_level=%d\n", weight_level);
  std::printf("ablation_level=%d\n", ablation_level);
  std::printf("warmup_count=%d\n", warmup_count);
  if (error_log_path) std::printf("error_log_path=%s\n", error_log_path);
#if ABLATION_LEVEL == 0
  free_a0_globals(g_a0);
#endif
  return 0;
}
