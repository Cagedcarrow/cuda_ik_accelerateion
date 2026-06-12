// panda_7dof_runner.cu — Standalone host runner for 7DOF Panda IK test
//
// Compilation:
//   mkdir build && cd build
//   cmake .. && make -j
//   ./panda_7dof_test --targets ../panda_test_targets_N10.bin --seeds ../panda_test_seeds_N10.bin
//
// This runner includes the kernel file directly and links only against CUDA runtime.
// It does NOT depend on the main benchmark framework or __constant__ memory.

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans) do { \
  cudaError_t _err = (ans); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                 __FILE__, __LINE__, cudaGetErrorString(_err)); \
    std::exit(1); \
  } \
} while(0)

// Model constants auto-generated from URDF
#include "panda_model_constants.h"

// The kernel (defines launch_ik_batch_solve_7dof and fk_verify_7dof)
#include "panda_7dof_kernel.cu"

// ---------------------------------------------------------------------------
// Helper: read binary file into vector<double>
// ---------------------------------------------------------------------------
static bool read_binary(const char* path, std::vector<double>* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "ERROR: failed to open %s\n", path); return false; }
  std::fseek(f, 0, SEEK_END);
  long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes % (long)sizeof(double) != 0) {
    std::fprintf(stderr, "ERROR: invalid file size %ld for %s\n", bytes, path);
    std::fclose(f); return false;
  }
  out->resize(bytes / (long)sizeof(double));
  if (std::fread(out->data(), sizeof(double), out->size(), f) != out->size()) {
    std::fprintf(stderr, "ERROR: short read for %s\n", path);
    std::fclose(f); return false;
  }
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Helper: upload constants to device (always via cudaMemcpy to device pointers)
// ---------------------------------------------------------------------------
static cudaError_t upload_constants(
    double** d_origins, double** d_axes, int** d_q_index,
    double** d_T_tcp, double** d_joint_limits, double** d_weights) {
  auto upload = [](auto*& d_ptr, const auto* h_data, size_t bytes) -> cudaError_t {
    cudaError_t err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) return err;
    return cudaMemcpy(d_ptr, h_data, bytes, cudaMemcpyHostToDevice);
  };
  CUDA_CHECK(upload(*d_origins,       k_origins,       sizeof(k_origins)));
  CUDA_CHECK(upload(*d_axes,          k_axes,          sizeof(k_axes)));
  CUDA_CHECK(upload(*d_q_index,       k_q_index,       sizeof(k_q_index)));
  CUDA_CHECK(upload(*d_T_tcp,         k_T_tcp,         sizeof(k_T_tcp)));
  CUDA_CHECK(upload(*d_joint_limits,  k_joint_limits,  sizeof(k_joint_limits)));
  CUDA_CHECK(upload(*d_weights,       k_weights,       sizeof(k_weights)));
  return cudaSuccess;
}

static void free_constants(
    double* d_origins, double* d_axes, int* d_q_index,
    double* d_T_tcp, double* d_joint_limits, double* d_weights) {
  if (d_origins)      cudaFree(d_origins);
  if (d_axes)         cudaFree(d_axes);
  if (d_q_index)      cudaFree(d_q_index);
  if (d_T_tcp)        cudaFree(d_T_tcp);
  if (d_joint_limits) cudaFree(d_joint_limits);
  if (d_weights)      cudaFree(d_weights);
}

// ---------------------------------------------------------------------------
// FK verification: run CUDA FK on N random q, compare vs CPU FK
// ---------------------------------------------------------------------------
static int run_fk_verify() {
  constexpr int N = 20;
  std::vector<double> h_q(N * 7);
  unsigned seed = 42;
  for (int i = 0; i < N * 7; ++i) {
    seed = seed * 1103515245 + 12345;
    double u = (double)(seed & 0x7fffffff) / 2147483648.0;
    h_q[i] = -M_PI + 2.0 * M_PI * u;
  }

  double *d_q = nullptr, *d_T = nullptr;
  cudaMalloc(&d_q, h_q.size() * sizeof(double));
  cudaMalloc(&d_T, N * 16 * sizeof(double));
  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(double), cudaMemcpyHostToDevice);

  // Upload FK constants
  double *d_origins = nullptr, *d_axes = nullptr, *d_T_tcp = nullptr;
  int *d_q_index = nullptr;
  cudaMalloc(&d_origins, sizeof(k_origins));
  cudaMalloc(&d_axes,    sizeof(k_axes));
  cudaMalloc(&d_q_index, sizeof(k_q_index));
  cudaMalloc(&d_T_tcp,   sizeof(k_T_tcp));
  cudaMemcpy(d_origins, k_origins, sizeof(k_origins), cudaMemcpyHostToDevice);
  cudaMemcpy(d_axes,    k_axes,    sizeof(k_axes),    cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_index, k_q_index, sizeof(k_q_index), cudaMemcpyHostToDevice);
  cudaMemcpy(d_T_tcp,   k_T_tcp,   sizeof(k_T_tcp),   cudaMemcpyHostToDevice);

  fk_verify_7dof<<<1, 32>>>(d_q, d_T, N, d_origins, d_axes, d_q_index, d_T_tcp);
  cudaDeviceSynchronize();

  std::vector<double> h_gpu(N * 16);
  cudaMemcpy(h_gpu.data(), d_T, N * 16 * sizeof(double), cudaMemcpyDeviceToHost);

  // CPU FK for verification
  auto cpu_fk = [](const double* q, double* T) {
    for (int i = 0; i < 16; ++i) T[i] = (i % 5 == 0) ? 1.0 : 0.0;
    double tmp[16], R[16];
    for (int seg = 0; seg < 7; ++seg) {
      // Multiply by origin
      for (int r = 0; r < 4; ++r) {
        double a0 = T[r*4+0], a1 = T[r*4+1], a2 = T[r*4+2], a3 = T[r*4+3];
        const double* O = &k_origins[seg * 16];
        tmp[r*4+0] = a0*O[0] + a1*O[4] + a2*O[8]  + a3*O[12];
        tmp[r*4+1] = a0*O[1] + a1*O[5] + a2*O[9]  + a3*O[13];
        tmp[r*4+2] = a0*O[2] + a1*O[6] + a2*O[10] + a3*O[14];
        tmp[r*4+3] = a0*O[3] + a1*O[7] + a2*O[11] + a3*O[15];
      }
      std::memcpy(T, tmp, 16 * sizeof(double));
      // Multiply by Rodrigues(axis, q[i])
      double theta = q[k_q_index[seg]];
      double ax = k_axes[seg*3+0], ay = k_axes[seg*3+1], az = k_axes[seg*3+2];
      double c = cos(theta), s = sin(theta), t = 1.0 - c;
      R[0] = t*ax*ax+c; R[1]=t*ax*ay+s*az; R[2]=t*ax*az-s*ay; R[3]=0;
      R[4]=t*ax*ay-s*az; R[5]=t*ay*ay+c; R[6]=t*ay*az+s*ax; R[7]=0;
      R[8]=t*ax*az+s*ay; R[9]=t*ay*az-s*ax; R[10]=t*az*az+c; R[11]=0;
      R[12]=0; R[13]=0; R[14]=0; R[15]=1;
      for (int r = 0; r < 4; ++r) {
        double a0 = T[r*4+0], a1 = T[r*4+1], a2 = T[r*4+2], a3 = T[r*4+3];
        tmp[r*4+0] = a0*R[0] + a1*R[4] + a2*R[8]  + a3*R[12];
        tmp[r*4+1] = a0*R[1] + a1*R[5] + a2*R[9]  + a3*R[13];
        tmp[r*4+2] = a0*R[2] + a1*R[6] + a2*R[10] + a3*R[14];
        tmp[r*4+3] = a0*R[3] + a1*R[7] + a2*R[11] + a3*R[15];
      }
      std::memcpy(T, tmp, 16 * sizeof(double));
    }
    // Apply TCP offset
    for (int r = 0; r < 4; ++r) {
      double a0 = T[r*4+0], a1 = T[r*4+1], a2 = T[r*4+2], a3 = T[r*4+3];
      tmp[r*4+0] = a0*k_T_tcp[0] + a1*k_T_tcp[4] + a2*k_T_tcp[8]  + a3*k_T_tcp[12];
      tmp[r*4+1] = a0*k_T_tcp[1] + a1*k_T_tcp[5] + a2*k_T_tcp[9]  + a3*k_T_tcp[13];
      tmp[r*4+2] = a0*k_T_tcp[2] + a1*k_T_tcp[6] + a2*k_T_tcp[10] + a3*k_T_tcp[14];
      tmp[r*4+3] = a0*k_T_tcp[3] + a1*k_T_tcp[7] + a2*k_T_tcp[11] + a3*k_T_tcp[15];
    }
    std::memcpy(T, tmp, 16 * sizeof(double));
  };

  double max_err = 0.0;
  std::vector<double> h_cpu(N * 16);
  for (int i = 0; i < N; ++i) {
    cpu_fk(&h_q[i * 7], &h_cpu[i * 16]);
    for (int j = 0; j < 16; ++j) {
      double err = std::abs(h_cpu[i * 16 + j] - h_gpu[i * 16 + j]);
      if (err > max_err) max_err = err;
    }
  }

  std::printf("fk_max_abs_error=%0.12e\n", max_err);
  std::printf("fk_verify=%s\n", max_err < 1e-10 ? "PASS" : "CHECK");

  cudaFree(d_q); cudaFree(d_T);
  cudaFree(d_origins); cudaFree(d_axes); cudaFree(d_q_index); cudaFree(d_T_tcp);
  return (max_err < 1e-10) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Main IK solver test
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const char* targets_path = nullptr;
  const char* seeds_path = nullptr;
  int max_iter = 160;
  double pos_tol = 0.03;
  double rot_tol = M_PI / 6.0;
  int weight_level = 0;
  bool verify_fk = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--targets") == 0 && i + 1 < argc)
      targets_path = argv[++i];
    else if (std::strcmp(argv[i], "--seeds") == 0 && i + 1 < argc)
      seeds_path = argv[++i];
    else if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc)
      max_iter = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--pos-tol") == 0 && i + 1 < argc)
      pos_tol = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--rot-tol") == 0 && i + 1 < argc)
      rot_tol = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--weight-level") == 0 && i + 1 < argc)
      weight_level = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--verify-fk") == 0)
      verify_fk = true;
    else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: %s --targets <bin> --seeds <bin> [--max-iter 160] [--verify-fk]\n", argv[0]);
      return 0;
    }
  }

  // Upload model constants to device
  double *d_origins = nullptr, *d_axes = nullptr, *d_T_tcp = nullptr;
  double *d_joint_limits = nullptr, *d_weights = nullptr;
  int *d_q_index = nullptr;

  if (upload_constants(&d_origins, &d_axes, &d_q_index,
                        &d_T_tcp, &d_joint_limits, &d_weights) != cudaSuccess) {
    std::fprintf(stderr, "ERROR: failed to upload constants\n");
    return 1;
  }

  // --verify-fk mode
  if (verify_fk) {
    int ret = run_fk_verify();
    free_constants(d_origins, d_axes, d_q_index, d_T_tcp, d_joint_limits, d_weights);
    return ret;
  }

  // Load test data
  if (!targets_path || !seeds_path) {
    std::fprintf(stderr, "ERROR: --targets and --seeds are required\n");
    std::fprintf(stderr, "  Use --verify-fk to run FK verification only\n");
    return 1;
  }

  std::vector<double> targets, seeds;
  if (!read_binary(targets_path, &targets)) return 1;
  if (!read_binary(seeds_path, &seeds)) return 1;

  int N = (int)(targets.size() / 16);
  if ((int)seeds.size() != N * 7) {
    std::fprintf(stderr, "ERROR: seeds contain %zu doubles, expected %d (N=%d)\n",
                 seeds.size(), N * 7, N);
    return 1;
  }

  std::printf("=== 7DOF Panda IK Test ===\n");
  std::printf("N=%d\n", N);
  std::printf("max_iter=%d\n", max_iter);
  std::printf("pos_tol=%0.4f\n", pos_tol);
  std::printf("rot_tol=%0.4f\n", rot_tol);
  std::printf("weight_level=%d\n", weight_level);

  // Allocate device buffers
  double *d_targets = nullptr, *d_seeds = nullptr;
  double *d_results = nullptr, *d_errors = nullptr, *d_iters = nullptr;
  cudaMalloc(&d_targets, targets.size() * sizeof(double));
  cudaMalloc(&d_seeds, seeds.size() * sizeof(double));
  cudaMalloc(&d_results, N * 7 * sizeof(double));
  cudaMalloc(&d_errors, N * 2 * sizeof(double));
  cudaMalloc(&d_iters, N * sizeof(double));

  cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seeds, seeds.data(), seeds.size() * sizeof(double), cudaMemcpyHostToDevice);

  // Warm-up run
  launch_ik_batch_solve_7dof(d_targets, d_seeds, d_results, d_errors, d_iters,
                              max_iter, pos_tol, rot_tol, weight_level, N,
                              d_origins, d_axes, d_q_index, d_T_tcp,
                              d_joint_limits, d_weights);
  cudaDeviceSynchronize();

  // Timed run
  auto t0 = std::chrono::steady_clock::now();
  launch_ik_batch_solve_7dof(d_targets, d_seeds, d_results, d_errors, d_iters,
                              max_iter, pos_tol, rot_tol, weight_level, N,
                              d_origins, d_axes, d_q_index, d_T_tcp,
                              d_joint_limits, d_weights);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Download results
  std::vector<double> h_results(N * 7), h_errors(N * 2), h_iters(N);
  cudaMemcpy(h_results.data(), d_results, N * 7 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_errors.data(), d_errors, N * 2 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_iters.data(), d_iters, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Print per-target results
  int converged = 0;
  double sum_pos = 0, sum_rot = 0, sum_iters = 0;
  for (int i = 0; i < N; ++i) {
    double pos_err = h_errors[i * 2 + 0];
    double rot_err = h_errors[i * 2 + 1];
    double iters = h_iters[i];
    bool conv = (pos_err < pos_tol && rot_err < rot_tol);
    if (conv) converged++;
    sum_pos  += pos_err;
    sum_rot  += rot_err;
    sum_iters += iters;
    std::printf("  [%d] %s iters=%.0f pos_err=%.6f rot_err=%.6f\n",
                i, conv ? "PASS" : "FAIL", iters, pos_err, rot_err);

    // Print final joint angles
    std::printf("       q=[");
    for (int j = 0; j < 7; ++j)
      std::printf("%s%.6f", j ? ", " : "", h_results[i * 7 + j]);
    std::printf("]\n");
  }

  double conv_rate = (double)converged / (double)N;
  std::printf("\n--- Summary ---\n");
  std::printf("converged=%d/%d (%.1f%%)\n", converged, N, conv_rate * 100.0);
  std::printf("avg_pos_err=%.8f\n", sum_pos / N);
  std::printf("avg_rot_err=%.8f\n", sum_rot / N);
  std::printf("avg_iterations=%.2f\n", sum_iters / N);
  std::printf("kernel_time_ms=%.3f\n", elapsed_ms);
  if (N > 0)
    std::printf("throughput_targets_per_s=%.1f\n", 1000.0 * N / elapsed_ms);
  std::printf("---\n");

  // Cleanup
  cudaFree(d_targets);
  cudaFree(d_seeds);
  cudaFree(d_results);
  cudaFree(d_errors);
  cudaFree(d_iters);
  free_constants(d_origins, d_axes, d_q_index, d_T_tcp, d_joint_limits, d_weights);

  std::printf("Done.\n");
  return 0;
}
