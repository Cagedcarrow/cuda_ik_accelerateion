// panda_7dof_kernel.cu — 7DOF CUDA batch IK kernel for Franka Panda
//
// This is a MINIMAL standalone kernel derived from the UR10 6DOF kernel,
// with all hardcoded dimensions changed from 6 to 7.
//
// Key differences from 6DOF:
//   - 7 active joints → 7×7 Hessian, 6×7 Jacobian, 7-dim step vector
//   - 7 segments in FK loop
//   - 7×7 LDL^T solver (replaces 6×6)
//   - 112 origins + 21 axes + 14 joint limits + 28 weights
//
// Architecture: Ada Lovelace (sm_89)
// Mapping: 1 block/target, 128 threads/block (4 warps)
// Ablation: A5 (adaptive damping, no step clamp, no branch align)
//
// Reuse from main codebase:
//   - Rodrigues formula (build_rotation_matrix)
//   - 4×4 matrix multiply (mat44_mul)
//   - Pose error (pose_error, rotation_geodesic_distance)
//   - PaddedMat6x8 (for J matrix — still 6 rows)
//   - 7-vector norm (cuda_norm7 — new)

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#ifndef CUDA_PI
#define CUDA_PI 3.14159265358979323846
#endif

// ============================================================================
// Device math helpers (copied from cuda_utilities.cuh)
// ============================================================================
__device__ __forceinline__ double cuda_clamp(double x, double lo, double hi) {
  return fmin(fmax(x, lo), hi);
}

__device__ __forceinline__ double cuda_norm6(const double* v) {
  return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] +
              v[3]*v[3] + v[4]*v[4] + v[5]*v[5]);
}

__device__ __forceinline__ double cuda_norm7(const double* v) {
  return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] +
              v[3]*v[3] + v[4]*v[4] + v[5]*v[5] + v[6]*v[6]);
}

// ============================================================================
// PaddedMat6x8 — type-safe 2D wrapper for Jacobian (6×PADDEDMAT_STRIDE)
// ============================================================================
#define PADDEDMAT_STRIDE 8  // Use padded layout (matching A2+)

struct PaddedMat6x8 {
  double* data;
  __device__ __forceinline__ PaddedMat6x8(double* d) : data(d) {}
  __device__ __forceinline__ double& operator()(int row, int col) {
    return data[row * PADDEDMAT_STRIDE + col];
  }
};

// ============================================================================
// PaddedMat7x8 — wrapper for Hessian (7×PADDEDMAT_STRIDE)
// ============================================================================
struct PaddedMat7x8 {
  double* data;
  __device__ __forceinline__ PaddedMat7x8(double* d) : data(d) {}
  __device__ __forceinline__ double& operator()(int row, int col) {
    return data[row * PADDEDMAT_STRIDE + col];
  }
};

// ============================================================================
// Rodrigues' rotation formula (same as cuda_utilities.cuh)
// ============================================================================
__device__ __forceinline__ void build_rotation_matrix(
    double ax, double ay, double az, double angle, double* R) {
  double c = cos(angle);
  double s = sin(angle);
  double t = 1.0 - c;
  R[0]  = t * ax * ax + c;
  R[1]  = t * ax * ay + s * az;
  R[2]  = t * ax * az - s * ay;
  R[3]  = 0.0;
  R[4]  = t * ax * ay - s * az;
  R[5]  = t * ay * ay + c;
  R[6]  = t * ay * az + s * ax;
  R[7]  = 0.0;
  R[8]  = t * ax * az + s * ay;
  R[9]  = t * ay * az - s * ax;
  R[10] = t * az * az + c;
  R[11] = 0.0;
  R[12] = 0.0;
  R[13] = 0.0;
  R[14] = 0.0;
  R[15] = 1.0;
}

__device__ __forceinline__ void mat44_mul(const double* A, const double* B, double* C) {
  for (int r = 0; r < 4; ++r) {
    double a0 = A[r*4+0], a1 = A[r*4+1], a2 = A[r*4+2], a3 = A[r*4+3];
    C[r*4+0] = a0*B[0] + a1*B[4] + a2*B[8]  + a3*B[12];
    C[r*4+1] = a0*B[1] + a1*B[5] + a2*B[9]  + a3*B[13];
    C[r*4+2] = a0*B[2] + a1*B[6] + a2*B[10] + a3*B[14];
    C[r*4+3] = a0*B[3] + a1*B[7] + a2*B[11] + a3*B[15];
  }
}

// ============================================================================
// Pose error (same as cuda_utilities.cuh)
// ============================================================================
__device__ __forceinline__ void pose_error(const double* T_cur, const double* T_tgt,
                                            double* err) {
  err[0] = T_tgt[3]  - T_cur[3];
  err[1] = T_tgt[7]  - T_cur[7];
  err[2] = T_tgt[11] - T_cur[11];

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

__device__ __forceinline__ double rotation_geodesic_distance(const double* T_cur, const double* T_tgt) {
  double r00 = T_cur[0], r01 = T_cur[1], r02 = T_cur[2];
  double r10 = T_cur[4], r11 = T_cur[5], r12 = T_cur[6];
  double r20 = T_cur[8], r21 = T_cur[9], r22 = T_cur[10];
  double t00 = T_tgt[0], t01 = T_tgt[1], t02 = T_tgt[2];
  double t10 = T_tgt[4], t11 = T_tgt[5], t12 = T_tgt[6];
  double t20 = T_tgt[8], t21 = T_tgt[9], t22 = T_tgt[10];
  double e00 = r00 * t00 + r10 * t10 + r20 * t20;
  double e11 = r01 * t01 + r11 * t11 + r21 * t21;
  double e22 = r02 * t02 + r12 * t12 + r22 * t22;
  double trace = e00 + e11 + e22;
  return acos(cuda_clamp((trace - 1.0) * 0.5, -1.0, 1.0));
}

// ============================================================================
// 7×7 LDL^T Cholesky solve (new for 7DOF)
// ============================================================================
__device__ __forceinline__ void ldlt_solve_7x7(const double* H, const double* g, double* dq) {
  double L[7][7] = {{0}};
  double D[7] = {0};
  double y[7] = {0};
  double x[7] = {0};

  double A[7][7];
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j)
      A[i][j] = H[i * 7 + j];

  // LDL^T decomposition (unrolled by hand for 7×7)
  for (int j = 0; j < 7; ++j) {
    double d = A[j][j];
    for (int k = 0; k < j; ++k)
      d -= L[j][k] * L[j][k] * D[k];
    D[j] = d;

    for (int i = j + 1; i < 7; ++i) {
      double sum = A[i][j];
      for (int k = 0; k < j; ++k)
        sum -= L[i][k] * L[j][k] * D[k];
      L[i][j] = sum / D[j];
    }
    L[j][j] = 1.0;
  }

  // Forward substitution: L * y = g
  for (int i = 0; i < 7; ++i) {
    double sum = g[i];
    for (int k = 0; k < i; ++k)
      sum -= L[i][k] * y[k];
    y[i] = sum;
  }

  // Diagonal scaling
  for (int i = 0; i < 7; ++i)
    y[i] = y[i] / D[i];

  // Backward substitution: L^T * x = z
  for (int i = 6; i >= 0; --i) {
    double sum = y[i];
    for (int k = i + 1; k < 7; ++k)
      sum -= L[k][i] * x[k];
    x[i] = sum;
  }

  for (int i = 0; i < 7; ++i) dq[i] = x[i];
}

// ============================================================================
// Forward kinematics for Panda 7DOF
//
// Segments: 7 active joints + tail fixed offset to panda_link8
// Uses __constant__ memory for origins/axes/q_index/T_tcp
// ============================================================================
__device__ __forceinline__ void forward_kinematics_7dof(
    const double* q, double* T_tip,
    const double* origins, const double* axes, const int* q_index,
    const double* T_tcp) {
  // Initialize to I_4
  for (int i = 0; i < 16; ++i) T_tip[i] = (i % 5 == 0) ? 1.0 : 0.0;

  double T_tmp[16], R[16];

  for (int seg = 0; seg < 7; ++seg) {
    // T = T * origin
    mat44_mul(T_tip, &origins[seg * 16], T_tmp);
    for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];

    // T = T * Rodrigues(axis, q[index])
    double theta = q[q_index[seg]];
    build_rotation_matrix(axes[seg * 3 + 0], axes[seg * 3 + 1],
                          axes[seg * 3 + 2], theta, R);
    mat44_mul(T_tip, R, T_tmp);
    for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
  }

  // Apply tail fixed offset
  mat44_mul(T_tip, T_tcp, T_tmp);
  for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
}

// ============================================================================
// Main kernel: ik_batch_solve_7dof
//
// Grid:  (N, 1, 1)  — one block per target
// Block: (128, 1, 1) — 4 warps
//
// Warp assignments (same as 6DOF):
//   Warp 0: FK + error
//   Warp 1: Numerical Jacobian (7 columns)
//   Warp 2: Hessian construction (7×7)
//   Warp 3: LDLT solve + convergence check
// ============================================================================
__global__ void ik_batch_solve_7dof(
    const double* __restrict__ d_targets,   // [N, 16] target transforms
    const double* __restrict__ d_seeds,      // [N, 7]  initial seeds
    double* __restrict__ d_results,          // [N, 7]  output joint angles
    double* __restrict__ d_errors,           // [N, 2]  (pos_err, rot_err)
    double* __restrict__ d_iterations,       // [N]     iterations used
    const int    max_iter,                   // max iterations
    const double pos_tol,                    // position tolerance (m)
    const double orient_tol,                 // orientation tolerance (rad)
    const int    weight_level,               // weight schedule
    const int    N,                          // total targets
    // Device pointers for FK parameters (instead of __constant__)
    const double* __restrict__ d_origins,    // [112] = 7×16
    const double* __restrict__ d_axes,       // [21]  = 7×3
    const int*    __restrict__ d_q_index,    // [7]
    const double* __restrict__ d_T_tcp,      // [16]
    const double* __restrict__ d_joint_limits, // [14] = 7×2
    const double* __restrict__ d_weights     // [28] = 4 levels × 7 joints
) {
  int tid = blockIdx.x;
  if (tid >= N) return;

  // === Shared memory (7DOF sizes) ===
  __shared__ double s_q[8];                    // 7(+1 padding) doubles
  __shared__ double s_T[16];                   // Current FK result (4×4)
  __shared__ double s_T_tgt[16];               // Target transform
  __shared__ double s_J[6 * PADDEDMAT_STRIDE]; // Jacobian: 6×8 padded (48 doubles)
  __shared__ double s_H[7 * PADDEDMAT_STRIDE]; // Hessian: 7×8 padded (56 doubles)
  __shared__ double s_err[6];                  // Pose error (6-DOF)
  __shared__ double s_g[7];                    // Gradient (7-dim)
  __shared__ double s_dq[7];                   // Step (7-dim)
  __shared__ double s_q_best[7];               // Best q
  __shared__ int    s_converged;
  __shared__ int    s_iter_count;
  __shared__ double s_lambda;
  __shared__ double s_best_pos_err;
  __shared__ int    s_stagnation;

  // Type-safe matrix views
  PaddedMat6x8 J(s_J);  // Jacobian: 6 rows × padded stride
  PaddedMat7x8 H(s_H);  // Hessian:  7 rows × padded stride

  // === Phase 1: Load seed and target ===
  if (threadIdx.x < 7) {
    s_q[threadIdx.x] = d_seeds[tid * 7 + threadIdx.x];
  }
  if (threadIdx.x < 16) {
    s_T_tgt[threadIdx.x] = d_targets[tid * 16 + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    s_converged = 0;
    s_iter_count = 0;
    s_best_pos_err = 1e100;
    s_stagnation = 0;
  }
  __syncthreads();

  // === Phase 2: DLS iteration loop ===
  for (int iter = 0; iter < max_iter && !s_converged; ++iter) {
    if (threadIdx.x == 0) s_iter_count = iter + 1;

    // ---- 2a: Forward Kinematics (7 segments) ----
    if (threadIdx.x == 0) {
      forward_kinematics_7dof(s_q, s_T, d_origins, d_axes, d_q_index, d_T_tcp);
    }
    __syncthreads();

    // ---- 2b: Pose Error ----
    double s_pos_err;
    if (threadIdx.x == 0) {
      pose_error(s_T, s_T_tgt, s_err);
      s_pos_err = sqrt(s_err[0]*s_err[0] + s_err[1]*s_err[1] + s_err[2]*s_err[2]);
    }
    __syncthreads();

    // ---- 2c: Convergence check + best tracking ----
    if (threadIdx.x == 0) {
      double rot_err = rotation_geodesic_distance(s_T, s_T_tgt);
      if (s_pos_err <= pos_tol && rot_err <= orient_tol) {
        s_converged = 1;
      }
      if (s_pos_err < s_best_pos_err) {
        s_best_pos_err = s_pos_err;
        for (int i = 0; i < 7; ++i) s_q_best[i] = s_q[i];
        s_stagnation = 0;
      } else {
        s_stagnation++;
      }
    }
    __syncthreads();
    if (s_converged) break;

    // Stagnation recovery (A5+)
    if (threadIdx.x == 0 && s_stagnation > 25) {
      for (int i = 0; i < 7; ++i) s_q[i] = s_q_best[i];
      forward_kinematics_7dof(s_q, s_T, d_origins, d_axes, d_q_index, d_T_tcp);
      s_converged = 1;
    }
    __syncthreads();
    if (s_converged) break;

    // ---- 2d: Numerical Jacobian (7 columns, threads 0-6) ----
    if (threadIdx.x < 7) {
      int j = threadIdx.x;
      const double eps = 1e-6;
      double q_plus[7], q_minus[7], T_p[16], T_m[16];
      for (int i = 0; i < 7; ++i) {
        q_plus[i] = s_q[i];
        q_minus[i] = s_q[i];
      }
      q_plus[j]  += eps;
      q_minus[j] -= eps;

      forward_kinematics_7dof(q_plus, T_p, d_origins, d_axes, d_q_index, d_T_tcp);
      forward_kinematics_7dof(q_minus, T_m, d_origins, d_axes, d_q_index, d_T_tcp);

      double inv_2eps = 0.5 / eps;

      // Position columns
      J(0, j) = (T_p[3]  - T_m[3])  * inv_2eps;
      J(1, j) = (T_p[7]  - T_m[7])  * inv_2eps;
      J(2, j) = (T_p[11] - T_m[11]) * inv_2eps;

      // Rotation columns
      double r00 = s_T[0], r01 = s_T[1], r02 = s_T[2];
      double r10 = s_T[4], r11 = s_T[5], r12 = s_T[6];
      double r20 = s_T[8], r21 = s_T[9], r22 = s_T[10];

      double dR[9];
      dR[0] = (r00*T_p[0]+r10*T_p[4]+r20*T_p[8]) - (r00*T_m[0]+r10*T_m[4]+r20*T_m[8]);
      dR[1] = (r00*T_p[1]+r10*T_p[5]+r20*T_p[9]) - (r00*T_m[1]+r10*T_m[5]+r20*T_m[9]);
      dR[2] = (r00*T_p[2]+r10*T_p[6]+r20*T_p[10]) - (r00*T_m[2]+r10*T_m[6]+r20*T_m[10]);
      dR[3] = (r01*T_p[0]+r11*T_p[4]+r21*T_p[8]) - (r01*T_m[0]+r11*T_m[4]+r21*T_m[8]);
      dR[4] = (r01*T_p[1]+r11*T_p[5]+r21*T_p[9]) - (r01*T_m[1]+r11*T_m[5]+r21*T_m[9]);
      dR[5] = (r01*T_p[2]+r11*T_p[6]+r21*T_p[10]) - (r01*T_m[2]+r11*T_m[6]+r21*T_m[10]);
      dR[6] = (r02*T_p[0]+r12*T_p[4]+r22*T_p[8]) - (r02*T_m[0]+r12*T_m[4]+r22*T_m[8]);
      dR[7] = (r02*T_p[1]+r12*T_p[5]+r22*T_p[9]) - (r02*T_m[1]+r12*T_m[5]+r22*T_m[9]);
      dR[8] = (r02*T_p[2]+r12*T_p[6]+r22*T_p[10]) - (r02*T_m[2]+r12*T_m[6]+r22*T_m[10]);

      J(3, j) = (dR[7] - dR[5]) * 0.5 * inv_2eps;
      J(4, j) = (dR[2] - dR[6]) * 0.5 * inv_2eps;
      J(5, j) = (dR[3] - dR[1]) * 0.5 * inv_2eps;
    }
    __syncthreads();

    // ---- 2e: Adaptive damping (A5) ----
    if (threadIdx.x == 0) {
      double pos_err = s_pos_err;
      double lambda_base  = 2e-4;
      double lambda_far   = 5e-2;
      double lambda_floor = 1e-4;
      double lambda_scale = 8e-2;

      if (pos_err > 0.1) {
        s_lambda = fmax(lambda_base, lambda_far * (pos_err / lambda_scale));
        s_lambda = fmin(s_lambda, lambda_far * 3.0);
      } else {
        s_lambda = lambda_floor + lambda_base * (pos_err / lambda_scale);
      }
      if (s_stagnation > 5) {
        s_lambda *= (1.0 + 0.3 * (s_stagnation - 5));
        s_lambda = fmin(s_lambda, 0.5);
      }
    }
    __syncthreads();

    // ---- 2f: Hessian H = J^T·W^2·J + λ·I  (7×7) ----
    // 49 threads (0..48) each compute one (row, col) element
    if (threadIdx.x < 49) {
      int row = threadIdx.x / 7;  // 0..6
      int col = threadIdx.x % 7;  // 0..6

      double sum = 0.0;
      for (int k = 0; k < 6; ++k) {
        double w_k = d_weights[weight_level * 7 + k]; // Note: 7 weights per level now
        double w2 = w_k * w_k;
        sum += J(k, row) * w2 * J(k, col);
      }

      if (row == col) sum += s_lambda;

      H(row, col) = sum;
    }
    __syncthreads();

    // ---- 2g: Gradient g = J^T·W^2·e  (7-dim) ----
    if (threadIdx.x < 7) {
      double sum = 0.0;
      for (int k = 0; k < 6; ++k) {
        double w_k = d_weights[weight_level * 7 + k];
        sum += J(k, threadIdx.x) * w_k * w_k * s_err[k];
      }
      s_g[threadIdx.x] = sum;
    }
    __syncthreads();

    // ---- 2h: 7×7 LDLT Solve (serial, lane 0) ----
    if (threadIdx.x == 0) {
      double H_dense[49], g_dense[7];  // 7×7 = 49
      for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
          H_dense[r * 7 + c] = H(r, c);
        }
        g_dense[r] = s_g[r];
      }
      ldlt_solve_7x7(H_dense, g_dense, s_dq);
    }
    __syncthreads();

    // ---- 2i: Apply step with joint limits ----
    if (threadIdx.x < 7) {
      int i = threadIdx.x;
      double lo = d_joint_limits[i * 2 + 0];
      double hi = d_joint_limits[i * 2 + 1];
      s_q[i] = cuda_clamp(s_q[i] + s_dq[i], lo, hi);
    }
    __syncthreads();
  }

  // === Phase 3: Write results ===
  if (threadIdx.x == 0) {
    // Final FK
    forward_kinematics_7dof(s_q, s_T, d_origins, d_axes, d_q_index, d_T_tcp);
  }
  __syncthreads();

  if (threadIdx.x < 7) {
    d_results[tid * 7 + threadIdx.x] = s_q[threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    pose_error(s_T, s_T_tgt, s_err);
    double pos_err = sqrt(s_err[0]*s_err[0] + s_err[1]*s_err[1] + s_err[2]*s_err[2]);
    double rot_err = rotation_geodesic_distance(s_T, s_T_tgt);
    d_errors[tid * 2 + 0] = pos_err;
    d_errors[tid * 2 + 1] = rot_err;
    d_iterations[tid] = (double)s_iter_count;
  }
}

// ============================================================================
// Kernel launch host wrapper
// ============================================================================
cudaError_t launch_ik_batch_solve_7dof(
    const double* d_targets, const double* d_seeds,
    double* d_results, double* d_errors, double* d_iterations,
    int max_iter, double pos_tol, double orient_tol, int weight_level, int N,
    const double* d_origins, const double* d_axes, const int* d_q_index,
    const double* d_T_tcp, const double* d_joint_limits,
    const double* d_weights,
    cudaStream_t stream = 0)
{
  dim3 grid(N, 1, 1);
  dim3 block(128, 1, 1);

  ik_batch_solve_7dof<<<grid, block, 0, stream>>>(
      d_targets, d_seeds, d_results, d_errors, d_iterations,
      max_iter, pos_tol, orient_tol, weight_level, N,
      d_origins, d_axes, d_q_index, d_T_tcp, d_joint_limits, d_weights);

  return cudaGetLastError();
}

// ============================================================================
// FK verification kernel: compute FK(q) and write transform
// ============================================================================
__global__ void fk_verify_7dof(
    const double* __restrict__ d_q,          // [N, 7] joint angles
    double* __restrict__ d_T,                // [N, 16] output transforms
    int N,
    const double* __restrict__ d_origins,
    const double* __restrict__ d_axes,
    const int*    __restrict__ d_q_index,
    const double* __restrict__ d_T_tcp)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  double q[7];
  for (int i = 0; i < 7; ++i) q[i] = d_q[tid * 7 + i];

  double T[16];
  forward_kinematics_7dof(q, T, d_origins, d_axes, d_q_index, d_T_tcp);

  for (int i = 0; i < 16; ++i) {
    d_T[tid * 16 + i] = T[i];
  }
}
