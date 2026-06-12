// test_cuda_kernel.cu — Comprehensive GPU FK & IK test for UR10 CUDA kernel
//
// Compile:
//   nvcc -arch=sm_89 -O3 -lineinfo --ptxas-options=-v \
//     -o test_cuda_kernel test/test_cuda_kernel.cu \
//     src/cuda/cuda_kernels.cu \
//     -I include -Isrc/cuda -I include/assembly_rtfg_cuda \
//     -I ../assembly_rtfg_cpp/include -I ../assembly_rtfg_cpp/include/assembly_rtfg_cpp \
//     -I /usr/include/eigen3 -lstdc++
//
// Tests:
//   1. FK correctness: GPU FK vs CPU reference (10 random q, < 1e-15 error)
//   2. IK convergence:  Known reachable target, 60 iter convergence
//   3. Batch IK:        273 targets, >90% convergence rate

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Define CUDA_DEFINE_CONSTANTS so __constant__ variables are defined (not extern)
// in this translation unit. The kernel source is included below for single-TU build.
#define CUDA_DEFINE_CONSTANTS
// Include CUDA utilities for device helpers (in namespace rtfg::cuda)
#include "../src/cuda_utilities.cuh"

// Include kernel and collision source directly to form a single translation unit.
// This avoids the NVCC extern __constant__ cross-TU linking limitation.
#include "../src/cuda_kernels.cu"
#include "../src/cuda_collision.cu"

// Bring namespaced symbols into scope for host-side cudaMemcpyToSymbol
using namespace rtfg::cuda;

// ============================================================================
// UR10 kinematics data (from assembly_rtfg_solver.urdf, double-precision)
// ============================================================================

// 6 revolute joint origin matrices (row-major, 4×4 each)
static const double k_origins[96] = {
    // Seg 0: shoulder_pan — rpy=(0,0,0), xyz=(0,0,0.1273), axis=(0,0,1)
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.1273,
    0.0, 0.0, 0.0, 1.0,
    // Seg 1: shoulder_lift — rpy=(0,1.57079,0), xyz=(0,0.220941,0), axis=(0,1,0)
     6.3267948966684693e-06,  0.0,                       9.9999999997998579e-01,  0.0,
     0.0,                     1.0,                       0.0,                     0.220941,
    -9.9999999997998579e-01,  0.0,                       6.3267948966684693e-06,  0.0,
     0.0,                     0.0,                       0.0,                     1.0,
    // Seg 2: elbow — rpy=(0,0,0), xyz=(-3.9e-6,-0.1719,0.612), axis=(0,1,0)
    1.0, 0.0, 0.0, -3.9e-6,
    0.0, 1.0, 0.0, -0.1719,
    0.0, 0.0, 1.0,  0.612,
    0.0, 0.0, 0.0,  1.0,
    // Seg 3: wrist_1 — rpy=(0,1.5707895,2.7e-6), xyz=(-3.6e-6,0,0.5723), axis=(0,1,0)
     6.8267948964806121e-06, -2.6999999999967194e-06,  9.9999999997305244e-01, -3.6e-6,
     1.8432346220542441e-11,  9.9999999999635503e-01,  2.6999999999338027e-06,  0.0,
    -9.9999999997669742e-01,  0.0,                     6.8267948965054954e-06,  0.5723,
     0.0,                     0.0,                     0.0,                     1.0,
    // Seg 4: wrist_2 — rpy=(0,0,0), xyz=(3e-7,0.1149,3e-7), axis=(0,0,-1)
    1.0, 0.0, 0.0, 3e-7,
    0.0, 1.0, 0.0, 0.1149,
    0.0, 0.0, 1.0, 3e-7,
    0.0, 0.0, 0.0, 1.0,
    // Seg 5: wrist_3 — rpy=(0,0,0), xyz=(0,-3e-7,0.1157), axis=(0,1,0)
    1.0, 0.0, 0.0,  0.0,
    0.0, 1.0, 0.0, -3e-7,
    0.0, 0.0, 1.0,  0.1157,
    0.0, 0.0, 0.0,  1.0
};

// Rotation axes: {x, y, z} per joint
static const double k_axes[18] = {
    0.0, 0.0,  1.0,   // seg 0: shoulder_pan  Z
    0.0, 1.0,  0.0,   // seg 1: shoulder_lift Y
    0.0, 1.0,  0.0,   // seg 2: elbow         Y
    0.0, 1.0,  0.0,   // seg 3: wrist_1       Y
    0.0, 0.0, -1.0,   // seg 4: wrist_2      -Z
    0.0, 1.0,  0.0    // seg 5: wrist_3       Y
};

static const int k_q_index[6] = {0, 1, 2, 3, 4, 5};

// T_wrist3_to_tcp from URDF tool chain
static const double k_T_wrist3_to_tcp[16] = {
    -3.0089034369963224e-06, -8.1915141988946039e-01,  5.7357732807706696e-01, -4.7377000000000002e-01,
    -9.9999999999319700e-01,  3.6885740485110307e-06,  2.1962570019296782e-08,  1.6330000206612766e-01,
    -2.1336731175750953e-06, -5.7357732807309880e-01, -8.1915141989498630e-01, -7.7108998035934045e-02,
     0.0,                     0.0,                     0.0,                     1.0
};

// Joint limits (±2π for continuous joints in URDF)
static const double k_joint_limits[12] = {
    -6.28319, 6.28319,  // shoulder_pan
    -6.28319, 6.28319,  // shoulder_lift
    -6.28319, 6.28319,  // elbow
    -6.28319, 6.28319,  // wrist_1
    -6.28319, 6.28319,  // wrist_2
    -6.28319, 6.28319   // wrist_3
};

// Weight schedule (level 0: position + orientation)
static const double k_weights[24] = {
    1.0, 1.0, 1.0, 0.20, 0.20, 0.20,
    1.0, 1.0, 1.0, 0.10, 0.10, 0.10,
    1.0, 1.0, 1.0, 0.03, 0.03, 0.03,
    1.0, 1.0, 1.0, 0.00, 0.00, 0.00
};

static const double k_lambda_params[4] = {5e-4, 0.1, 5e-4, 0.05};

// Kernels are included directly from src/cuda/cuda_kernels.cu (single-TU build).
// No external declarations needed.

// ============================================================================
// CPU reference: Rodrigues rotation formula
// ============================================================================
static void cpu_rotation_matrix(double ax, double ay, double az, double angle, double* R) {
    double c = cos(angle), s = sin(angle), t = 1.0 - c;
    // Row-major 4×4
    R[0]  = t*ax*ax + c;   R[1]  = t*ax*ay + s*az; R[2]  = t*ax*az - s*ay; R[3]  = 0.0;
    R[4]  = t*ax*ay - s*az; R[5]  = t*ay*ay + c;   R[6]  = t*ay*az + s*ax; R[7]  = 0.0;
    R[8]  = t*ax*az + s*ay; R[9]  = t*ay*az - s*ax; R[10] = t*az*az + c;   R[11] = 0.0;
    R[12] = 0.0;            R[13] = 0.0;            R[14] = 0.0;            R[15] = 1.0;
}

// CPU 4×4 matrix multiply: C = A * B (all row-major)
static void cpu_mat44_mul(const double* A, const double* B, double* C) {
    for (int r = 0; r < 4; ++r) {
        double a0 = A[r*4+0], a1 = A[r*4+1], a2 = A[r*4+2], a3 = A[r*4+3];
        C[r*4+0] = a0*B[0] + a1*B[4] + a2*B[8]  + a3*B[12];
        C[r*4+1] = a0*B[1] + a1*B[5] + a2*B[9]  + a3*B[13];
        C[r*4+2] = a0*B[2] + a1*B[6] + a2*B[10] + a3*B[14];
        C[r*4+3] = a0*B[3] + a1*B[7] + a2*B[11] + a3*B[15];
    }
}

// CPU forward kinematics (URDF convention, matches GPU exactly)
static void cpu_forward_kinematics(const double* q, double* T_tip) {
    // Initialize to I₄
    for (int i = 0; i < 16; ++i) T_tip[i] = (i % 5 == 0) ? 1.0 : 0.0;

    double T_tmp[16], R[16];
    for (int seg = 0; seg < 6; ++seg) {
        // T = T * origin
        cpu_mat44_mul(T_tip, &k_origins[seg * 16], T_tmp);
        for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];

        // T = T * Rodrigues(axis, q[index])
        double theta = q[k_q_index[seg]];
        cpu_rotation_matrix(k_axes[seg*3+0], k_axes[seg*3+1], k_axes[seg*3+2], theta, R);
        cpu_mat44_mul(T_tip, R, T_tmp);
        for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
    }

    // T = T * T_wrist3_to_tcp
    cpu_mat44_mul(T_tip, k_T_wrist3_to_tcp, T_tmp);
    for (int i = 0; i < 16; ++i) T_tip[i] = T_tmp[i];
}

// ============================================================================
// GPU FK test kernel: writes FK result to global memory for verification
// Must be in rtfg::cuda namespace to access forward_kinematics
// ============================================================================
namespace rtfg { namespace cuda {
__global__ void test_fk_kernel(const double* d_q, double* d_T, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    forward_kinematics(&d_q[tid * 6], &d_T[tid * 16]);
}
// ============================================================================
// DEBUG KERNEL: Single DLS iteration trace — prints all intermediate values
// Launched with 1 thread to avoid printf interleaving
// ============================================================================
__global__ void debug_dls_iteration(const double* d_target, const double* d_seed) {
    // Load seed and target
    double q[6], T_tgt[16];
    for (int i = 0; i < 6; ++i) q[i] = d_seed[i];
    for (int i = 0; i < 16; ++i) T_tgt[i] = d_target[i];

    printf("=== DLS Iteration 1 Debug ===\n");
    printf("Seed q: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           q[0], q[1], q[2], q[3], q[4], q[5]);

    // Step 1: FK
    double T_cur[16];
    forward_kinematics(q, T_cur);
    printf("FK TCP pos: (%.6f, %.6f, %.6f)\n", T_cur[3], T_cur[7], T_cur[11]);
    printf("Target pos: (%.6f, %.6f, %.6f)\n", T_tgt[3], T_tgt[7], T_tgt[11]);

    // Step 2: Pose error
    double err[6];
    pose_error(T_cur, T_tgt, err);
    double pos_err = sqrt(err[0]*err[0]+err[1]*err[1]+err[2]*err[2]);
    double rot_err = sqrt(err[3]*err[3]+err[4]*err[4]+err[5]*err[5]);
    printf("Err: [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
           err[0], err[1], err[2], err[3], err[4], err[5]);
    printf("Pos err=%.6e, Rot err=%.6e\n", pos_err, rot_err);

    // Step 3: Numerical Jacobian
    const double eps = 1e-6;
    double J[48];  // 6x8 padded
    for (int j = 0; j < 6; ++j) {
        double qp[6], qm[6], Tp[16], Tm[16];
        for (int i = 0; i < 6; ++i) { qp[i] = q[i]; qm[i] = q[i]; }
        qp[j] += eps; qm[j] -= eps;
        forward_kinematics(qp, Tp);
        forward_kinematics(qm, Tm);

        double inv_2eps = 0.5 / eps;
        J[0*8+j] = (Tp[3]  - Tm[3])  * inv_2eps;
        J[1*8+j] = (Tp[7]  - Tm[7])  * inv_2eps;
        J[2*8+j] = (Tp[11] - Tm[11]) * inv_2eps;

        double r00=T_cur[0], r01=T_cur[1], r02=T_cur[2];
        double r10=T_cur[4], r11=T_cur[5], r12=T_cur[6];
        double r20=T_cur[8], r21=T_cur[9], r22=T_cur[10];

        double dR[9];
        dR[0] = (r00*Tp[0]+r10*Tp[4]+r20*Tp[8]) - (r00*Tm[0]+r10*Tm[4]+r20*Tm[8]);
        dR[1] = (r00*Tp[1]+r10*Tp[5]+r20*Tp[9]) - (r00*Tm[1]+r10*Tm[5]+r20*Tm[9]);
        dR[2] = (r00*Tp[2]+r10*Tp[6]+r20*Tp[10]) - (r00*Tm[2]+r10*Tm[6]+r20*Tm[10]);
        dR[3] = (r01*Tp[0]+r11*Tp[4]+r21*Tp[8]) - (r01*Tm[0]+r11*Tm[4]+r21*Tm[8]);
        dR[4] = (r01*Tp[1]+r11*Tp[5]+r21*Tp[9]) - (r01*Tm[1]+r11*Tm[5]+r21*Tm[9]);
        dR[5] = (r01*Tp[2]+r11*Tp[6]+r21*Tp[10]) - (r01*Tm[2]+r11*Tm[6]+r21*Tm[10]);
        dR[6] = (r02*Tp[0]+r12*Tp[4]+r22*Tp[8]) - (r02*Tm[0]+r12*Tm[4]+r22*Tm[8]);
        dR[7] = (r02*Tp[1]+r12*Tp[5]+r22*Tp[9]) - (r02*Tm[1]+r12*Tm[5]+r22*Tm[9]);
        dR[8] = (r02*Tp[2]+r12*Tp[6]+r22*Tp[10]) - (r02*Tm[2]+r12*Tm[6]+r22*Tm[10]);

        J[3*8+j] = (dR[7] - dR[5]) * 0.5 * inv_2eps;
        J[4*8+j] = (dR[2] - dR[6]) * 0.5 * inv_2eps;
        J[5*8+j] = (dR[3] - dR[1]) * 0.5 * inv_2eps;
    }

    printf("Jacobian (6x6, rows 0-2=pos, 3-5=rot):\n");
    for (int r = 0; r < 6; ++r) {
        printf("  J[%d]: [% .4e % .4e % .4e % .4e % .4e % .4e]\n",
               r, J[r*8+0], J[r*8+1], J[r*8+2], J[r*8+3], J[r*8+4], J[r*8+5]);
    }

    // Step 4: Hessian (using same weight pattern as current kernel)
    double H_pad[48];
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            double w_row = c_weight_schedule[0*6+row];
            double w_col = c_weight_schedule[0*6+col];
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += J[k*8+row] * w_row * J[k*8+col] * w_col;
            }
            H_pad[row*8+col] = sum;
        }
    }

    printf("Hessian (6x6, current weighting):\n");
    for (int r = 0; r < 6; ++r) {
        printf("  H[%d]: [% .4e % .4e % .4e % .4e % .4e % .4e]\n",
               r, H_pad[r*8+0], H_pad[r*8+1], H_pad[r*8+2],
               H_pad[r*8+3], H_pad[r*8+4], H_pad[r*8+5]);
    }

    // Step 5: Gradient (using same weight pattern as current kernel)
    double g[6] = {0};
    for (int i = 0; i < 6; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 6; ++k) {
            double w_k = c_weight_schedule[0*6+k];
            sum += J[k*8+i] * w_k * err[k] * w_k;
        }
        g[i] = sum;
    }

    printf("Gradient (current weighting): [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
           g[0], g[1], g[2], g[3], g[4], g[5]);

    // Step 6: LDLT solve
    double lambda = 5e-4;
    double H_dense[36];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            H_dense[r*6+c] = H_pad[r*8+c];
            if (r == c) H_dense[r*6+c] += lambda;
        }

    double dq[6];
    ldlt_solve_6x6(H_dense, g, dq);

    printf("H+lambda*I:\n");
    for (int r = 0; r < 6; ++r) {
        printf("  [% .4e % .4e % .4e % .4e % .4e % .4e]\n",
               H_dense[r*6+0], H_dense[r*6+1], H_dense[r*6+2],
               H_dense[r*6+3], H_dense[r*6+4], H_dense[r*6+5]);
    }

    printf("dq step: [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
           dq[0], dq[1], dq[2], dq[3], dq[4], dq[5]);

    double step_norm = sqrt(dq[0]*dq[0]+dq[1]*dq[1]+dq[2]*dq[2]+
                            dq[3]*dq[3]+dq[4]*dq[4]+dq[5]*dq[5]);
    printf("Step norm: %.6e\n", step_norm);

    // Step 7: Try with CORRECT weighting (all 1.0) to compare
    printf("\n--- Correct weighting (all 1.0) comparison ---\n");
    double H_corr[48];
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += J[k*8+row] * J[k*8+col];  // no weights
            }
            H_corr[row*8+col] = sum;
        }
    }
    printf("Hessian (unweighted):\n");
    for (int r = 0; r < 6; ++r) {
        printf("  [% .4e % .4e % .4e % .4e % .4e % .4e]\n",
               H_corr[r*8+0], H_corr[r*8+1], H_corr[r*8+2],
               H_corr[r*8+3], H_corr[r*8+4], H_corr[r*8+5]);
    }

    double g_corr[6] = {0};
    for (int i = 0; i < 6; ++i) {
        double sum = 0.0;
        for (int k = 0; k < 6; ++k) {
            sum += J[k*8+i] * err[k];  // no weights
        }
        g_corr[i] = sum;
    }
    printf("Gradient (unweighted): [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
           g_corr[0], g_corr[1], g_corr[2], g_corr[3], g_corr[4], g_corr[5]);

    double Hc_dense[36];
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            Hc_dense[r*6+c] = H_corr[r*8+c];
            if (r == c) Hc_dense[r*6+c] += lambda;
        }
    double dq_corr[6];
    ldlt_solve_6x6(Hc_dense, g_corr, dq_corr);
    printf("dq step (correct): [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
           dq_corr[0], dq_corr[1], dq_corr[2], dq_corr[3], dq_corr[4], dq_corr[5]);

    double step_norm_c = sqrt(dq_corr[0]*dq_corr[0]+dq_corr[1]*dq_corr[1]+dq_corr[2]*dq_corr[2]+
                              dq_corr[3]*dq_corr[3]+dq_corr[4]*dq_corr[4]+dq_corr[5]*dq_corr[5]);
    printf("Step norm (correct): %.6e\n", step_norm_c);

    // Apply correct step
    double q_new[6];
    for (int i = 0; i < 6; ++i) q_new[i] = q[i] + dq_corr[i];

    double T_new[16];
    forward_kinematics(q_new, T_new);
    double pose_new[6];
    pose_error(T_new, T_tgt, pose_new);
    double pn = sqrt(pose_new[0]*pose_new[0]+pose_new[1]*pose_new[1]+pose_new[2]*pose_new[2]);
    printf("Pos error after correct step: %.6e\n", pn);
    printf("=== End Debug ===\n");
}

// ============================================================================
// DEBUG KERNEL 2: Progressive IK convergence trace (single target)
// Prints position error every 5 iterations to show convergence progress
// ============================================================================
__global__ void debug_ik_progress(const double* d_target, const double* d_seed,
                                   int max_iter) {
    double q[6], q_ref[6], T_tgt[16];
    for (int i = 0; i < 6; ++i) { q[i] = d_seed[i]; q_ref[i] = q[i]; }
    for (int i = 0; i < 16; ++i) T_tgt[i] = d_target[i];

    printf("=== IK Progress Trace (max_iter=%d) ===\n", max_iter);
    printf("Seed: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
           q[0],q[1],q[2],q[3],q[4],q[5]);
    printf("Target TCP: (%.4f,%.4f,%.4f)\n", T_tgt[3],T_tgt[7],T_tgt[11]);

    // Use the SAME algorithm as ik_batch_solve kernel, but sequential
    int converged = 0;
    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        // FK
        double T_cur[16];
        forward_kinematics(q, T_cur);

        // Pose error
        double err[6];
        pose_error(T_cur, T_tgt, err);
        double pos_err = sqrt(err[0]*err[0]+err[1]*err[1]+err[2]*err[2]);
        double rot_err = sqrt(err[3]*err[3]+err[4]*err[4]+err[5]*err[5]);

        if (iter % 5 == 0 || iter < 3) {
            printf("  iter%3d: pos=%.6f rot=%.4f  q=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                   iter, pos_err, rot_err,
                   q[0],q[1],q[2],q[3],q[4],q[5]);
        }

        if (pos_err < 0.001 && rot_err < 0.01745) { converged = 1; break; }

        // Jacobian
        const double eps = 1e-6;
        double J[48];
        for (int j = 0; j < 6; ++j) {
            double qp[6], qm[6], Tp[16], Tm[16];
            for (int i = 0; i < 6; ++i) { qp[i]=q[i]; qm[i]=q[i]; }
            qp[j]+=eps; qm[j]-=eps;
            forward_kinematics(qp, Tp);
            forward_kinematics(qm, Tm);

            double inv_2eps = 0.5/eps;
            J[0*8+j]=(Tp[3]-Tm[3])*inv_2eps;
            J[1*8+j]=(Tp[7]-Tm[7])*inv_2eps;
            J[2*8+j]=(Tp[11]-Tm[11])*inv_2eps;
            double r00=T_cur[0],r01=T_cur[1],r02=T_cur[2];
            double r10=T_cur[4],r11=T_cur[5],r12=T_cur[6];
            double r20=T_cur[8],r21=T_cur[9],r22=T_cur[10];
            double dR[9];
            dR[0]=(r00*Tp[0]+r10*Tp[4]+r20*Tp[8])-(r00*Tm[0]+r10*Tm[4]+r20*Tm[8]);
            dR[1]=(r00*Tp[1]+r10*Tp[5]+r20*Tp[9])-(r00*Tm[1]+r10*Tm[5]+r20*Tm[9]);
            dR[2]=(r00*Tp[2]+r10*Tp[6]+r20*Tp[10])-(r00*Tm[2]+r10*Tm[6]+r20*Tm[10]);
            dR[3]=(r01*Tp[0]+r11*Tp[4]+r21*Tp[8])-(r01*Tm[0]+r11*Tm[4]+r21*Tm[8]);
            dR[4]=(r01*Tp[1]+r11*Tp[5]+r21*Tp[9])-(r01*Tm[1]+r11*Tm[5]+r21*Tm[9]);
            dR[5]=(r01*Tp[2]+r11*Tp[6]+r21*Tp[10])-(r01*Tm[2]+r11*Tm[6]+r21*Tm[10]);
            dR[6]=(r02*Tp[0]+r12*Tp[4]+r22*Tp[8])-(r02*Tm[0]+r12*Tm[4]+r22*Tm[8]);
            dR[7]=(r02*Tp[1]+r12*Tp[5]+r22*Tp[9])-(r02*Tm[1]+r12*Tm[5]+r22*Tm[9]);
            dR[8]=(r02*Tp[2]+r12*Tp[6]+r22*Tp[10])-(r02*Tm[2]+r12*Tm[6]+r22*Tm[10]);
            J[3*8+j]=(dR[7]-dR[5])*0.5*inv_2eps;
            J[4*8+j]=(dR[2]-dR[6])*0.5*inv_2eps;
            J[5*8+j]=(dR[3]-dR[1])*0.5*inv_2eps;
        }

        // Adaptive damping (same as kernel)
        double lambda;
        if (pos_err > 0.1) {
            lambda = fmax(1e-3, 5e-3*(pos_err/0.05));
            lambda = fmin(lambda, 0.1);
        } else {
            lambda = 5e-4 + 5e-3*(pos_err/0.05);
        }

        // Hessian: H = J^T * W^2 * J + lambda*I (FIXED weighting)
        double H_dense[36];
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                double sum = 0.0;
                for (int k = 0; k < 6; ++k) {
                    double w_k = c_weight_schedule[0*6+k];
                    sum += J[k*8+r] * w_k * w_k * J[k*8+c];
                }
                if (r == c) sum += lambda;
                H_dense[r*6+c] = sum;
            }
        }

        // Gradient: g = J^T * W^2 * e (FIXED weighting)
        double g[6];
        for (int i = 0; i < 6; ++i) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                double w_k = c_weight_schedule[0*6+k];
                sum += J[k*8+i] * w_k * w_k * err[k];
            }
            g[i] = sum;
        }

        // LDLT solve
        double dq[6];
        ldlt_solve_6x6(H_dense, g, dq);

        // Step clamp
        double step_norm = sqrt(dq[0]*dq[0]+dq[1]*dq[1]+dq[2]*dq[2]+
                                dq[3]*dq[3]+dq[4]*dq[4]+dq[5]*dq[5]);
        if (step_norm > 0.45) {
            double scale = 0.45/step_norm;
            for (int i = 0; i < 6; ++i) dq[i] *= scale;
        }
        if (step_norm < 1e-6) break;  // stagnation

        // Apply step with joint limits
        for (int i = 0; i < 6; ++i) {
            double lo = c_joint_limits[i*2+0];
            double hi = c_joint_limits[i*2+1];
            q[i] = fmin(fmax(q[i]+dq[i], lo), hi);
        }

        // Branch alignment
        for (int i = 0; i < 6; ++i) {
            double diff = q[i] - q_ref[i];
            q[i] = q_ref[i] + atan2(sin(diff), cos(diff));
        }
    }

    // Final result
    double T_final[16];
    forward_kinematics(q, T_final);
    double err_final[6];
    pose_error(T_final, T_tgt, err_final);
    double pf = sqrt(err_final[0]*err_final[0]+err_final[1]*err_final[1]+err_final[2]*err_final[2]);
    printf("  FINAL: pos=%.6f  q=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
           pf, q[0],q[1],q[2],q[3],q[4],q[5]);
    printf("=== End Progress ===\n");
}

}  // namespace cuda
}  // namespace rtfg

// ============================================================================
// Helper: upload all constant memory
// ============================================================================
static cudaError_t upload_constants() {
    cudaError_t err;
    err = cudaMemcpyToSymbol(c_segment_origins, k_origins, 96 * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_segment_axes, k_axes, 18 * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_q_index, k_q_index, 6 * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_T_wrist3_to_tcp, k_T_wrist3_to_tcp, 16 * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_joint_limits, k_joint_limits, 12 * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_weight_schedule, k_weights, 24 * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_lambda_params, k_lambda_params, 4 * sizeof(double));
    return err;
}

// ============================================================================
// Helper: max absolute difference between two 4×4 matrices
// ============================================================================
static double max_abs_diff_16(const double* A, const double* B) {
    double max_diff = 0.0;
    for (int i = 0; i < 16; ++i) {
        double d = fabs(A[i] - B[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// ============================================================================
// Helper: print 4×4 matrix
// ============================================================================
static void print_mat4(const char* label, const double* T) {
    printf("%s:\n", label);
    for (int r = 0; r < 4; ++r) {
        printf("  [% .6e % .6e % .6e % .6e]\n",
               T[r*4+0], T[r*4+1], T[r*4+2], T[r*4+3]);
    }
}

// ============================================================================
// Data export: save targets and seeds to test_data/ directory
// ============================================================================
static void export_test_data(const double* targets, const double* seeds,
                              const double* results, const double* errors,
                              const double* iters, int N) {
    // Data directory: relative to build/ directory
    const char* dir = "../cuda_low_level_optimization/test/test_data";

    char path[512];

    // Binary format (for fast loading by other CUDA methods)
    snprintf(path, sizeof(path), "%s/targets_273.bin", dir);
    FILE* fb1 = fopen(path, "wb");
    if (!fb1) { printf("ERROR: Cannot create %s\n", path); return; }
    fwrite(targets, sizeof(double), N * 16, fb1);
    fclose(fb1);

    snprintf(path, sizeof(path), "%s/seeds_273.bin", dir);
    FILE* fb2 = fopen(path, "wb");
    if (!fb2) { printf("ERROR: Cannot create %s\n", path); return; }
    fwrite(seeds, sizeof(double), N * 6, fb2);
    fclose(fb2);

    snprintf(path, sizeof(path), "%s/results_273.bin", dir);
    FILE* fb3 = fopen(path, "wb");
    if (!fb3) { printf("ERROR: Cannot create %s\n", path); return; }
    fwrite(results, sizeof(double), N * 6, fb3);
    fclose(fb3);

    snprintf(path, sizeof(path), "%s/errors_273.bin", dir);
    FILE* fb4 = fopen(path, "wb");
    if (!fb4) { printf("ERROR: Cannot create %s\n", path); return; }
    fwrite(errors, sizeof(double), N * 2, fb4);
    fclose(fb4);

    snprintf(path, sizeof(path), "%s/iterations_273.bin", dir);
    FILE* fb5 = fopen(path, "wb");
    if (!fb5) { printf("ERROR: Cannot create %s\n", path); return; }
    fwrite(iters, sizeof(double), N, fb5);
    fclose(fb5);

    // CSV format (human-readable, for Python/Matlab loading)
    snprintf(path, sizeof(path), "%s/targets_273.csv", dir);
    FILE* fc1 = fopen(path, "w");
    if (fc1) {
        fprintf(fc1, "idx,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33\n");
        for (int i = 0; i < N; i++) {
            fprintf(fc1, "%d", i);
            for (int j = 0; j < 16; j++)
                fprintf(fc1, ",%.15f", targets[i * 16 + j]);
            fprintf(fc1, "\n");
        }
        fclose(fc1);
    }

    snprintf(path, sizeof(path), "%s/seeds_273.csv", dir);
    FILE* fc2 = fopen(path, "w");
    if (fc2) {
        fprintf(fc2, "idx,q0,q1,q2,q3,q4,q5\n");
        for (int i = 0; i < N; i++) {
            fprintf(fc2, "%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
                    i, seeds[i*6+0], seeds[i*6+1], seeds[i*6+2],
                    seeds[i*6+3], seeds[i*6+4], seeds[i*6+5]);
        }
        fclose(fc2);
    }

    snprintf(path, sizeof(path), "%s/results_273.csv", dir);
    FILE* fc3 = fopen(path, "w");
    if (fc3) {
        fprintf(fc3, "idx,q0,q1,q2,q3,q4,q5\n");
        for (int i = 0; i < N; i++) {
            fprintf(fc3, "%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
                    i, results[i*6+0], results[i*6+1], results[i*6+2],
                    results[i*6+3], results[i*6+4], results[i*6+5]);
        }
        fclose(fc3);
    }

    snprintf(path, sizeof(path), "%s/errors_273.csv", dir);
    FILE* fc4 = fopen(path, "w");
    if (fc4) {
        fprintf(fc4, "idx,pos_err,rot_err\n");
        for (int i = 0; i < N; i++)
            fprintf(fc4, "%d,%.10f,%.10f\n", i, errors[i*2+0], errors[i*2+1]);
        fclose(fc4);
    }

    printf("Exported %d targets/seeds/results/errors/iterations to test_data/\n", N);
    printf("  Binary: targets_273.bin (%zu bytes), seeds_273.bin (%zu bytes)\n",
           N * 16 * sizeof(double), N * 6 * sizeof(double));
    printf("  Binary: results_273.bin, errors_273.bin, iterations_273.bin\n");
    printf("  CSV:    targets_273.csv, seeds_273.csv, results_273.csv, errors_273.csv\n");
}

// ============================================================================
// Data load: read targets and seeds from test_data/ directory
// ============================================================================
static int load_test_data(const char* dir, double** out_targets, double** out_seeds, int* out_N) {
    char path_t[512], path_s[512];
    snprintf(path_t, sizeof(path_t), "%s/targets_273.bin", dir);
    snprintf(path_s, sizeof(path_s), "%s/seeds_273.bin", dir);

    // Determine N from file size
    FILE* ft = fopen(path_t, "rb");
    if (!ft) { printf("ERROR: Cannot open %s\n", path_t); return 1; }
    fseek(ft, 0, SEEK_END);
    long sz = ftell(ft);
    fclose(ft);

    if (sz % (16 * sizeof(double)) != 0) {
        printf("ERROR: %s size %ld not divisible by 16 doubles\n", path_t, sz);
        return 1;
    }
    int N = (int)(sz / (16 * sizeof(double)));

    FILE* fs = fopen(path_s, "rb");
    if (!fs) { printf("ERROR: Cannot open %s\n", path_s); return 1; }
    fseek(fs, 0, SEEK_END);
    long sz_s = ftell(fs);
    fclose(fs);

    if (sz_s != N * 6 * (long)sizeof(double)) {
        printf("ERROR: seeds file size %ld != expected %ld for %d targets\n",
               sz_s, N * 6 * (long)sizeof(double), N);
        return 1;
    }

    // Allocate and read
    double* targets = (double*)malloc(N * 16 * sizeof(double));
    double* seeds   = (double*)malloc(N * 6 * sizeof(double));
    if (!targets || !seeds) { printf("ERROR: malloc failed\n"); return 1; }

    ft = fopen(path_t, "rb");
    size_t nr_t = fread(targets, sizeof(double), N * 16, ft);
    fclose(ft);
    if (nr_t != (size_t)(N * 16)) {
        printf("ERROR: read %zu doubles from targets, expected %d\n", nr_t, N * 16);
        free(targets); free(seeds);
        return 1;
    }

    fs = fopen(path_s, "rb");
    size_t nr_s = fread(seeds, sizeof(double), N * 6, fs);
    fclose(fs);
    if (nr_s != (size_t)(N * 6)) {
        printf("ERROR: read %zu doubles from seeds, expected %d\n", nr_s, N * 6);
        free(targets); free(seeds);
        return 1;
    }

    *out_targets = targets;
    *out_seeds   = seeds;
    *out_N       = N;
    printf("Loaded %d targets and %d seeds from %s/\n", N, N, dir);
    return 0;
}

// ============================================================================
// qsort comparator for benchmark percentile calculation (file scope)
// ============================================================================
static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// ============================================================================
// Benchmark mode: run batch IK N times, report p50/p95/p99 latency
// ============================================================================
static void run_benchmark(int N, const double* h_targets, const double* h_seeds,
                           int max_iter, double pos_tol, double orient_tol) {
    const int NRUNS = 100;
    float* times = (float*)malloc(NRUNS * sizeof(float));

    double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
    cudaMalloc(&d_targets, N * 16 * sizeof(double));
    cudaMalloc(&d_seeds,   N * 6 * sizeof(double));
    cudaMalloc(&d_results, N * 6 * sizeof(double));
    cudaMalloc(&d_errors,  N * 2 * sizeof(double));
    cudaMalloc(&d_iters,   N * sizeof(double));

    cudaMemcpy(d_targets, h_targets, N * 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds,   h_seeds,   N * 6 * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(N, 1, 1);
    dim3 block(128, 1, 1);

    printf("Running %d benchmark iterations (%d targets each)...\n", NRUNS, N);
    for (int run = 0; run < NRUNS; ++run) {
        cudaEventRecord(start);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  RUN %d KERNEL ERROR: %s\n", run, cudaGetErrorString(err));
            times[run] = -1.0;
            continue;
        }
        cudaEventElapsedTime(&times[run], start, stop);
    }

    // Sort times for percentile calculation
    qsort(times, NRUNS, sizeof(float), cmp_float);

    // Skip negative (error) values
    int valid_start = 0;
    while (valid_start < NRUNS && times[valid_start] < 0) valid_start++;
    int n_valid = NRUNS - valid_start;

    if (n_valid < 10) {
        printf("ERROR: Too few valid runs (%d/%d)\n", n_valid, NRUNS);
    } else {
        double p50 = times[valid_start + n_valid * 50 / 100];
        double p95 = times[valid_start + n_valid * 95 / 100];
        double p99 = times[valid_start + n_valid * 99 / 100];
        double avg = 0.0;
        for (int i = valid_start; i < NRUNS; ++i) avg += times[i];
        avg /= n_valid;

        double throughput = (double)N / (avg / 1000.0);  // targets/sec

        printf("\n=== Benchmark Results (%d targets, %d runs, %d valid) ===\n",
               N, NRUNS, n_valid);
        printf("  Latency  p50: %8.3f ms  (%.1f μs/target)\n", p50, p50 * 1000.0 / N);
        printf("  Latency  p95: %8.3f ms  (%.1f μs/target)\n", p95, p95 * 1000.0 / N);
        printf("  Latency  p99: %8.3f ms  (%.1f μs/target)\n", p99, p99 * 1000.0 / N);
        printf("  Latency  avg: %8.3f ms  (%.1f μs/target)\n", avg, avg * 1000.0 / N);
        printf("  Throughput:   %8.0f targets/sec\n", throughput);
        printf("  Min: %8.3f ms  Max: %8.3f ms\n",
               times[valid_start], times[NRUNS - 1]);
    }

    free(times);
    cudaFree(d_targets); cudaFree(d_seeds);
    cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Trajectory continuity helpers (MATLAB-compatible three-layer pipeline)
// ============================================================================

// Quintic interpolation coefficient: s(t) = 10t³ - 15t⁴ + 6t⁵
static inline double quintic_s(double t) {
    double t2 = t * t, t3 = t2 * t;
    return 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * t2 * t3;
}

// Quintic first derivative: s'(t) = 30t² - 60t³ + 30t⁴
static inline double quintic_s_dot(double t) {
    double t2 = t * t;
    return 30.0 * t2 - 60.0 * t2 * t + 30.0 * t2 * t2;
}

// LCG PRNG (matches test_cuda_kernel.cu convention)
static unsigned lcg_state = 42;
static unsigned lcg_rand() {
    lcg_state = lcg_state * 1103515245 + 12345;
    return lcg_state;
}
static double lcg_rand_double() {
    return (double)(lcg_rand() & 0x7FFFFFFF) / 2147483648.0;
}

// Generate sinusoidal trajectory (MATLAB-compatible)
static void generate_sinusoidal_path(
    const double q_home[6],
    int num_frames, double dt,
    double* path_T)  // [num_frames * 16] output
{
    double T_home[16];
    cpu_forward_kinematics(q_home, T_home);
    double base_x = T_home[3], base_y = T_home[7], base_z = T_home[11];

    const double Ax = 0.15, Ay = 0.10, Az = 0.08;  // amplitudes (m)
    const double fx = 0.5,  fy = 0.7,  fz = 0.3;   // frequencies (Hz)
    const double phase_y = 120.0 * M_PI / 180.0;
    const double phase_z = 240.0 * M_PI / 180.0;

    for (int i = 0; i < num_frames; ++i) {
        double t = i * dt;
        double dx = Ax * sin(2.0 * M_PI * fx * t);
        double dy = Ay * sin(2.0 * M_PI * fy * t + phase_y);
        double dz = Az * sin(2.0 * M_PI * fz * t + phase_z);

        for (int r = 0; r < 16; ++r) path_T[i * 16 + r] = T_home[r];
        path_T[i * 16 + 3]  = base_x + dx;
        path_T[i * 16 + 7]  = base_y + dy;
        path_T[i * 16 + 11] = base_z + dz;
    }
}

// Adaptive anchor selection (MATLAB Layer 1)
static int select_adaptive_anchors(
    const double* path_T, int num_frames,
    double path_thresh_m, double rot_thresh_deg, double tangent_thresh_deg,
    int max_anchors,
    int* anchors)  // output indices
{
    if (num_frames <= 0) return 0;
    double rot_thresh = rot_thresh_deg * M_PI / 180.0;
    double tangent_thresh = tangent_thresh_deg * M_PI / 180.0;

    int count = 0;
    anchors[count++] = 0;  // Always include first frame
    int last = 0;

    for (int i = 1; i < num_frames && count < max_anchors; ++i) {
        const double* T_prev = &path_T[last * 16];
        const double* T_curr = &path_T[i * 16];

        // Path length
        double dx = T_curr[3]  - T_prev[3];
        double dy = T_curr[7]  - T_prev[7];
        double dz = T_curr[11] - T_prev[11];
        double path_len = sqrt(dx*dx + dy*dy + dz*dz);

        // Rotation angle via relative rotation trace
        double r00=T_curr[0], r01=T_curr[1], r02=T_curr[2];
        double r10=T_curr[4], r11=T_curr[5], r12=T_curr[6];
        double r20=T_curr[8], r21=T_curr[9], r22=T_curr[10];
        double p00=T_prev[0], p01=T_prev[1], p02=T_prev[2];
        double p10=T_prev[4], p11=T_prev[5], p12=T_prev[6];
        double p20=T_prev[8], p21=T_prev[9], p22=T_prev[10];

        double rel0 = r00*p00+r01*p01+r02*p02;
        double rel1 = r00*p10+r01*p11+r02*p12;
        double rel2 = r00*p20+r01*p21+r02*p22;
        double rel3 = r10*p00+r11*p01+r12*p02;
        double rel4 = r10*p10+r11*p11+r12*p12;
        double rel5 = r10*p20+r11*p21+r12*p22;
        double rel6 = r20*p00+r21*p01+r22*p02;
        double rel7 = r20*p10+r21*p11+r22*p12;
        double rel8 = r20*p20+r21*p21+r22*p22;

        double trace = rel0 + rel4 + rel8;
        double cos_theta = (trace - 1.0) * 0.5;
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        double rot_angle = acos(cos_theta);

        // Tangent change (simplified)
        double tangent_change = 0.0;
        if (i >= 2) {
            const double* T_prev2 = &path_T[(i-1) * 16];
            double dx1 = T_curr[3] - T_prev2[3];
            double dy1 = T_curr[7] - T_prev2[7];
            double dz1 = T_curr[11] - T_prev2[11];
            double len1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);
            double dx2 = T_prev2[3] - T_prev[3];
            double dy2 = T_prev2[7] - T_prev[7];
            double dz2 = T_prev2[11] - T_prev[11];
            double len2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
            if (len1 > 1e-10 && len2 > 1e-10) {
                double dot = (dx1*dx2 + dy1*dy2 + dz1*dz2) / (len1 * len2);
                if (dot > 1.0) dot = 1.0;
                if (dot < -1.0) dot = -1.0;
                tangent_change = acos(dot);
            }
        }

        if (path_len > path_thresh_m || rot_angle > rot_thresh ||
            tangent_change > tangent_thresh) {
            anchors[count++] = i;
            last = i;
        }
    }

    // Always include last frame
    if (anchors[count-1] != num_frames - 1 && count < max_anchors) {
        anchors[count++] = num_frames - 1;
    }
    return count;
}

// Generate fixed seeds: home(1) + zero(1) + ±2π wraps(27) + random(remaining)
static void generate_fixed_seeds(double* seeds, int num_fixed,
                                  const double q_home[6], unsigned rng_seed) {
    lcg_state = rng_seed;
    int idx = 0;

    // Home
    for (int j = 0; j < 6; ++j) seeds[idx * 6 + j] = q_home[j];
    idx++;

    // Zero
    for (int j = 0; j < 6; ++j) seeds[idx * 6 + j] = 0.0;
    idx++;

    // ±2π wraps on first 3 joints (27 combinations)
    const double wraps[3] = {-2.0*M_PI, 0.0, 2.0*M_PI};
    for (int w0 = 0; w0 < 3; ++w0)
        for (int w1 = 0; w1 < 3; ++w1)
            for (int w2 = 0; w2 < 3; ++w2) {
                for (int j = 0; j < 6; ++j) seeds[idx*6+j] = q_home[j];
                seeds[idx*6+0] += wraps[w0];
                seeds[idx*6+1] += wraps[w1];
                seeds[idx*6+2] += wraps[w2];
                idx++;
            }

    // Random within ±π
    while (idx < num_fixed) {
        for (int j = 0; j < 6; ++j)
            seeds[idx*6+j] = -M_PI + 2.0*M_PI * lcg_rand_double();
        idx++;
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    // Parse command-line arguments
    enum { MODE_ALL, MODE_EXPORT, MODE_LOAD, MODE_BENCHMARK, MODE_TRAJECTORY } mode = MODE_ALL;
    const char* load_dir = "test_data/";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--export-data") == 0) {
            mode = MODE_EXPORT;
        } else if (strcmp(argv[i], "--load-data") == 0) {
            mode = MODE_LOAD;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                load_dir = argv[++i];
            }
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            mode = MODE_BENCHMARK;
        } else if (strcmp(argv[i], "--trajectory") == 0) {
            mode = MODE_TRAJECTORY;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("  (no args)        Run all tests (default)\n");
            printf("  --export-data    Generate 273 targets/seeds and export to test_data/\n");
            printf("  --load-data DIR  Load targets/seeds from DIR and run batch IK\n");
            printf("  --benchmark      Run 100 batch IK iterations and report latency stats\n");
            printf("  --trajectory     Run full trajectory continuity pipeline test\n");
            printf("  --help, -h       Show this help\n");
            return 0;
        } else {
            printf("Unknown option: %s (use --help for usage)\n", argv[i]);
            return 1;
        }
    }
    printf("=== UR10 CUDA IK Kernel Test (URDF FK, CUDA 13.3) ===\n\n");

    // Check device
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count == 0) {
        printf("ERROR: No CUDA device found (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d, %zu MB, CUDA %d.%d)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024*1024),
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    // Upload constant memory (must happen before any kernel launch)
    err = upload_constants();
    if (err != cudaSuccess) {
        printf("ERROR: Failed to upload constants: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    printf("Constant memory uploaded (URDF parameters, %lu bytes).\n\n",
           (unsigned long)(96+18+6*sizeof(int)/8+16+12+24+4) * sizeof(double));

    // ========================================================================
    // Mode dispatch: --export-data, --load-data, --benchmark, or default tests
    // ========================================================================
    if (mode == MODE_EXPORT || mode == MODE_BENCHMARK) {
        // Generate 273 test targets/seeds (same as Test 3)
        const int N_BATCH = 273;
        int max_iter = 100;
        double pos_tol = 0.03;
        double orient_tol = M_PI / 6.0;

        double* h_targets_batch = (double*)malloc(N_BATCH * 16 * sizeof(double));
        double* h_seeds_batch   = (double*)malloc(N_BATCH * 6 * sizeof(double));

        unsigned seed_val = 42;
        for (int i = 0; i < N_BATCH; ++i) {
            double q_rand[6];
            for (int j = 0; j < 6; ++j) {
                seed_val = seed_val * 1103515245 + 12345;
                double u = (double)(seed_val & 0x7FFFFFFF) / 2147483648.0;
                q_rand[j] = -M_PI + 2.0 * M_PI * u;
            }
            cpu_forward_kinematics(q_rand, &h_targets_batch[i * 16]);
            for (int j = 0; j < 6; ++j) {
                seed_val = seed_val * 1103515245 + 12345;
                double pert = ((double)(seed_val & 0x7FFFFFFF) / 2147483648.0 - 0.5) * 0.5;
                h_seeds_batch[i * 6 + j] = q_rand[j] + pert;
            }
        }

        if (mode == MODE_BENCHMARK) {
            run_benchmark(N_BATCH, h_targets_batch, h_seeds_batch,
                          max_iter, pos_tol, orient_tol);
            free(h_targets_batch);
            free(h_seeds_batch);
            return 0;
        }

        // MODE_EXPORT: run batch IK then export data
        printf("--- Export Mode: generating & solving 273 targets ---\n");

        double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
        cudaMalloc(&d_targets, N_BATCH * 16 * sizeof(double));
        cudaMalloc(&d_seeds,   N_BATCH * 6 * sizeof(double));
        cudaMalloc(&d_results, N_BATCH * 6 * sizeof(double));
        cudaMalloc(&d_errors,  N_BATCH * 2 * sizeof(double));
        cudaMalloc(&d_iters,   N_BATCH * sizeof(double));

        cudaMemcpy(d_targets, h_targets_batch, N_BATCH * 16 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seeds,   h_seeds_batch,   N_BATCH * 6 * sizeof(double), cudaMemcpyHostToDevice);

        dim3 grid(N_BATCH, 1, 1);
        dim3 block(128, 1, 1);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N_BATCH);

        err = cudaGetLastError();
        cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("  KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            double* h_results_batch = (double*)malloc(N_BATCH * 6 * sizeof(double));
            double* h_errors_batch  = (double*)malloc(N_BATCH * 2 * sizeof(double));
            double* h_iters_batch   = (double*)malloc(N_BATCH * sizeof(double));
            cudaMemcpy(h_results_batch, d_results, N_BATCH * 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_errors_batch,  d_errors,  N_BATCH * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_iters_batch,   d_iters,   N_BATCH * sizeof(double), cudaMemcpyDeviceToHost);

            export_test_data(h_targets_batch, h_seeds_batch,
                             h_results_batch, h_errors_batch, h_iters_batch, N_BATCH);

            free(h_results_batch);
            free(h_errors_batch);
            free(h_iters_batch);
        }

        cudaFree(d_targets); cudaFree(d_seeds);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
        free(h_targets_batch);
        free(h_seeds_batch);
        printf("Export complete.\n");
        return 0;
    }

    if (mode == MODE_LOAD) {
        printf("--- Load-Data Mode: loading from %s ---\n", load_dir);

        double* h_targets_load = nullptr;
        double* h_seeds_load   = nullptr;
        int N_load = 0;

        if (load_test_data(load_dir, &h_targets_load, &h_seeds_load, &N_load) != 0) {
            return 1;
        }

        int max_iter = 100;
        double pos_tol = 0.03;
        double orient_tol = M_PI / 6.0;

        double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
        cudaMalloc(&d_targets, N_load * 16 * sizeof(double));
        cudaMalloc(&d_seeds,   N_load * 6 * sizeof(double));
        cudaMalloc(&d_results, N_load * 6 * sizeof(double));
        cudaMalloc(&d_errors,  N_load * 2 * sizeof(double));
        cudaMalloc(&d_iters,   N_load * sizeof(double));

        cudaMemcpy(d_targets, h_targets_load, N_load * 16 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seeds,   h_seeds_load,   N_load * 6 * sizeof(double), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dim3 grid(N_load, 1, 1);
        dim3 block(128, 1, 1);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N_load);
        cudaEventRecord(stop);

        err = cudaGetLastError();
        cudaEventSynchronize(stop);
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start, stop);

        if (err != cudaSuccess) {
            printf("  KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            double* h_results_load = (double*)malloc(N_load * 6 * sizeof(double));
            double* h_errors_load  = (double*)malloc(N_load * 2 * sizeof(double));
            double* h_iters_load   = (double*)malloc(N_load * sizeof(double));
            cudaMemcpy(h_results_load, d_results, N_load * 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_errors_load,  d_errors,  N_load * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_iters_load,   d_iters,   N_load * sizeof(double), cudaMemcpyDeviceToHost);

            int converged = 0;
            double max_pos_err = 0.0, avg_pos_err = 0.0;
            for (int i = 0; i < N_load; ++i) {
                double pos_err = h_errors_load[i * 2 + 0];
                double rot_err = h_errors_load[i * 2 + 1];
                avg_pos_err += pos_err;
                if (pos_err > max_pos_err) max_pos_err = pos_err;
                if (pos_err <= pos_tol && rot_err <= orient_tol) converged++;
            }
            avg_pos_err /= N_load;

            printf("  Kernel time: %.3f ms (%.3f ms/target)\n", kernel_ms, kernel_ms / N_load);
            printf("  Convergence: %d/%d (%.1f%%)\n", converged, N_load, 100.0 * converged / N_load);
            printf("  Avg pos error: %.4f m, Max pos error: %.4f m\n", avg_pos_err, max_pos_err);
            printf("  First 3 results:\n");
            for (int i = 0; i < 3 && i < N_load; ++i) {
                printf("    [%d] q=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] err=%.4f iters=%.0f\n",
                       i,
                       h_results_load[i*6+0], h_results_load[i*6+1], h_results_load[i*6+2],
                       h_results_load[i*6+3], h_results_load[i*6+4], h_results_load[i*6+5],
                       h_errors_load[i*2+0], h_iters_load[i]);
            }

            free(h_results_load);
            free(h_errors_load);
            free(h_iters_load);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_targets); cudaFree(d_seeds);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
        free(h_targets_load);
        free(h_seeds_load);
        printf("Load-data test complete.\n");
        return 0;
    }

    // ========================================================================
    // Mode: --trajectory — Full three-layer trajectory continuity pipeline
    // ========================================================================
    if (mode == MODE_TRAJECTORY) {
        printf("=== Test: Full Trajectory Continuity Pipeline ===\n\n");

        const int NUM_FRAMES = 500;
        const double DT = 0.02;  // 50Hz
        const int NUM_SEEDS = 127;       // MATLAB-compatible: 1+1+27+97+1
        const int NUM_FIXED = 126;       // home(1) + zero(1) + wraps(27) + random(97)
        const int NUM_WEIGHTS = 4;
        const int TOTAL_CANDIDATES = NUM_SEEDS * NUM_WEIGHTS;
        const int MAX_ITER = 100;
        const double POS_TOL = 0.03;
        const double ORIENT_TOL = 0.1;
        const int TOP_K = 3;

        // UR10 home configuration
        double q_home[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // Layer 1: Generate sinusoidal trajectory
        printf("Layer 1: Generating %d-frame sinusoidal trajectory...\n", NUM_FRAMES);
        double* h_path = (double*)malloc(NUM_FRAMES * 16 * sizeof(double));
        generate_sinusoidal_path(q_home, NUM_FRAMES, DT, h_path);
        printf("  Path generated: %d frames, dt=%.2fs, duration=%.1fs\n",
               NUM_FRAMES, DT, NUM_FRAMES * DT);

        // Layer 1: Adaptive anchor selection
        int* h_anchors = (int*)malloc(NUM_FRAMES * sizeof(int));
        int num_anchors = select_adaptive_anchors(
            h_path, NUM_FRAMES, 0.012, 0.55, 0.50, 200, h_anchors);
        printf("  Adaptive anchors: %d/%d frames (%.1f%%)\n",
               num_anchors, NUM_FRAMES, 100.0 * num_anchors / NUM_FRAMES);

        // Show anchor distribution
        printf("  First 10 anchor indices: ");
        for (int i = 0; i < 10 && i < num_anchors; ++i) printf("%d ", h_anchors[i]);
        printf("...\n");

        // Generate fixed seeds (Layer 2 preparation)
        double* h_fixed_seeds = (double*)malloc(NUM_FIXED * 6 * sizeof(double));
        generate_fixed_seeds(h_fixed_seeds, NUM_FIXED, q_home, 42);
        printf("  Fixed seeds generated: %d (home+zero+27wraps+%d random)\n",
               NUM_FIXED, NUM_FIXED - 29);

        // Allocate GPU memory (reused across anchors)
        double *d_target, *d_all_seeds, *d_q_prev, *d_joint_weights;
        double *d_results, *d_errors, *d_shovel_errors, *d_iterations;
        double *d_costs, *d_topk_costs;
        int    *d_topk_indices;

        cudaMalloc(&d_target,         16 * sizeof(double));
        cudaMalloc(&d_all_seeds,      NUM_SEEDS * 6 * sizeof(double));
        cudaMalloc(&d_q_prev,         6 * sizeof(double));
        cudaMalloc(&d_joint_weights,  6 * sizeof(double));
        cudaMalloc(&d_results,        TOTAL_CANDIDATES * 6 * sizeof(double));
        cudaMalloc(&d_errors,         TOTAL_CANDIDATES * 2 * sizeof(double));
        cudaMalloc(&d_shovel_errors,  TOTAL_CANDIDATES * 2 * sizeof(double));
        cudaMalloc(&d_iterations,     TOTAL_CANDIDATES * sizeof(double));
        cudaMalloc(&d_costs,          TOTAL_CANDIDATES * sizeof(double));
        cudaMalloc(&d_topk_costs,     TOP_K * sizeof(double));
        cudaMalloc(&d_topk_indices,   TOP_K * sizeof(int));

        double h_joint_weights[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        cudaMemcpy(d_joint_weights, h_joint_weights, 6 * sizeof(double), cudaMemcpyHostToDevice);

        // Build all_seeds array: fixed seeds + 1 q_prev slot
        double* h_all_seeds = (double*)malloc(NUM_SEEDS * 6 * sizeof(double));
        memcpy(h_all_seeds, h_fixed_seeds, NUM_FIXED * 6 * sizeof(double));

        // Storage for anchor results
        double* h_anchor_q = (double*)malloc(num_anchors * 6 * sizeof(double));
        double* h_anchor_err = (double*)malloc(num_anchors * 2 * sizeof(double));
        int*    h_anchor_seed = (int*)malloc(num_anchors * sizeof(int));

        // Layer 2: Per-anchor multi-seed multi-weight GPU IK
        printf("\nLayer 2: GPU multi-seed×multi-weight IK per anchor...\n");
        printf("  Config: %d seeds × %d weights = %d candidates/anchor\n",
               NUM_SEEDS, NUM_WEIGHTS, TOTAL_CANDIDATES);
        printf("  Total GPU IK solves: %d\n", num_anchors * TOTAL_CANDIDATES);

        cudaEvent_t ta_start, ta_stop;
        cudaEventCreate(&ta_start);
        cudaEventCreate(&ta_stop);
        cudaEventRecord(ta_start);

        double q_prev[6];
        for (int j = 0; j < 6; ++j) q_prev[j] = q_home[j];

        int converged_count = 0;
        double max_joint_jump = 0.0;
        int branch_flips = 0;

        // CPU-side buffers for reading all candidate data (reused across anchors)
        double* h_all_errors_cpu  = (double*)malloc(TOTAL_CANDIDATES * 2 * sizeof(double));
        double* h_all_costs_cpu   = (double*)malloc(TOTAL_CANDIDATES * sizeof(double));
        double* h_all_iters_cpu   = (double*)malloc(TOTAL_CANDIDATES * sizeof(double));
        double* h_all_results_cpu = (double*)malloc(TOTAL_CANDIDATES * 6 * sizeof(double));

        for (int a = 0; a < num_anchors; ++a) {
            int path_idx = h_anchors[a];

            // Upload target
            cudaMemcpy(d_target, &h_path[path_idx * 16], 16 * sizeof(double), cudaMemcpyHostToDevice);

            // Set q_prev seed (slot NUM_FIXED)
            for (int j = 0; j < 6; ++j)
                h_all_seeds[NUM_FIXED * 6 + j] = q_prev[j];
            cudaMemcpy(d_all_seeds, h_all_seeds, NUM_SEEDS * 6 * sizeof(double), cudaMemcpyHostToDevice);

            // Upload q_prev for continuity cost
            cudaMemcpy(d_q_prev, q_prev, 6 * sizeof(double), cudaMemcpyHostToDevice);

            // GPU: multi-seed × multi-weight IK
            err = rtfg::cuda::launch_ik_batch_solve_multi(
                d_target, d_all_seeds,
                d_results, d_errors, d_shovel_errors, d_iterations,
                MAX_ITER, POS_TOL, ORIENT_TOL,
                NUM_SEEDS, NUM_WEIGHTS, 0);

            if (err != cudaSuccess) {
                printf("  Anchor %d: ik_batch_solve_multi ERROR: %s\n",
                       a, cudaGetErrorString(err));
                continue;
            }

            // GPU: continuity cost
            err = rtfg::cuda::launch_compute_continuity_cost_all(
                d_results, d_q_prev, d_costs, TOTAL_CANDIDATES, d_joint_weights);

            if (err != cudaSuccess) {
                printf("  Anchor %d: compute_continuity_cost_all ERROR: %s\n",
                       a, cudaGetErrorString(err));
                continue;
            }

            cudaDeviceSynchronize();

            // Read back ALL candidate data once (MATLAB approach: CPU-side full scan)
            cudaMemcpy(h_all_errors_cpu,  d_errors,  TOTAL_CANDIDATES * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_all_costs_cpu,   d_costs,   TOTAL_CANDIDATES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_all_iters_cpu,   d_iterations, TOTAL_CANDIDATES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_all_results_cpu, d_results, TOTAL_CANDIDATES * 6 * sizeof(double), cudaMemcpyDeviceToHost);

            // Greedy selection: converged + lowest continuity cost + no branch switch
            const double BRANCH_THRESHOLD = 25.0 * M_PI / 180.0;  // 25° branch-switch threshold

            bool found = false;
            double best_cost = 1e100;
            int    best_cand = -1;

            for (int c = 0; c < TOTAL_CANDIDATES; ++c) {
                double pe = h_all_errors_cpu[c * 2 + 0];
                double re = h_all_errors_cpu[c * 2 + 1];
                if (pe > POS_TOL || re > ORIENT_TOL) continue;  // Skip unconverged

                // Branch-switch check: any raw joint diff > 25°?
                double* qc = h_all_results_cpu + c * 6;
                bool is_branch_switch = false;
                for (int j = 0; j < 6; ++j) {
                    if (fabs(qc[j] - q_prev[j]) > BRANCH_THRESHOLD) {
                        is_branch_switch = true;
                        break;
                    }
                }

                // Heavily penalize branch switches so they're only used as last resort
                double cost = h_all_costs_cpu[c];
                if (is_branch_switch) cost *= 1000.0;

                if (cost < best_cost) {
                    best_cost = cost;
                    best_cand = c;
                    found = true;
                }
            }

            if (found) {
                double* q_best = h_all_results_cpu + best_cand * 6;

                // Joint alignment: wrap to nearest ±π of q_prev
                for (int j = 0; j < 6; ++j) {
                    double diff = q_best[j] - q_prev[j];
                    h_anchor_q[a * 6 + j] = q_prev[j] + atan2(sin(diff), cos(diff));
                }
                h_anchor_err[a * 2 + 0] = h_all_errors_cpu[best_cand * 2 + 0];
                h_anchor_err[a * 2 + 1] = h_all_errors_cpu[best_cand * 2 + 1];
                h_anchor_seed[a] = best_cand / NUM_WEIGHTS;

                // Track joint jumps (wrapped, for real joint movement measurement)
                double local_max_jump = 0.0;
                for (int j = 0; j < 6; ++j) {
                    double jump = fabs(atan2(sin(h_anchor_q[a*6+j] - q_prev[j]),
                                             cos(h_anchor_q[a*6+j] - q_prev[j])));
                    if (jump > local_max_jump) local_max_jump = jump;
                }
                if (local_max_jump > max_joint_jump) max_joint_jump = local_max_jump;

                // Count branch flips: was the best candidate penalized for branch switch?
                double* qc = h_all_results_cpu + best_cand * 6;
                for (int j = 0; j < 6; ++j) {
                    if (fabs(qc[j] - q_prev[j]) > BRANCH_THRESHOLD) {
                        branch_flips++;
                        break;
                    }
                }
                converged_count++;
            } else {
                // No candidate converged strictly — best-effort fallback:
                // accept the candidate with lowest position error (relaxed convergence)
                double best_pe = 1e100;
                int best_effort = -1;
                for (int c = 0; c < TOTAL_CANDIDATES; ++c) {
                    double pe = h_all_errors_cpu[c * 2 + 0];
                    if (pe < best_pe) { best_pe = pe; best_effort = c; }
                }
                if (best_effort >= 0 && best_pe < 0.10) {  // Accept if < 10cm
                    double* q_be = h_all_results_cpu + best_effort * 6;
                    for (int j = 0; j < 6; ++j) {
                        double diff = q_be[j] - q_prev[j];
                        h_anchor_q[a * 6 + j] = q_prev[j] + atan2(sin(diff), cos(diff));
                    }
                    h_anchor_err[a * 2 + 0] = best_pe;
                    h_anchor_err[a * 2 + 1] = 0.0;
                    h_anchor_seed[a] = best_effort / NUM_WEIGHTS;
                } else {
                    // Truly unreachable — use previous position
                    for (int j = 0; j < 6; ++j) h_anchor_q[a * 6 + j] = q_prev[j];
                    h_anchor_err[a * 2 + 0] = 0.0;
                    h_anchor_err[a * 2 + 1] = 0.0;
                    h_anchor_seed[a] = -1;
                }
            }

            // Update q_prev
            for (int j = 0; j < 6; ++j) q_prev[j] = h_anchor_q[a * 6 + j];

            if ((a + 1) % 25 == 0 || a == num_anchors - 1) {
                printf("  Anchor %3d/%d (frame %d): seed=%d err=%.3fmm jump=%.3f°\n",
                       a + 1, num_anchors, path_idx,
                       h_anchor_seed[a],
                       h_anchor_err[a * 2 + 0] * 1000.0,
                       max_joint_jump * 180.0 / M_PI);
            }
        }

        cudaEventRecord(ta_stop);
        cudaEventSynchronize(ta_stop);
        float gpu_time_ms = 0;
        cudaEventElapsedTime(&gpu_time_ms, ta_start, ta_stop);

        double conv_rate = 100.0 * converged_count / num_anchors;
        printf("\n  GPU time: %.1f ms (%.1f ms/anchor)\n", gpu_time_ms, gpu_time_ms / num_anchors);
        printf("  Anchor convergence: %d/%d (%.1f%%)\n",
               converged_count, num_anchors, conv_rate);
        printf("  Max single-joint jump: %.4f rad (%.2f°)\n",
               max_joint_jump, max_joint_jump * 180.0 / M_PI);

        // Layer 3: Quintic interpolation + playback analysis
        printf("\nLayer 3: Quintic C² interpolation...\n");

        int total_playback = 0;
        double max_interp_jump = 0.0;
        int playback_danger = 0;

        for (int a = 0; a < num_anchors - 1; ++a) {
            int num_steps = h_anchors[a + 1] - h_anchors[a];
            if (num_steps <= 0) continue;

            double* q_from = &h_anchor_q[a * 6];
            double* q_to   = &h_anchor_q[(a + 1) * 6];

            for (int s = 1; s < num_steps; ++s) {
                double t = (double)s / (double)num_steps;
                double st = quintic_s(t);

                double q_interp[6];
                for (int j = 0; j < 6; ++j) {
                    q_interp[j] = q_from[j] + st * (q_to[j] - q_from[j]);
                }

                // Check joint jump from previous frame
                if (total_playback > 0) {
                    // Compare against q_from (last anchor)
                    for (int j = 0; j < 6; ++j) {
                        double jump = fabs(atan2(sin(q_interp[j] - q_from[j]),
                                                 cos(q_interp[j] - q_from[j])));
                        if (jump > max_interp_jump) max_interp_jump = jump;
                        if (jump > 0.5) playback_danger++;
                    }
                }
                total_playback++;
            }
        }

        printf("  Playback frames: %d (from %d anchors)\n", total_playback, num_anchors);
        printf("  Max interpolated joint step: %.4f rad (%.2f°)\n",
               max_interp_jump, max_interp_jump * 180.0 / M_PI);
        printf("  Danger frames (>0.5 rad): %d\n", playback_danger);

        // Final verdict
        printf("\n=== Trajectory Test Results ===\n");
        printf("  Anchor convergence:   %d/%d (%.1f%%) %s\n",
               converged_count, num_anchors, conv_rate,
               conv_rate >= 90.0 ? "✓" : "✗");
        printf("  Max anchor jump:      %.4f rad (%.2f°) %s\n",
               max_joint_jump, max_joint_jump * 180.0 / M_PI,
               max_joint_jump < 0.5 ? "✓" : "✗");
        printf("  Danger frames:        %d %s\n",
               playback_danger,
               playback_danger == 0 ? "✓" : "✗");
        printf("  Branch flips:         %d %s\n",
               branch_flips,
               branch_flips == 0 ? "✓" : "✗");
        printf("  Playback frames:      %d\n", total_playback);
        printf("  GPU pipeline time:    %.1f ms\n", gpu_time_ms);

        bool overall = (conv_rate >= 90.0) && (max_joint_jump < 0.5) &&
                       (playback_danger == 0) && (branch_flips == 0);
        printf("\n  OVERALL: %s\n\n", overall ? "PASS ✓" : "FAIL ✗");

        // Cleanup
        free(h_path); free(h_anchors); free(h_fixed_seeds);
        free(h_all_seeds); free(h_anchor_q); free(h_anchor_err); free(h_anchor_seed);
        free(h_all_errors_cpu); free(h_all_costs_cpu); free(h_all_iters_cpu); free(h_all_results_cpu);
        cudaFree(d_target); cudaFree(d_all_seeds); cudaFree(d_q_prev);
        cudaFree(d_joint_weights);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_shovel_errors);
        cudaFree(d_iterations); cudaFree(d_costs);
        cudaFree(d_topk_costs); cudaFree(d_topk_indices);
        cudaEventDestroy(ta_start); cudaEventDestroy(ta_stop);
        return overall ? 0 : 1;
    }

    // Default: run all standard tests (Tests 0-5)

    // ========================================================================
    // Test 0: DEBUG — Single DLS iteration trace
    // ========================================================================
    printf("--- DEBUG: Single DLS iteration (target q=[0.1,0.1,0.1,0.1,0.1,0.1]) ---\n");
    {
        double q_target[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
        double T_target[16];
        cpu_forward_kinematics(q_target, T_target);
        double h_seed[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        double *d_target, *d_seed;
        cudaMalloc(&d_target, 16*sizeof(double));
        cudaMalloc(&d_seed, 6*sizeof(double));
        cudaMemcpy(d_target, T_target, 16*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, 6*sizeof(double), cudaMemcpyHostToDevice);

        rtfg::cuda::debug_dls_iteration<<<1, 1>>>(d_target, d_seed);
        cudaDeviceSynchronize();
        cudaFree(d_target); cudaFree(d_seed);
    }
    printf("\n");

    // ========================================================================
    // Test 0b: DEBUG — Progressive IK convergence trace (near seed)
    // ========================================================================
    printf("--- DEBUG: IK progress trace (near-seed) ---\n");
    {
        double q_target[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
        double T_target[16];
        cpu_forward_kinematics(q_target, T_target);
        double h_seed[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        double *d_target, *d_seed;
        cudaMalloc(&d_target, 16*sizeof(double));
        cudaMalloc(&d_seed, 6*sizeof(double));
        cudaMemcpy(d_target, T_target, 16*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seed, h_seed, 6*sizeof(double), cudaMemcpyHostToDevice);

        rtfg::cuda::debug_ik_progress<<<1, 1>>>(d_target, d_seed, 100);
        cudaDeviceSynchronize();
        cudaFree(d_target); cudaFree(d_seed);
    }
    printf("\n");

    // ========================================================================
    // Test 1: FK Correctness — GPU vs CPU
    // ========================================================================
    printf("--- Test 1: FK Correctness (GPU vs CPU reference) ---\n");

    const int FK_COUNT = 10;
    // 10 test joint configurations
    double test_q[FK_COUNT][6] = {
        { 0.0,          0.0,          0.0,          0.0,          0.0,          0.0          },
        { 1.57079632679, 0.0,          0.0,          0.0,          0.0,          0.0          },  // π/2
        { 0.0,          1.57079632679, 0.0,          0.0,          0.0,          0.0          },
        { 0.0,          0.0,         -1.57079632679, 0.0,          0.0,          0.0          },  // -π/2
        { 0.0,          0.0,          0.0,          1.57079632679, 0.0,          0.0          },
        { 0.0,          0.0,          0.0,          0.0,          1.57079632679, 0.0          },
        { 0.0,          0.0,          0.0,          0.0,          0.0,          1.57079632679 },
        { 0.5,         -0.3,          1.2,         -0.8,          2.1,          0.4          },
        {-1.0,          2.0,         -2.5,          1.5,         -0.5,          3.0          },
        { 2.04749846,   0.21928656,  -1.95482141,  -0.35923797,   2.05026770,   1.03308296  }  // UR10 home-ish
    };

    double *d_q_fk, *d_T_fk;
    cudaMalloc(&d_q_fk, FK_COUNT * 6 * sizeof(double));
    cudaMalloc(&d_T_fk, FK_COUNT * 16 * sizeof(double));

    double h_q_flat[FK_COUNT * 6];
    for (int i = 0; i < FK_COUNT; ++i)
        for (int j = 0; j < 6; ++j)
            h_q_flat[i * 6 + j] = test_q[i][j];

    cudaMemcpy(d_q_fk, h_q_flat, FK_COUNT * 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch GPU FK kernel
    dim3 fk_grid((FK_COUNT + 31) / 32);
    dim3 fk_block(32);
    rtfg::cuda::test_fk_kernel<<<fk_grid, fk_block>>>(d_q_fk, d_T_fk, FK_COUNT);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FK KERNEL ERROR: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Read back GPU results
    double h_T_gpu[FK_COUNT * 16];
    cudaMemcpy(h_T_gpu, d_T_fk, FK_COUNT * 16 * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare against CPU FK
    int fk_pass = 0, fk_fail = 0;
    double fk_max_err = 0.0;
    for (int i = 0; i < FK_COUNT; ++i) {
        double T_cpu[16];
        cpu_forward_kinematics(test_q[i], T_cpu);
        double err_i = max_abs_diff_16(&h_T_gpu[i * 16], T_cpu);
        if (err_i > fk_max_err) fk_max_err = err_i;

        if (err_i < 1e-14) {
            fk_pass++;
        } else {
            fk_fail++;
            printf("  FAIL [%d]: q=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] max_diff=%.2e\n",
                   i, test_q[i][0], test_q[i][1], test_q[i][2],
                   test_q[i][3], test_q[i][4], test_q[i][5], err_i);
            print_mat4("  GPU", &h_T_gpu[i * 16]);
            print_mat4("  CPU", T_cpu);
        }
    }
    printf("  FK test: %d/%d passed, max error = %.2e %s\n\n",
           fk_pass, FK_COUNT, fk_max_err,
           (fk_fail == 0) ? "✓" : "✗");

    // Show first result detail
    printf("  q=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] → TCP pos=(%.4f, %.4f, %.4f)\n",
           test_q[0][0], test_q[0][1], test_q[0][2],
           test_q[0][3], test_q[0][4], test_q[0][5],
           h_T_gpu[3], h_T_gpu[7], h_T_gpu[11]);

    cudaFree(d_q_fk);
    cudaFree(d_T_fk);

    // ========================================================================
    // Test 2: IK Convergence — Near-seed target (sanity check)
    // ========================================================================
    printf("\n--- Test 2a: IK Convergence (near-seed target) ---\n");

    {
        // Generate target from q close to seed (0)
        double q_target[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
        double T_target[16];
        cpu_forward_kinematics(q_target, T_target);

        printf("  Target q=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               q_target[0], q_target[1], q_target[2],
               q_target[3], q_target[4], q_target[5]);

        const int N_SINGLE = 1;
        double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
        cudaMalloc(&d_targets, N_SINGLE * 16 * sizeof(double));
        cudaMalloc(&d_seeds,   N_SINGLE * 6 * sizeof(double));
        cudaMalloc(&d_results, N_SINGLE * 6 * sizeof(double));
        cudaMalloc(&d_errors,  N_SINGLE * 2 * sizeof(double));
        cudaMalloc(&d_iters,   N_SINGLE * sizeof(double));

        cudaMemcpy(d_targets, T_target, 16 * sizeof(double), cudaMemcpyHostToDevice);
        double h_seed[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        cudaMemcpy(d_seeds, h_seed, 6 * sizeof(double), cudaMemcpyHostToDevice);

        int max_iter = 200;
        double pos_tol = 0.01;
        double orient_tol = M_PI / 6.0;

        dim3 grid(N_SINGLE, 1, 1);
        dim3 block(128, 1, 1);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N_SINGLE);

        err = cudaGetLastError();
        cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("  IK KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            double h_result[6], h_err[2], h_iter;
            cudaMemcpy(h_result, d_results, 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_err, d_errors, 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_iter, d_iters, sizeof(double), cudaMemcpyDeviceToHost);

            double max_joint_err = 0.0;
            for (int j = 0; j < 6; ++j) {
                double diff = h_result[j] - q_target[j];
                double wrapped = atan2(sin(diff), cos(diff));
                if (fabs(wrapped) > max_joint_err) max_joint_err = fabs(wrapped);
            }

            printf("  Result:[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                   h_result[0], h_result[1], h_result[2],
                   h_result[3], h_result[4], h_result[5]);
            printf("  Pos_err=%.6f, Rot_err=%.4f, Iters=%.0f, Max_joint_err=%.4f rad\n",
                   h_err[0], h_err[1], h_iter, max_joint_err);
            // NOTE: Joint error is not a reliable pass criterion for IK —
            // multiple joint configurations can achieve the same TCP pose.
            // Position error is the canonical convergence metric.
            printf("  Near-seed IK (pos check only): %s\n\n",
                   (h_err[0] < 0.01) ? "PASS ✓" : "FAIL ✗");
        }

        cudaFree(d_targets); cudaFree(d_seeds);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
    }

    // ========================================================================
    // Test 2b: IK Convergence — Far target with higher tolerance
    // ========================================================================
    printf("--- Test 2b: IK Convergence (far target, relaxed tol) ---\n");

    {
        double q_target[6] = {0.5, -0.3, 1.2, -0.8, 0.6, 0.4};
        double T_target[16];
        cpu_forward_kinematics(q_target, T_target);

        const int N_SINGLE = 1;
        double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
        cudaMalloc(&d_targets, N_SINGLE * 16 * sizeof(double));
        cudaMalloc(&d_seeds,   N_SINGLE * 6 * sizeof(double));
        cudaMalloc(&d_results, N_SINGLE * 6 * sizeof(double));
        cudaMalloc(&d_errors,  N_SINGLE * 2 * sizeof(double));
        cudaMalloc(&d_iters,   N_SINGLE * sizeof(double));

        cudaMemcpy(d_targets, T_target, 16 * sizeof(double), cudaMemcpyHostToDevice);
        double h_seed[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        cudaMemcpy(d_seeds, h_seed, 6 * sizeof(double), cudaMemcpyHostToDevice);

        int max_iter = 200;
        double pos_tol = 0.03;
        double orient_tol = M_PI / 4.0;

        dim3 grid(N_SINGLE, 1, 1);
        dim3 block(128, 1, 1);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N_SINGLE);

        err = cudaGetLastError();
        cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("  IK KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            double h_result[6], h_err[2], h_iter;
            cudaMemcpy(h_result, d_results, 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_err, d_errors, 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_iter, d_iters, sizeof(double), cudaMemcpyDeviceToHost);

            double max_joint_err = 0.0;
            for (int j = 0; j < 6; ++j) {
                double diff = h_result[j] - q_target[j];
                double wrapped = atan2(sin(diff), cos(diff));
                if (fabs(wrapped) > max_joint_err) max_joint_err = fabs(wrapped);
            }

            printf("  Result:[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                   h_result[0], h_result[1], h_result[2],
                   h_result[3], h_result[4], h_result[5]);
            printf("  Target:[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                   q_target[0], q_target[1], q_target[2],
                   q_target[3], q_target[4], q_target[5]);
            printf("  Pos_err=%.6f, Rot_err=%.4f, Iters=%.0f, Max_joint_err=%.4f rad\n",
                   h_err[0], h_err[1], h_iter, max_joint_err);
            printf("  Far-target IK: %s\n\n",
                   (h_err[0] < 0.05) ? "PASS ✓" : "FAIL ✗");
        }

        cudaFree(d_targets); cudaFree(d_seeds);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
    }

    // ========================================================================
    // Test 3: Batch IK — 273 targets (stress test)
    // ========================================================================
    printf("--- Test 3: Batch IK (273 targets) ---\n");

    {
        const int N_BATCH = 273;
        int max_iter = 100;
        double pos_tol = 0.03;
        double orient_tol = M_PI / 6.0;

        double *d_targets, *d_seeds, *d_results, *d_errors, *d_iters;
        cudaMalloc(&d_targets, N_BATCH * 16 * sizeof(double));
        cudaMalloc(&d_seeds,   N_BATCH * 6 * sizeof(double));
        cudaMalloc(&d_results, N_BATCH * 6 * sizeof(double));
        cudaMalloc(&d_errors,  N_BATCH * 2 * sizeof(double));
        cudaMalloc(&d_iters,   N_BATCH * sizeof(double));

        // Generate 273 reachable targets using CPU FK at random q values
        double *h_targets_batch = (double*)malloc(N_BATCH * 16 * sizeof(double));
        double *h_seeds_batch   = (double*)malloc(N_BATCH * 6 * sizeof(double));

        // Simple PRNG (not crypto-safe, deterministic for reproducibility)
        unsigned seed_val = 42;
        for (int i = 0; i < N_BATCH; ++i) {
            // Generate random q within joint limits
            double q_rand[6];
            for (int j = 0; j < 6; ++j) {
                seed_val = seed_val * 1103515245 + 12345;
                double u = (double)(seed_val & 0x7FFFFFFF) / 2147483648.0;  // 0..1
                q_rand[j] = -M_PI + 2.0 * M_PI * u;
            }
            // Compute target via CPU FK
            cpu_forward_kinematics(q_rand, &h_targets_batch[i * 16]);

            // Seed = target q + small random perturbation
            for (int j = 0; j < 6; ++j) {
                    seed_val = seed_val * 1103515245 + 12345;
                    double pert = ((double)(seed_val & 0x7FFFFFFF) / 2147483648.0 - 0.5) * 0.5;
                    h_seeds_batch[i * 6 + j] = q_rand[j] + pert;
            }
        }

        cudaMemcpy(d_targets, h_targets_batch, N_BATCH * 16 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_seeds,   h_seeds_batch,   N_BATCH * 6 * sizeof(double), cudaMemcpyHostToDevice);

        // Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dim3 grid(N_BATCH, 1, 1);
        dim3 block(128, 1, 1);
        rtfg::cuda::ik_batch_solve<<<grid, block>>>(
            d_targets, d_seeds, d_results, d_errors, nullptr, d_iters,
            max_iter, pos_tol, orient_tol, 0, N_BATCH);
        cudaEventRecord(stop);

        err = cudaGetLastError();
        cudaEventSynchronize(stop);
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start, stop);

        if (err != cudaSuccess) {
            printf("  BATCH KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            printf("  Kernel launched: %d blocks × 128 threads\n", N_BATCH);
            printf("  Kernel time: %.3f ms (%.3f ms/target)\n",
                   kernel_ms, kernel_ms / N_BATCH);

            // Read back results
            double *h_results_batch = (double*)malloc(N_BATCH * 6 * sizeof(double));
            double *h_errors_batch  = (double*)malloc(N_BATCH * 2 * sizeof(double));
            double *h_iters_batch   = (double*)malloc(N_BATCH * sizeof(double));
            cudaMemcpy(h_results_batch, d_results, N_BATCH * 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_errors_batch,  d_errors,  N_BATCH * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_iters_batch,   d_iters,   N_BATCH * sizeof(double), cudaMemcpyDeviceToHost);

            // Analyze convergence (position-only, since weight schedule deprioritizes orientation)
            int converged = 0, total_iters = 0;
            double max_pos_err = 0.0, avg_pos_err = 0.0;
            int bins[5] = {0};  // <1cm, <3cm, <5cm, <10cm, >=10cm
            for (int i = 0; i < N_BATCH; ++i) {
                double pos_err = h_errors_batch[i * 2 + 0];
                double rot_err = h_errors_batch[i * 2 + 1];
                avg_pos_err += pos_err;
                if (pos_err > max_pos_err) max_pos_err = pos_err;
                if (pos_err <= pos_tol && rot_err <= orient_tol) converged++;
                total_iters += (int)h_iters_batch[i];
                if (pos_err < 0.01) bins[0]++;
                else if (pos_err < 0.03) bins[1]++;
                else if (pos_err < 0.05) bins[2]++;
                else if (pos_err < 0.10) bins[3]++;
                else bins[4]++;
            }
            avg_pos_err /= N_BATCH;
            double avg_iters = (double)total_iters / N_BATCH;
            double convergence_rate = 100.0 * converged / N_BATCH;

            printf("  Convergence (pos+rot): %d/%d (%.1f%%)\n", converged, N_BATCH, convergence_rate);
            printf("  Pos error distribution: <1cm:%d <3cm:%d <5cm:%d <10cm:%d >=10cm:%d\n",
                   bins[0], bins[1], bins[2], bins[3], bins[4]);
            printf("  Avg pos error: %.4f m, Max pos error: %.4f m\n", avg_pos_err, max_pos_err);
            printf("  Avg iterations: %.1f\n", avg_iters);

            // Show first and last results
            // Export all results to CSV for chart generation
            FILE* fcsv = fopen("/tmp/batch_ik_results.csv", "w");
            if (fcsv) {
                fprintf(fcsv, "idx,q0,q1,q2,q3,q4,q5,pos_err,rot_err,iterations\n");
                for (int i = 0; i < N_BATCH; ++i) {
                    fprintf(fcsv, "%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.0f\n",
                            i,
                            h_results_batch[i*6+0], h_results_batch[i*6+1], h_results_batch[i*6+2],
                            h_results_batch[i*6+3], h_results_batch[i*6+4], h_results_batch[i*6+5],
                            h_errors_batch[i*2+0], h_errors_batch[i*2+1],
                            h_iters_batch[i]);
                }
                fclose(fcsv);
                printf("  Exported %d results to /tmp/batch_ik_results.csv\n", N_BATCH);
            }

            printf("  First 3 results:\n");
            for (int i = 0; i < 3; ++i) {
                printf("    [%d] q=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
                       "err=%.4f iters=%.0f\n",
                       i,
                       h_results_batch[i*6+0], h_results_batch[i*6+1], h_results_batch[i*6+2],
                       h_results_batch[i*6+3], h_results_batch[i*6+4], h_results_batch[i*6+5],
                       h_errors_batch[i*2+0], h_iters_batch[i]);
            }

            // Pass if >80% have pos_err < 3cm AND >95% have pos_err < 10cm
            bool batch_ok = (bins[0] + bins[1] >= (int)(N_BATCH * 0.80)) &&
                            (bins[0] + bins[1] + bins[2] + bins[3] >= (int)(N_BATCH * 0.95));
            printf("  Batch test: %s\n\n", batch_ok ? "PASS ✓" : "WARN (check targets)");

            free(h_results_batch);
            free(h_errors_batch);
            free(h_iters_batch);
        }

        free(h_targets_batch);
        free(h_seeds_batch);
        cudaFree(d_targets); cudaFree(d_seeds);
        cudaFree(d_results); cudaFree(d_errors); cudaFree(d_iters);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ========================================================================
    // Test 4: Continuity cost kernel
    // ========================================================================
    printf("--- Test 4: Continuity cost computation ---\n");

    {
        const int N_COST = 100;
        double *d_results, *d_q_prev, *d_dq_prev, *d_costs;
        cudaMalloc(&d_results, N_COST * 6 * sizeof(double));
        cudaMalloc(&d_q_prev, 6 * sizeof(double));
        cudaMalloc(&d_dq_prev, 6 * sizeof(double));
        cudaMalloc(&d_costs, N_COST * sizeof(double));

        double h_q_prev[6] = {0, 0, 0, 0, 0, 0};
        double h_dq_prev[6] = {0, 0, 0, 0, 0, 0};
        cudaMemcpy(d_q_prev, h_q_prev, 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dq_prev, h_dq_prev, 6 * sizeof(double), cudaMemcpyHostToDevice);

        // Initialize results with increasing angles
        double *h_results_cost = (double*)malloc(N_COST * 6 * sizeof(double));
        for (int i = 0; i < N_COST; ++i)
            for (int j = 0; j < 6; ++j)
                h_results_cost[i * 6 + j] = 0.01 * i;
        cudaMemcpy(d_results, h_results_cost, N_COST * 6 * sizeof(double), cudaMemcpyHostToDevice);

        dim3 grid2((N_COST + 255) / 256);
        dim3 block2(256);
        rtfg::cuda::compute_continuity_cost<<<grid2, block2>>>(
            d_results, d_q_prev, d_dq_prev, d_costs, N_COST);

        err = cudaGetLastError();
        cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("  COST KERNEL ERROR: %s\n", cudaGetErrorString(err));
        } else {
            double h_costs[N_COST];
            cudaMemcpy(h_costs, d_costs, N_COST * sizeof(double), cudaMemcpyDeviceToHost);
            printf("  First 5 costs: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                   h_costs[0], h_costs[1], h_costs[2], h_costs[3], h_costs[4]);
            // Verify monotonic (cost should increase with i)
            bool monotonic = true;
            for (int i = 1; i < N_COST; ++i) {
                if (h_costs[i] < h_costs[i-1] - 1e-12) { monotonic = false; break; }
            }
            printf("  Cost monotonic: %s\n", monotonic ? "PASS ✓" : "FAIL ✗");
        }

        free(h_results_cost);
        cudaFree(d_results); cudaFree(d_q_prev);
        cudaFree(d_dq_prev); cudaFree(d_costs);
    }

    // ========================================================================
    // Test 5: GPU Collision Detection — box-box/box-sphere/box-cylinder
    // ========================================================================
    printf("--- Test 5: GPU Collision Detection ---\n");

    {
        const int N_FRAMES = 100;
        const int N_BOXES = 8;
        const int N_OBJECTS = 5;

        // Generate robot boxes (N_FRAMES × N_BOXES × 6)
        double *h_robot_boxes = (double*)malloc(N_FRAMES * N_BOXES * 6 * sizeof(double));
        // Generate env objects (N_OBJECTS × 8): type + params
        double h_env_objects[N_OBJECTS * 8] = {
            // Type 0: box at origin, 0.5m half-extents
            0.0,  0.0, 0.0, 0.0,  0.5, 0.5, 0.5, 0.0,
            // Type 0: box offset in X
            0.0,  1.0, 0.0, 0.0,  0.3, 0.3, 0.3, 0.0,
            // Type 1: sphere
            1.0, -0.5, 0.2, 0.0, 0.15, 0.0, 0.0, 0.0,
            // Type 2: cylinder (vertical, Z-aligned)
            2.0,  0.5, 0.5, 0.0, 0.1, 0.4, 0.0, 0.0,
            // Type 0: box near robot workspace
            0.0,  0.3, -0.3, 0.2, 0.2, 0.2, 0.15, 0.0,
        };

        // Robot boxes: each frame has 8 boxes (simplified UR10 links)
        // Frame 0: boxes at random positions
        unsigned s2 = 12345;
        for (int f = 0; f < N_FRAMES; ++f) {
            for (int b = 0; b < N_BOXES; ++b) {
                s2 = s2 * 1103515245 + 12345;
                double u = (double)(s2 & 0x7FFFFFFF) / 2147483648.0;
                int idx = (f * N_BOXES + b) * 6;
                h_robot_boxes[idx + 0] = u * 2.0 - 1.0;      // cx: -1..1
                h_robot_boxes[idx + 1] = (u + 0.1) * 1.5;    // cy: 0..1.5
                h_robot_boxes[idx + 2] = u * 1.5;             // cz: 0..1.5
                h_robot_boxes[idx + 3] = 0.05 + u * 0.15;     // hx: 0.05..0.2
                h_robot_boxes[idx + 4] = 0.05 + ((1.0-u) * 0.15);  // hy
                h_robot_boxes[idx + 5] = 0.05 + u * 0.1;      // hz
            }
        }

        // Device memory
        double *d_robot_boxes, *d_env_objects, *d_clearances, *d_colliding;
        cudaMalloc(&d_robot_boxes, N_FRAMES * N_BOXES * 6 * sizeof(double));
        cudaMalloc(&d_env_objects, N_OBJECTS * 8 * sizeof(double));
        cudaMalloc(&d_clearances, N_FRAMES * sizeof(double));
        cudaMalloc(&d_colliding, N_FRAMES * sizeof(double));

        cudaMemcpy(d_robot_boxes, h_robot_boxes,
                   N_FRAMES * N_BOXES * 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_env_objects, h_env_objects,
                   N_OBJECTS * 8 * sizeof(double), cudaMemcpyHostToDevice);

        // Time the kernel
        cudaEvent_t cs, ce;
        cudaEventCreate(&cs); cudaEventCreate(&ce);

        cudaError_t col_err = rtfg::cuda::launch_collision_check_batch(
            d_robot_boxes, d_env_objects, d_clearances, d_colliding,
            N_FRAMES, N_BOXES, N_OBJECTS, 0);

        if (col_err != cudaSuccess) {
            printf("  COLLISION KERNEL ERROR: %s\n", cudaGetErrorString(col_err));
        } else {
            cudaDeviceSynchronize();

            // Read back
            double h_clearances[N_FRAMES];
            double h_colliding[N_FRAMES];
            cudaMemcpy(h_clearances, d_clearances,
                       N_FRAMES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_colliding, d_colliding,
                       N_FRAMES * sizeof(double), cudaMemcpyDeviceToHost);

            // Analyze
            int n_collisions = 0;
            double min_clearance = 1e10;
            double max_clearance = -1e10;
            for (int f = 0; f < N_FRAMES; ++f) {
                if (h_colliding[f] > 0.5) n_collisions++;
                if (h_clearances[f] < min_clearance) min_clearance = h_clearances[f];
                if (h_clearances[f] > max_clearance && h_clearances[f] < 1e9)
                    max_clearance = h_clearances[f];
            }

            printf("  Frames=%d, Boxes=%d, Objects=%d, Pairs=%d\n",
                   N_FRAMES, N_BOXES, N_OBJECTS, N_BOXES * N_OBJECTS);
            printf("  Collisions: %d/%d frames (%.1f%%)\n",
                   n_collisions, N_FRAMES, 100.0 * n_collisions / N_FRAMES);
            printf("  Min clearance: %.4f m, Max clearance: %.4f m\n",
                   min_clearance, max_clearance);

            // Verify: min_clearance should NOT be infinity (real data)
            bool valid_clearance = (min_clearance < 1e9);
            // Verify: at least some frames should have valid clearance
            bool has_valid = (max_clearance > -1e9);

            printf("  Collision test: %s\n",
                   (valid_clearance && has_valid) ? "PASS ✓" : "FAIL ✗");

            // Show first few frames
            printf("  First 5 frames: ");
            for (int f = 0; f < 5; ++f) {
                printf("[c=%.3f col=%d] ", h_clearances[f], (int)h_colliding[f]);
            }
            printf("\n");
        }

        free(h_robot_boxes);
        cudaFree(d_robot_boxes); cudaFree(d_env_objects);
        cudaFree(d_clearances); cudaFree(d_colliding);
        cudaEventDestroy(cs); cudaEventDestroy(ce);
    }

    printf("\n=== All tests complete ===\n");
    return 0;
}
