// trajectory_fit.cpp — GPU-accelerated trajectory fitting pipeline implementation
//
// Three-layer architecture:
//   Layer 1: Adaptive anchor selection (CPU, O(N) scan)
//   Layer 2: GPU multi-seed × multi-weight IK + continuity cost + top-K
//   Layer 3: Quintic polynomial C² interpolation (CPU)

#include "cuda_low_level_optimization/trajectory_fit.h"
#include "cuda_low_level_optimization/cuda_kernels.h"
#include "cuda_low_level_optimization/cuda_memory.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rtfg {
namespace trajectory {

// ============================================================================
// Forward kinematics (CPU reference, matches GPU exactly)
// ============================================================================
namespace {

// UR10 kinematics constants (must match cuda_utilities.cuh / test_cuda_kernel.cu)
static const double k_origins[96] = {
    1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.1273,  0.0, 0.0, 0.0, 1.0,
    6.3267948966684693e-06, 0.0, 9.9999999997998579e-01, 0.0,
    0.0, 1.0, 0.0, 0.220941,
    -9.9999999997998579e-01, 0.0, 6.3267948966684693e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, -3.9e-6,  0.0, 1.0, 0.0, -0.1719,  0.0, 0.0, 1.0, 0.612,  0.0, 0.0, 0.0, 1.0,
    6.8267948964806121e-06, -2.6999999999967194e-06, 9.9999999997305244e-01, -3.6e-6,
    1.8432346220542441e-11, 9.9999999999635503e-01, 2.6999999999338027e-06, 0.0,
    -9.9999999997669742e-01, 0.0, 6.8267948965054954e-06, 0.5723,
    0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 3e-7,  0.0, 1.0, 0.0, 0.1149,  0.0, 0.0, 1.0, 3e-7,  0.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, -3e-7,  0.0, 0.0, 1.0, 0.1157,  0.0, 0.0, 0.0, 1.0
};

static const double k_axes[18] = {
    0.0, 0.0,  1.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0, 1.0, 0.0
};

static const int k_q_index[6] = {0, 1, 2, 3, 4, 5};

static const double k_T_tcp[16] = {
    -3.0089034369963224e-06, -8.1915141988946039e-01,  5.7357732807706696e-01, -4.7377000000000002e-01,
    -9.9999999999319700e-01,  3.6885740485110307e-06,  2.1962570019296782e-08,  1.6330000206612766e-01,
    -2.1336731175750953e-06, -5.7357732807309880e-01, -8.1915141989498630e-01, -7.7108998035934045e-02,
     0.0, 0.0, 0.0, 1.0
};

static void cpu_mat44_mul(const double* A, const double* B, double* C) {
    for (int r = 0; r < 4; ++r) {
        double a0 = A[r*4+0], a1 = A[r*4+1], a2 = A[r*4+2], a3 = A[r*4+3];
        C[r*4+0] = a0*B[0] + a1*B[4] + a2*B[8]  + a3*B[12];
        C[r*4+1] = a0*B[1] + a1*B[5] + a2*B[9]  + a3*B[13];
        C[r*4+2] = a0*B[2] + a1*B[6] + a2*B[10] + a3*B[14];
        C[r*4+3] = a0*B[3] + a1*B[7] + a2*B[11] + a3*B[15];
    }
}

static void cpu_rotation_matrix(double ax, double ay, double az, double angle, double* R) {
    double c = cos(angle), s = sin(angle), t = 1.0 - c;
    R[0]  = t*ax*ax + c;   R[1]  = t*ax*ay + s*az; R[2]  = t*ax*az - s*ay; R[3]  = 0.0;
    R[4]  = t*ax*ay - s*az; R[5]  = t*ay*ay + c;   R[6]  = t*ay*az + s*ax; R[7]  = 0.0;
    R[8]  = t*ax*az + s*ay; R[9]  = t*ay*az - s*ax; R[10] = t*az*az + c;   R[11] = 0.0;
    R[12] = 0.0;            R[13] = 0.0;            R[14] = 0.0;            R[15] = 1.0;
}

// CPU forward kinematics — returns TCP transform
void cpu_fk(const double q[6], double T_tcp[16]) {
    double T[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    double T_tmp[16], R[16];

    for (int seg = 0; seg < 6; ++seg) {
        cpu_mat44_mul(T, &k_origins[seg * 16], T_tmp);
        for (int i = 0; i < 16; ++i) T[i] = T_tmp[i];

        double theta = q[k_q_index[seg]];
        cpu_rotation_matrix(k_axes[seg*3+0], k_axes[seg*3+1], k_axes[seg*3+2], theta, R);
        cpu_mat44_mul(T, R, T_tmp);
        for (int i = 0; i < 16; ++i) T[i] = T_tmp[i];
    }

    cpu_mat44_mul(T, k_T_tcp, T_tcp);
}

// Simple PRNG (matches test_cuda_kernel.cu LCG)
unsigned int lcg_rand(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

double lcg_rand_double(unsigned int* state) {
    return (double)(lcg_rand(state) & 0x7FFFFFFF) / 2147483648.0;
}

}  // anonymous namespace

// ============================================================================
// Generate sinusoidal test trajectory
// ============================================================================
void generate_sinusoidal_trajectory(
    const double q_home[6],
    int num_frames,
    double dt,
    std::vector<PoseTarget>& path)
{
    path.clear();
    path.reserve(num_frames);

    // Home TCP position
    double T_home[16];
    cpu_fk(q_home, T_home);
    double base_x = T_home[3];
    double base_y = T_home[7];
    double base_z = T_home[11];

    // Sinusoidal parameters (MATLAB-compatible)
    const double Ax = 0.15, Ay = 0.10, Az = 0.08;  // amplitudes (m)
    const double fx = 0.5,  fy = 0.7,  fz = 0.3;   // frequencies (Hz)
    const double phase_y = 120.0 * M_PI / 180.0;     // 120° phase offset
    const double phase_z = 240.0 * M_PI / 180.0;     // 240° phase offset

    for (int i = 0; i < num_frames; ++i) {
        double t = i * dt;
        double dx = Ax * sin(2.0 * M_PI * fx * t);
        double dy = Ay * sin(2.0 * M_PI * fy * t + phase_y);
        double dz = Az * sin(2.0 * M_PI * fz * t + phase_z);

        PoseTarget pt;
        for (int r = 0; r < 16; ++r) pt.T[r] = T_home[r];
        pt.T[3]  = base_x + dx;
        pt.T[7]  = base_y + dy;
        pt.T[11] = base_z + dz;
        path.push_back(pt);
    }
}

// ============================================================================
// Adaptive anchor selection
// ============================================================================
std::vector<int> select_adaptive_anchors(
    const std::vector<PoseTarget>& path,
    const FitConfig& cfg)
{
    std::vector<int> anchors;
    if (path.empty()) return anchors;

    // Always include first frame
    anchors.push_back(0);

    double rot_thresh = cfg.rotation_threshold_deg * M_PI / 180.0;
    int last_anchor = 0;

    for (int i = 1; i < (int)path.size(); ++i) {
        if ((int)anchors.size() >= cfg.max_anchors) break;

        const double* T_prev = path[last_anchor].T;
        const double* T_curr = path[i].T;

        // Path length from last anchor
        double dx = T_curr[3]  - T_prev[3];
        double dy = T_curr[7]  - T_prev[7];
        double dz = T_curr[11] - T_prev[11];
        double path_len = sqrt(dx*dx + dy*dy + dz*dz);

        // Rotation angle from last anchor (trace of relative rotation)
        double rot_angle = 0.0;
        {
            double r00 = T_curr[0], r01 = T_curr[1], r02 = T_curr[2];
            double r10 = T_curr[4], r11 = T_curr[5], r12 = T_curr[6];
            double r20 = T_curr[8], r21 = T_curr[9], r22 = T_curr[10];

            double p00 = T_prev[0], p01 = T_prev[1], p02 = T_prev[2];
            double p10 = T_prev[4], p11 = T_prev[5], p12 = T_prev[6];
            double p20 = T_prev[8], p21 = T_prev[9], p22 = T_prev[10];

            // R_rel = R_curr * R_prev^T
            double rel[9];
            rel[0] = r00*p00 + r01*p01 + r02*p02;
            rel[1] = r00*p10 + r01*p11 + r02*p12;
            rel[2] = r00*p20 + r01*p21 + r02*p22;
            rel[3] = r10*p00 + r11*p01 + r12*p02;
            rel[4] = r10*p10 + r11*p11 + r12*p12;
            rel[5] = r10*p20 + r11*p21 + r12*p22;
            rel[6] = r20*p00 + r21*p01 + r22*p02;
            rel[7] = r20*p10 + r21*p11 + r22*p12;
            rel[8] = r20*p20 + r21*p21 + r22*p22;

            double trace = rel[0] + rel[4] + rel[8];
            double cos_theta = (trace - 1.0) * 0.5;
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            rot_angle = acos(cos_theta);
        }

        // Tangent change (simplified: use position direction change)
        double tangent_change = 0.0;
        if (i >= 2) {
            const double* T_prev2 = path[i-1].T;
            double dx1 = T_curr[3] - T_prev2[3];
            double dy1 = T_curr[7] - T_prev2[7];
            double dz1 = T_curr[11] - T_prev2[11];
            double len1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dy1);
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

        // Anchor condition: any threshold exceeded
        if (path_len > cfg.path_length_threshold ||
            rot_angle > rot_thresh ||
            tangent_change > cfg.tangent_change_threshold * M_PI / 180.0)
        {
            anchors.push_back(i);
            last_anchor = i;
        }
    }

    // Always include last frame
    if (anchors.back() != (int)path.size() - 1) {
        anchors.push_back((int)path.size() - 1);
    }

    return anchors;
}

// ============================================================================
// Generate fixed seeds (shared across all anchors)
// ============================================================================
void generate_fixed_seeds(
    double* fixed_seeds,
    int num_fixed_seeds,
    const double q_home[6],
    unsigned int random_seed)
{
    // Seed layout: home(1) + zero(1) + ±2π_wraps(27) + random(remaining)
    // Total: 1 + 1 + 27 + (num_fixed_seeds - 29) = num_fixed_seeds
    int idx = 0;

    // Seed 0: home configuration
    for (int j = 0; j < 6; ++j) fixed_seeds[idx * 6 + j] = q_home[j];
    idx++;

    // Seed 1: all zeros
    for (int j = 0; j < 6; ++j) fixed_seeds[idx * 6 + j] = 0.0;
    idx++;

    // Seeds 2-28: ±2π wraps on first 3 joints (3³ = 27)
    // For each of the 3 major joints: -2π, 0, +2π offset from home
    const double wrap_offsets[3] = {-2.0 * M_PI, 0.0, 2.0 * M_PI};
    for (int w0 = 0; w0 < 3; ++w0) {
        for (int w1 = 0; w1 < 3; ++w1) {
            for (int w2 = 0; w2 < 3; ++w2) {
                for (int j = 0; j < 6; ++j) fixed_seeds[idx * 6 + j] = q_home[j];
                fixed_seeds[idx * 6 + 0] += wrap_offsets[w0];
                fixed_seeds[idx * 6 + 1] += wrap_offsets[w1];
                fixed_seeds[idx * 6 + 2] += wrap_offsets[w2];
                idx++;
            }
        }
    }

    // Remaining seeds: random within joint limits (±π)
    unsigned int rng_state = random_seed;
    while (idx < num_fixed_seeds) {
        for (int j = 0; j < 6; ++j) {
            fixed_seeds[idx * 6 + j] = -M_PI + 2.0 * M_PI * lcg_rand_double(&rng_state);
        }
        idx++;
    }
}

// ============================================================================
// Single-anchor GPU solve: 51 seeds × 4 weights → top-K candidates
// ============================================================================
static bool solve_single_anchor(
    const double* d_target,           // [16] target transform
    const double* d_all_seeds,        // [num_seeds * 6] all seeds
    const double* d_q_prev,           // [6] previous joint angles
    const double* d_joint_weights,    // [6] joint weights for cost
    double* d_results,                // [num_seeds * num_weights * 6] output
    double* d_errors,                 // [num_seeds * num_weights * 2]
    double* d_shovel_errors,          // [num_seeds * num_weights * 2]
    double* d_iterations,             // [num_seeds * num_weights]
    double* d_costs,                  // [num_seeds * num_weights]
    double* d_topk_costs,             // [top_k]
    int*    d_topk_indices,           // [top_k]
    const FitConfig& cfg,
    std::vector<Candidate>& candidates)
{
    using namespace rtfg::cuda;

    int total = cfg.num_seeds * cfg.num_weight_levels;

    // Step 1: GPU multi-seed × multi-weight IK
    cudaError_t err = launch_ik_batch_solve_multi(
        d_target, d_all_seeds,
        d_results, d_errors, d_shovel_errors, d_iterations,
        cfg.max_iter, cfg.pos_tol, cfg.orient_tol,
        cfg.num_seeds, cfg.num_weight_levels);
    if (err != cudaSuccess) {
        fprintf(stderr, "  ik_batch_solve_multi ERROR: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaDeviceSynchronize();

    // Step 2: GPU continuity cost computation
    err = launch_compute_continuity_cost_all(
        d_results, d_q_prev, d_costs, total, d_joint_weights);
    if (err != cudaSuccess) {
        fprintf(stderr, "  compute_continuity_cost_all ERROR: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaDeviceSynchronize();

    // Step 3: GPU bitonic sort → top-K
    err = launch_filter_topk_per_target(
        d_costs, d_iterations, d_topk_costs, d_topk_indices,
        total, cfg.top_k, 0);  // min_iterations=0: accept all
    if (err != cudaSuccess) {
        fprintf(stderr, "  filter_topk_per_target ERROR: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaDeviceSynchronize();

    // Step 4: Read back top-K candidates
    double h_topk_costs[8];
    int    h_topk_indices[8];
    cudaMemcpy(h_topk_costs,  d_topk_costs,  cfg.top_k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_indices, d_topk_indices, cfg.top_k * sizeof(int), cudaMemcpyDeviceToHost);

    candidates.clear();
    for (int k = 0; k < cfg.top_k; ++k) {
        if (h_topk_costs[k] >= 1e99) continue;  // Invalid (unconverged)
        Candidate c;
        c.cost       = h_topk_costs[k];
        c.index      = h_topk_indices[k];
        c.seed_idx   = c.index / cfg.num_weight_levels;
        c.weight_idx = c.index % cfg.num_weight_levels;
        candidates.push_back(c);
    }

    return !candidates.empty();
}

// ============================================================================
// Greedy selection: pick lowest-cost converged candidate
// ============================================================================
static bool greedy_select(
    const std::vector<Candidate>& candidates,
    const double* d_results,        // [total * 6]
    const double* d_errors,         // [total * 2]
    const double* d_iterations,     // [total]
    const double q_prev[6],
    AnchorFrame& anchor,
    const FitConfig& cfg)
{
    if (candidates.empty()) return false;

    // Read back all needed results for candidates
    // For simplicity, read the best candidate's full data
    const Candidate& best = candidates[0];  // Already sorted by cost

    double h_result[6], h_error[2], h_iter;
    cudaMemcpy(h_result, d_results + best.index * 6, 6 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error,  d_errors  + best.index * 2, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_iter,  d_iterations + best.index, sizeof(double), cudaMemcpyDeviceToHost);

    // Check convergence
    if (h_error[0] > cfg.pos_tol || h_error[1] > cfg.orient_tol) {
        // Try next candidates
        for (size_t k = 1; k < candidates.size(); ++k) {
            const Candidate& alt = candidates[k];
            cudaMemcpy(h_result, d_results + alt.index * 6, 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_error,  d_errors  + alt.index * 2, 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_iter,  d_iterations + alt.index, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_error[0] <= cfg.pos_tol && h_error[1] <= cfg.orient_tol) {
                // Found converged alternative
                anchor.seed_used   = alt.seed_idx;
                anchor.weight_used = alt.weight_idx;
                goto converged;
            }
        }
        // No candidate converged
        return false;
    }

converged:
    // Joint alignment: wrap to nearest ±π of q_prev
    for (int j = 0; j < 6; ++j) {
        double diff = h_result[j] - q_prev[j];
        anchor.q[j] = q_prev[j] + atan2(sin(diff), cos(diff));
    }
    anchor.pos_err = h_error[0];
    anchor.rot_err = h_error[1];
    anchor.seed_used   = best.seed_idx;
    anchor.weight_used = best.weight_idx;

    return true;
}

// ============================================================================
// Full trajectory pipeline
// ============================================================================
bool run_trajectory_pipeline(
    const std::vector<PoseTarget>& path,
    const std::vector<int>& anchor_indices,
    const double q_home[6],
    const FitConfig& cfg,
    std::vector<AnchorFrame>& anchors,
    std::vector<PlaybackFrame>& playback)
{
    int num_anchors = (int)anchor_indices.size();
    int total_per_anchor = cfg.num_seeds * cfg.num_weight_levels;

    printf("\n=== GPU Trajectory Fitting Pipeline ===\n");
    printf("  Anchors: %d, Seeds/anchor: %d, Weights: %d, Candidates/anchor: %d\n",
           num_anchors, cfg.num_seeds, cfg.num_weight_levels, total_per_anchor);
    printf("  Total IK solves: %d\n", num_anchors * total_per_anchor);

    // =========================================================================
    // Allocate GPU memory (reused across anchors)
    // =========================================================================
    double *d_target, *d_all_seeds, *d_q_prev, *d_joint_weights;
    double *d_results, *d_errors, *d_shovel_errors, *d_iterations;
    double *d_costs, *d_topk_costs;
    int    *d_topk_indices;

    size_t seeds_bytes    = cfg.num_seeds * 6 * sizeof(double);
    size_t results_bytes  = total_per_anchor * 6 * sizeof(double);
    size_t errors_bytes   = total_per_anchor * 2 * sizeof(double);
    size_t iters_bytes    = total_per_anchor * sizeof(double);
    size_t costs_bytes    = total_per_anchor * sizeof(double);

    cudaMalloc(&d_target,        16 * sizeof(double));
    cudaMalloc(&d_all_seeds,     seeds_bytes);
    cudaMalloc(&d_q_prev,        6 * sizeof(double));
    cudaMalloc(&d_joint_weights, 6 * sizeof(double));
    cudaMalloc(&d_results,       results_bytes);
    cudaMalloc(&d_errors,        errors_bytes);
    cudaMalloc(&d_shovel_errors, errors_bytes);
    cudaMalloc(&d_iterations,    iters_bytes);
    cudaMalloc(&d_costs,         costs_bytes);
    cudaMalloc(&d_topk_costs,    cfg.top_k * sizeof(double));
    cudaMalloc(&d_topk_indices,  cfg.top_k * sizeof(int));

    // Joint weights: equal weighting for continuity cost
    double h_joint_weights[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    cudaMemcpy(d_joint_weights, h_joint_weights, 6 * sizeof(double), cudaMemcpyHostToDevice);

    // Generate fixed seeds (shared across all anchors)
    std::vector<double> h_fixed_seeds(cfg.num_fixed_seeds * 6);
    generate_fixed_seeds(h_fixed_seeds.data(), cfg.num_fixed_seeds, q_home, 42);

    // All seeds array: fixed seeds + 1 slot for q_prev seed
    std::vector<double> h_all_seeds(cfg.num_seeds * 6);
    // Copy fixed seeds (indices 0..num_fixed_seeds-1)
    memcpy(h_all_seeds.data(), h_fixed_seeds.data(),
           cfg.num_fixed_seeds * 6 * sizeof(double));
    // Seed index num_fixed_seeds is the q_prev slot (filled per anchor)

    // =========================================================================
    // Process anchors sequentially (q_prev dependency)
    // =========================================================================
    anchors.clear();
    anchors.reserve(num_anchors);

    // Anchor 0 uses home as q_prev reference
    double q_prev[6];
    for (int j = 0; j < 6; ++j) q_prev[j] = q_home[j];

    int converged_count = 0;
    double max_joint_jump = 0.0;  // Track worst jump for diagnostics

    for (int a = 0; a < num_anchors; ++a) {
        int path_idx = anchor_indices[a];

        // Upload target
        cudaMemcpy(d_target, path[path_idx].T, 16 * sizeof(double), cudaMemcpyHostToDevice);

        // Set q_prev seed (index num_fixed_seeds)
        for (int j = 0; j < 6; ++j)
            h_all_seeds[cfg.num_fixed_seeds * 6 + j] = q_prev[j];
        cudaMemcpy(d_all_seeds, h_all_seeds.data(), seeds_bytes, cudaMemcpyHostToDevice);

        // Upload q_prev for continuity cost
        cudaMemcpy(d_q_prev, q_prev, 6 * sizeof(double), cudaMemcpyHostToDevice);

        // Solve on GPU
        std::vector<Candidate> candidates;
        bool ok = solve_single_anchor(
            d_target, d_all_seeds, d_q_prev, d_joint_weights,
            d_results, d_errors, d_shovel_errors, d_iterations,
            d_costs, d_topk_costs, d_topk_indices,
            cfg, candidates);

        if (!ok) {
            fprintf(stderr, "  Anchor %d (path_idx=%d): NO converged candidate!\n", a, path_idx);
            // Fallback: use q_prev (no movement)
            AnchorFrame af;
            af.path_index = path_idx;
            for (int j = 0; j < 6; ++j) af.q[j] = q_prev[j];
            af.pos_err = 0.0; af.rot_err = 0.0;
            af.seed_used = -1; af.weight_used = -1;
            anchors.push_back(af);
            continue;
        }

        // Greedy select best candidate
        AnchorFrame af;
        af.path_index = path_idx;
        bool converged = greedy_select(
            candidates, d_results, d_errors, d_iterations,
            q_prev, af, cfg);

        if (!converged) {
            fprintf(stderr, "  Anchor %d (path_idx=%d): top-%d unconverged, using best-effort\n",
                    a, path_idx, cfg.top_k);
            // Best-effort: use first candidate anyway
            const Candidate& best = candidates[0];
            double h_result[6];
            cudaMemcpy(h_result, d_results + best.index * 6, 6 * sizeof(double), cudaMemcpyDeviceToHost);
            for (int j = 0; j < 6; ++j) {
                double diff = h_result[j] - q_prev[j];
                af.q[j] = q_prev[j] + atan2(sin(diff), cos(diff));
            }
            af.seed_used = best.seed_idx;
            af.weight_used = best.weight_idx;
        } else {
            converged_count++;
        }

        // Compute TCP transform at this anchor
        cpu_fk(af.q, af.T_tcp);

        // Track max joint jump
        for (int j = 0; j < 6; ++j) {
            double jump = fabs(af.q[j] - q_prev[j]);
            double wrapped_jump = fabs(atan2(sin(af.q[j] - q_prev[j]), cos(af.q[j] - q_prev[j])));
            if (wrapped_jump > max_joint_jump) max_joint_jump = wrapped_jump;
        }

        anchors.push_back(af);

        // Update q_prev for next anchor
        for (int j = 0; j < 6; ++j) q_prev[j] = af.q[j];

        if ((a + 1) % 20 == 0 || a == num_anchors - 1) {
            printf("  Anchor %3d/%d (path_idx=%d): seed=%d weight=%d err=%.3fmm jump=%.3f°\n",
                   a + 1, num_anchors, path_idx,
                   af.seed_used, af.weight_used,
                   af.pos_err * 1000.0,
                   max_joint_jump * 180.0 / M_PI);
        }
    }

    double conv_rate = 100.0 * converged_count / num_anchors;
    printf("  Convergence: %d/%d (%.1f%%), Max joint jump: %.3f°\n",
           converged_count, num_anchors, conv_rate,
           max_joint_jump * 180.0 / M_PI);

    // =========================================================================
    // Quintic interpolation between anchors
    // =========================================================================
    playback.clear();

    // Compute timestamps: assume uniform dt between path points
    // Use the path index to determine timing
    for (size_t a = 0; a + 1 < anchors.size(); ++a) {
        const AnchorFrame& from = anchors[a];
        const AnchorFrame& to   = anchors[a + 1];

        int num_steps = to.path_index - from.path_index;
        if (num_steps <= 0) continue;

        // Add 'from' frame
        PlaybackFrame pf_from;
        for (int j = 0; j < 6; ++j) {
            pf_from.q[j] = from.q[j];
            pf_from.dq[j] = 0.0;
        }
        for (int i = 0; i < 16; ++i) pf_from.T_tcp[i] = from.T_tcp[i];
        pf_from.timestamp = from.path_index * 0.02;  // 50Hz
        playback.push_back(pf_from);

        // Quintic interpolation for interior frames
        for (int s = 1; s < num_steps; ++s) {
            double t = (double)s / (double)num_steps;  // t ∈ [0, 1]
            double st = quintic_s(t);

            PlaybackFrame pf;
            for (int j = 0; j < 6; ++j) {
                pf.q[j] = from.q[j] + st * (to.q[j] - from.q[j]);
            }
            // Velocity via quintic derivative
            double dt_total = num_steps * 0.02;
            double st_dot = quintic_s_dot(t);
            for (int j = 0; j < 6; ++j) {
                pf.dq[j] = st_dot * (to.q[j] - from.q[j]) / dt_total;
            }
            cpu_fk(pf.q, pf.T_tcp);
            pf.timestamp = (from.path_index + s) * 0.02;
            playback.push_back(pf);
        }
    }

    // Add final frame
    {
        const AnchorFrame& last = anchors.back();
        PlaybackFrame pf_last;
        for (int j = 0; j < 6; ++j) {
            pf_last.q[j] = last.q[j];
            pf_last.dq[j] = 0.0;
        }
        for (int i = 0; i < 16; ++i) pf_last.T_tcp[i] = last.T_tcp[i];
        pf_last.timestamp = last.path_index * 0.02;
        playback.push_back(pf_last);
    }

    printf("  Playback frames: %zu (anchors: %zu)\n", playback.size(), anchors.size());

    // =========================================================================
    // Cleanup
    // =========================================================================
    cudaFree(d_target);
    cudaFree(d_all_seeds);
    cudaFree(d_q_prev);
    cudaFree(d_joint_weights);
    cudaFree(d_results);
    cudaFree(d_errors);
    cudaFree(d_shovel_errors);
    cudaFree(d_iterations);
    cudaFree(d_costs);
    cudaFree(d_topk_costs);
    cudaFree(d_topk_indices);

    return converged_count == num_anchors;
}

// ============================================================================
// Quintic interpolation (standalone)
// ============================================================================
void quintic_interpolate(
    const AnchorFrame& from,
    const AnchorFrame& to,
    double dt_total,
    const FitConfig& cfg,
    std::vector<PlaybackFrame>& frames)
{
    // Determine max joint step
    double max_dq = 0.0;
    for (int j = 0; j < 6; ++j) {
        double dq = fabs(to.q[j] - from.q[j]);
        if (dq > max_dq) max_dq = dq;
    }

    // Adaptive sub-step count
    int num_substeps = 1;
    double joint_step_rad = cfg.max_joint_step_deg * M_PI / 180.0;
    if (max_dq > joint_step_rad) {
        num_substeps = (int)ceil(max_dq / joint_step_rad);
    }
    if (num_substeps < 2) num_substeps = 2;  // At least start and end

    for (int s = 0; s <= num_substeps; ++s) {
        double t = (double)s / (double)num_substeps;
        double st = quintic_s(t);
        double st_dot = quintic_s_dot(t);

        PlaybackFrame pf;
        for (int j = 0; j < 6; ++j) {
            pf.q[j]  = from.q[j] + st * (to.q[j] - from.q[j]);
            pf.dq[j] = st_dot * (to.q[j] - from.q[j]) / dt_total;
        }
        cpu_fk(pf.q, pf.T_tcp);
        pf.timestamp = from.path_index * 0.02 + t * dt_total;
        frames.push_back(pf);
    }
}

}  // namespace trajectory
}  // namespace rtfg
