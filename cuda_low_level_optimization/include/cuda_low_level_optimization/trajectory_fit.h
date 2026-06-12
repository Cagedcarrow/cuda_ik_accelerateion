// trajectory_fit.h — GPU-accelerated trajectory fitting pipeline
//
// Three-layer architecture (MATLAB-compatible):
//   Layer 1: Adaptive anchor selection (CPU)
//   Layer 2: GPU multi-seed × multi-weight IK + continuity cost + top-K (GPU)
//   Layer 3: Quintic polynomial C² interpolation (CPU)
//
// The pipeline solves the IK multi-solution branch-flip problem by:
//   1. Selecting sparse keyframes (anchors) from dense path points
//   2. At each anchor, trying 51 seeds × 4 weights = 204 candidate IK solutions
//   3. Selecting the candidate with lowest joint-space continuity cost vs q_prev
//   4. Interpolating between anchors with C²-continuous quintic polynomials
//
// This replaces the naive sequential single-seed approach which achieved
// only 3.6% convergence on 500-frame sinusoidal trajectories.

#pragma once

#include <vector>
#include <cstddef>

namespace rtfg {
namespace trajectory {

// ============================================================================
// Data Structures
// ============================================================================

// A single Cartesian pose target (4×4 homogeneous transform, row-major)
struct PoseTarget {
    double T[16];  // 4×4 row-major homogeneous transform
};

// A single anchor frame with its IK result
struct AnchorFrame {
    int    path_index;      // Index in the original dense path
    double q[6];            // Selected joint angles (rad)
    double T_tcp[16];       // TCP transform at this anchor
    double pos_err;         // Residual position error (m)
    double rot_err;         // Residual rotation error (rad)
    int    seed_used;       // Which seed index converged (for diagnostics)
    int    weight_used;     // Which weight level converged
};

// A dense playback frame (interpolated)
struct PlaybackFrame {
    double q[6];            // Joint angles
    double dq[6];           // Joint velocities (rad/s)
    double T_tcp[16];       // TCP transform
    double timestamp;       // Time from start (s)
};

// Continuity cost candidate
struct Candidate {
    double cost;            // Continuity cost (lower = smoother)
    int    index;           // Index into (seed×weight) results array
    int    seed_idx;        // Seed index
    int    weight_idx;      // Weight level index
};

// Trajectory fitting configuration
struct FitConfig {
    // Anchor selection thresholds
    double path_length_threshold;    // > this → anchor (default: 0.012 m = 12 mm)
    double rotation_threshold_deg;   // > this → anchor (default: 0.55°)
    double tangent_change_threshold; // > this → anchor (default: 0.50°)
    int    max_anchors;             // Maximum anchor count (default: 200)

    // IK solver parameters
    int    num_seeds;        // Total seeds/anchor (default: 51)
    int    num_fixed_seeds;  // Fixed seeds (home+zero+wraps+random, default: 50)
    int    num_weight_levels;// Weight schedule levels (default: 4)
    int    max_iter;         // Max DLS iterations per solve (default: 60)
    double pos_tol;          // Position tolerance in meters (default: 0.03)
    double orient_tol;       // Orientation tolerance in rad (default: 0.1)

    // Continuity cost
    int    top_k;            // Top-K candidates to keep per anchor (default: 3)
    double alpha_velocity;   // Velocity penalty weight (default: 0.65)
    double branch_threshold_deg; // > this → branch-switch penalty (default: 25°)

    // Interpolation
    double max_joint_step_deg;    // Max joint step for sub-step generation (default: 0.70°)
    double max_position_step_mm;  // Max position step (default: 3.0 mm)
    double max_rotation_step_deg; // Max rotation step (default: 0.30°)

    // Default constructor with MATLAB-compatible defaults
    FitConfig()
        : path_length_threshold(0.012)
        , rotation_threshold_deg(0.55)
        , tangent_change_threshold(0.50)
        , max_anchors(200)
        , num_seeds(51)
        , num_fixed_seeds(50)
        , num_weight_levels(4)
        , max_iter(60)
        , pos_tol(0.03)
        , orient_tol(0.1)
        , top_k(3)
        , alpha_velocity(0.65)
        , branch_threshold_deg(25.0)
        , max_joint_step_deg(0.70)
        , max_position_step_mm(3.0)
        , max_rotation_step_deg(0.30)
    {}
};

// ============================================================================
// Pipeline API
// ============================================================================

// Generate a 500-frame sinusoidal test trajectory (MATLAB-compatible)
//   f_x=0.5Hz, f_y=0.7Hz, f_z=0.3Hz, phase offsets 120°
//   A_x=0.15m, A_y=0.10m, A_z=0.08m, dt=0.02s (50Hz)
//   q_home: 6-element home joint angles (rad), typically UR10 zero config
void generate_sinusoidal_trajectory(
    const double q_home[6],
    int num_frames,
    double dt,
    std::vector<PoseTarget>& path);

// Select adaptive anchors from dense path points
// Returns indices into the path vector
std::vector<int> select_adaptive_anchors(
    const std::vector<PoseTarget>& path,
    const FitConfig& cfg);

// Generate fixed seeds (home, zero, ±2π wraps, random)
// These are the SAME for all anchors (don't depend on q_prev)
//   fixed_seeds: [num_fixed_seeds * 6] output array
//   q_home: UR10 home/zero configuration (6 doubles)
void generate_fixed_seeds(
    double* fixed_seeds,
    int num_fixed_seeds,
    const double q_home[6],
    unsigned int random_seed = 42);

// Run the full trajectory fitting pipeline
//
// Input:
//   path[0..N-1]: dense Cartesian path points
//   anchor_indices: selected keyframe indices into path
//   q_home[6]: UR10 home configuration
//   cfg: pipeline configuration
//
// Output:
//   anchors: IK results for each anchor frame
//   playback: dense interpolated playback frames
//
// Returns true on success, false if any anchor failed to converge.
bool run_trajectory_pipeline(
    const std::vector<PoseTarget>& path,
    const std::vector<int>& anchor_indices,
    const double q_home[6],
    const FitConfig& cfg,
    std::vector<AnchorFrame>& anchors,
    std::vector<PlaybackFrame>& playback);

// Quintic polynomial interpolation between two anchors
//   s(t) = 10t³ - 15t⁴ + 6t⁵  (C² continuous: s(0)=s'(0)=s''(0)=0, s(1)=1,s'(1)=s''(1)=0)
// Generates sub-steps respecting max_joint_step, max_position_step, max_rotation_step.
void quintic_interpolate(
    const AnchorFrame& from,
    const AnchorFrame& to,
    double dt_total,                    // Total time between anchors (s)
    const FitConfig& cfg,
    std::vector<PlaybackFrame>& frames);

// Single-step quintic interpolation coefficient
inline double quintic_s(double t) {
    // s(t) = 10t³ - 15t⁴ + 6t⁵  for t ∈ [0, 1]
    double t2 = t * t;
    double t3 = t2 * t;
    return 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * t2 * t3;
}

// Quintic first derivative: s'(t) = 30t² - 60t³ + 30t⁴
inline double quintic_s_dot(double t) {
    double t2 = t * t;
    return 30.0 * t2 - 60.0 * t2 * t + 30.0 * t2 * t2;
}

}  // namespace trajectory
}  // namespace rtfg
