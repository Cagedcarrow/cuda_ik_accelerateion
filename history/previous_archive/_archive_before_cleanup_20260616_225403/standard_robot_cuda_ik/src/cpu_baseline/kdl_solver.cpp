#include <kdl/chain.hpp>
#include <string>

namespace standard_robot_cuda_ik {

std::string kdl_solver_status() {
  KDL::Chain chain;
  return chain.getNrOfSegments() == 0 ? "kdl_solver_compiled" : "kdl_solver_ready";
}

}  // namespace standard_robot_cuda_ik

