#pragma once

#include <vector>

namespace byte_tracker {

class KalmanFilterXYAH {
public:
    KalmanFilterXYAH();
    std::pair<std::vector<float>, std::vector<float>> predict(const std::vector<float>& mean, const std::vector<float>& covariance);
    std::pair<std::vector<float>, std::vector<float>> update(const std::vector<float>& mean, const std::vector<float>& covariance, const std::vector<float>& measurement);
    std::pair<std::vector<float>, std::vector<float>> initiate(const std::vector<float>& measurement);

private:
    std::vector<float> motion_mat_;  // 状态转移矩阵
    std::vector<float> update_mat_;  // 更新矩阵
    float std_weight_position_;      // 位置标准差权重
    float std_weight_velocity_;      // 速度标准差权重
};

} // namespace byte_tracker 