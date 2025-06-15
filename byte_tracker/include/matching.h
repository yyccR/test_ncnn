#pragma once

#include <vector>

namespace byte_tracker {

class STrack;  // 前向声明

std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<float>>& cost_matrix, float thresh);
std::vector<std::vector<float>> iou_distance(const std::vector<STrack>& tracks, const std::vector<STrack>& detections);
std::vector<std::vector<float>> fuse_score(const std::vector<std::vector<float>>& dists, const std::vector<STrack>& detections);
float calculate_iou(const std::vector<float>& bbox1, const std::vector<float>& bbox2);

} // namespace byte_tracker 