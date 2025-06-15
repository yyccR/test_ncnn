#include "matching.h"
#include "byte_tracker.h"  // 确保可以访问 STrack 的实现
#include <algorithm>
#include <limits>
#include <numeric>

namespace byte_tracker {

// 匈牙利算法实现：用于最优分配检测与轨迹，最小化代价矩阵
std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<float>>& cost_matrix, float thresh) {
    std::vector<std::pair<int, int>> matches; // 存储匹配结果
    if (cost_matrix.empty() || cost_matrix[0].empty()) return matches; // 空矩阵直接返回
    std::vector<int> row_indices(cost_matrix.size()); // 行索引
    std::vector<int> col_indices(cost_matrix[0].size()); // 列索引
    std::iota(row_indices.begin(), row_indices.end(), 0); // 行索引初始化为0,1,2...
    std::iota(col_indices.begin(), col_indices.end(), 0); // 列索引初始化为0,1,2...

    // 简单贪心实现匈牙利算法
    while (!row_indices.empty() && !col_indices.empty()) {
        float min_cost = std::numeric_limits<float>::max(); // 当前最小代价
        int min_row = -1, min_col = -1;
        for (int i : row_indices) {
            for (int j : col_indices) {
                if (cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    min_row = i;
                    min_col = j;
                }
            }
        }
        if (min_cost > thresh) { // 超过阈值则停止
            break;
        }
        matches.push_back({min_row, min_col}); // 记录匹配
        row_indices.erase(std::remove(row_indices.begin(), row_indices.end(), min_row), row_indices.end()); // 移除已匹配行
        col_indices.erase(std::remove(col_indices.begin(), col_indices.end(), min_col), col_indices.end()); // 移除已匹配列
    }
    return matches;
}

// 计算IOU距离，1-IOU越小越相似
std::vector<std::vector<float>> iou_distance(const std::vector<STrack>& tracks, const std::vector<STrack>& detections) {
    std::vector<std::vector<float>> dists(tracks.size(), std::vector<float>(detections.size())); // 距离矩阵
    for (size_t i = 0; i < tracks.size(); i++) {
        for (size_t j = 0; j < detections.size(); j++) {
            dists[i][j] = 1 - calculate_iou(tracks[i].get_xyxy(), detections[j].get_xyxy()); // 1-IOU
        }
    }
    return dists;
}

// 计算两个框的IOU
float calculate_iou(const std::vector<float>& bbox1, const std::vector<float>& bbox2) {
    float x1 = std::max(bbox1[0], bbox2[0]); // 交集左上角x
    float y1 = std::max(bbox1[1], bbox2[1]); // 交集左上角y
    float x2 = std::min(bbox1[2], bbox2[2]); // 交集右下角x
    float y2 = std::min(bbox1[3], bbox2[3]); // 交集右下角y
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1); // 交集面积
    float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]); // 框1面积
    float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]); // 框2面积
    return intersection / (area1 + area2 - intersection); // IOU=交/并
}

// 融合分数到距离矩阵（可选，用于更复杂的匹配策略）
std::vector<std::vector<float>> fuse_score(const std::vector<std::vector<float>>& dists, const std::vector<STrack>& detections) {
    std::vector<std::vector<float>> fused_dists = dists; // 拷贝距离矩阵
    for (size_t i = 0; i < dists.size(); i++) {
        for (size_t j = 0; j < dists[i].size(); j++) {
            fused_dists[i][j] = dists[i][j] * (1 - detections[j].get_score()); // 距离乘以(1-分数)
        }
    }
    return fused_dists;
}

} // namespace byte_tracker 