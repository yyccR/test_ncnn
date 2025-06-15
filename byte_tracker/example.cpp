#include "byte_tracker.h"
#include <iostream>

int main() {
    byte_tracker::BYTETracker tracker(30, 0.6f, 0.1f, 0.8f);

    // 模拟检测结果，格式为 [x, y, w, h]
    std::vector<std::vector<float>> detections = {
        {100, 100, 50, 50},  // [x, y, w, h]
        {200, 200, 50, 50}
    };
    std::vector<float> scores = {0.9, 0.8};
    std::vector<int> classes = {0, 0};

    // 更新跟踪器
    int img_w = 1280, img_h = 720; // 假设测试分辨率
    std::vector<std::vector<float>> results = tracker.update(detections, scores, classes, img_w, img_h);
    return 0;
} 