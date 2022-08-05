
#ifndef TEST_NCNN_COMMON_H
#define TEST_NCNN_COMMON_H

#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace common {

    struct object_rect {
        int x;
        int y;
        int width;
        int height;
    };

    struct YoloBoxInfo
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    };

    void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<YoloBoxInfo>& bboxes, object_rect effect_roi);

}

#endif //TEST_NCNN_COMMON_H
