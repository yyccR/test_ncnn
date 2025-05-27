

#ifndef test_ncnn_COMMON_H
#define test_ncnn_COMMON_H


#pragma once
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <iostream>


#define MAX_STRIDE 64
#define PERMUTE 0

namespace common {
    struct Object {
        cv::Rect_<float> rect;
        int label{};
        float prob{};
        std::vector<float> mask_feat;
        cv::Mat cv_mask;
        std::vector<cv::Point3f> key_points;
    };

    const unsigned char colors[81][3] = {
        {56,  0,   255},
        {226, 255, 0},
        {0,   94,  255},
        {0,   37,  255},
        {0,   255, 94},
        {255, 226, 0},
        {0,   18,  255},
        {255, 151, 0},
        {170, 0,   255},
        {0,   255, 56},
        {255, 0,   75},
        {0,   75,  255},
        {0,   255, 169},
        {255, 0,   207},
        {75,  255, 0},
        {207, 0,   255},
        {37,  0,   255},
        {0,   207, 255},
        {94,  0,   255},
        {0,   255, 113},
        {255, 18,  0},
        {255, 0,   56},
        {18,  0,   255},
        {0,   255, 226},
        {170, 255, 0},
        {255, 0,   245},
        {151, 255, 0},
        {132, 255, 0},
        {75,  0,   255},
        {151, 0,   255},
        {0,   151, 255},
        {132, 0,   255},
        {0,   255, 245},
        {255, 132, 0},
        {226, 0,   255},
        {255, 37,  0},
        {207, 255, 0},
        {0,   255, 207},
        {94,  255, 0},
        {0,   226, 255},
        {56,  255, 0},
        {255, 94,  0},
        {255, 113, 0},
        {0,   132, 255},
        {255, 0,   132},
        {255, 170, 0},
        {255, 0,   188},
        {113, 255, 0},
        {245, 0,   255},
        {113, 0,   255},
        {255, 188, 0},
        {0,   113, 255},
        {255, 0,   0},
        {0,   56,  255},
        {255, 0,   113},
        {0,   255, 188},
        {255, 0,   94},
        {255, 0,   18},
        {18,  255, 0},
        {0,   255, 132},
        {0,   188, 255},
        {0,   245, 255},
        {0,   169, 255},
        {37,  255, 0},
        {255, 0,   151},
        {188, 0,   255},
        {0,   255, 37},
        {0,   255, 0},
        {255, 0,   170},
        {255, 0,   37},
        {255, 75,  0},
        {0,   0,   255},
        {255, 207, 0},
        {255, 0,   226},
        {255, 245, 0},
        {188, 255, 0},
        {0,   255, 18},
        {0,   255, 75},
        {0,   255, 151},
        {255, 56,  0},
        {245, 255, 0}
    };

    void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color);

    void draw_pose(cv::Mat& bgr, std::vector<cv::Point3f> key_points);

    void transpose(const ncnn::Mat& in, ncnn::Mat& out);

    void matPrint(const ncnn::Mat& m);

    void matVisualize(const char* title, const ncnn::Mat& m, bool save = 0);

    void softmax(ncnn::Mat& bottom);

    void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis);

    void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out);

    void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d);

    void sigmoid(ncnn::Mat& bottom);

    void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob);

    void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                     const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                     ncnn::Mat& mask_pred_result);


    inline float intersection_area(const Object& a, const Object& b);

    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

    void qsort_descent_inplace(std::vector<Object>& faceobjects);

    float fast_exp(float x);

    float sigmoid(float x);

    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);

    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true);


}

#endif //test_ncnn_COMMON_H
