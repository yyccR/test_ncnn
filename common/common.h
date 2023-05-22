

#ifndef TEST_NCNN2_COMMON_H
#define TEST_NCNN2_COMMON_H


#pragma once
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#define MAX_STRIDE 64
#define PERMUTE 0

struct Object {
    cv::Rect_<float> rect;
    int label{};
    float prob{};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
};

extern const unsigned char colors[81][3];

void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color);

void matPrint(const ncnn::Mat& m);

void matVisualize(const char* title, const ncnn::Mat& m, bool save = 0);

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


#endif //TEST_NCNN2_COMMON_H
