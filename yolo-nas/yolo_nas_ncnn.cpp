

#include "iostream"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <float.h>
#include <vector>
#include <fstream>

#include "../common/common.h"

ncnn::Net yolo_nas_ncnn_net;
std::string yolo_nas_ncnn_in_blob;
std::string yolo_nas_ncnn_out_blob;
std::string yolo_nas_ncnn_out1_blob;
std::string yolo_nas_ncnn_out2_blob;
std::string yolo_nas_ncnn_out3_blob;
std::string yolo_nas_ncnn_seg_blob;
int target_size = 640;
float prob_threshold = 0.50;
float nms_threshold  = 0.45;
std::vector<std::string> class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
};

void ncnn_clear() {
    yolo_nas_ncnn_net.clear();
}

int load(const std::string& bin, const std::string& param) {
    if (yolo_nas_ncnn_net.load_param(param.c_str())){
        return -1;
    }
    if (yolo_nas_ncnn_net.load_model(bin.c_str())){
        return -1;
    }
    return 0;
}

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& box_pred, const ncnn::Mat& cls_pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = cls_pred.row(i);

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = *(scores+k);
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        if (score >= prob_threshold)
        {

            const float *dis_after_sm = box_pred.row(i);
            float x1 = (-(dis_after_sm[0]) + grid_strides[i].grid0) * grid_strides[i].stride;
            float y1 = (-(dis_after_sm[1])  + grid_strides[i].grid1) * grid_strides[i].stride;
            float x2 = (dis_after_sm[2]  + grid_strides[i].grid0) * grid_strides[i].stride;
            float y2 = (dis_after_sm[3]  + grid_strides[i].grid1) * grid_strides[i].stride;
            std::cout << dis_after_sm[0] << " " << dis_after_sm[1] << " " << dis_after_sm[2] << " " << dis_after_sm[3] << std::endl;
            std::cout << x1<< " " << y1 << " " << x2 << " " << y2 << std::endl;
            std::cout << "---" << std::endl;

//            pred_ltrb[0] = dis_after_sm[0];
//            pred_ltrb[1] = dis_after_sm[1];
//            pred_ltrb[2] = dis_after_sm[2];
//            pred_ltrb[3] = dis_after_sm[3];



            Object obj;
            obj.rect.x = (x2 + x1) / 2;
            obj.rect.y = (y2 + y1) / 2;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            obj.label = label;
            obj.prob = score;
//            obj.mask_feat.resize(32);
//            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            objects.push_back(obj);
        }
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0 + 0.5;
                gs.grid1 = g1 + 0.5;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}


void get_blob_name(std::string in, std::string out, std::string out1, std::string out2, std::string out3, std::string seg){
    yolo_nas_ncnn_in_blob = in;
    yolo_nas_ncnn_out_blob = out;
    yolo_nas_ncnn_out1_blob = out1;
    yolo_nas_ncnn_out2_blob = out2;
    yolo_nas_ncnn_out3_blob = out3;
    yolo_nas_ncnn_seg_blob = seg;
}

int detect(const cv::Mat& bgr, std::vector<Object>& objects) {
    // load image, resize and pad to 640x640
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    std::cout << "in: " << in.w << " " << in.h << " " <<  in.d << " " <<  in.c << " " << scale << " " << std::endl;

    // pad to target_size rectangle
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    //inference
    ncnn::Extractor ex = yolo_nas_ncnn_net.create_extractor();
    ex.input(yolo_nas_ncnn_in_blob.c_str(), in_pad);
    ncnn::Mat box_out;
    ex.extract(yolo_nas_ncnn_out_blob.c_str(), box_out);
    ncnn::Mat cls_out;
    ex.extract(yolo_nas_ncnn_out1_blob.c_str(), cls_out);
    std::cout << "in_pad: " << in_pad.w << " " << in_pad.h << " " <<  in_pad.d << " " <<  in_pad.c << std::endl;
    std::cout << "box_out: " << box_out.w << " " << box_out.h << " " <<  box_out.d << " " <<  box_out.c << std::endl;
    std::cout << "cls_out: " << cls_out.w << " " << cls_out.h << " " <<  cls_out.d << " " <<  cls_out.c << std::endl;

//    ncnn::Mat mask_proto;
//    ex.extract(yolo_nas_seg_ncnn_seg_blob.c_str(), mask_proto);
//    std::cout << "mask_proto: " << mask_proto.w << " " << mask_proto.h << " " <<  mask_proto.d << " " <<  mask_proto.c << std::endl;

    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::cout << "grids_and_stride: " << grid_strides.size() << std::endl;

//    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(grid_strides, box_out, cls_out, prob_threshold, objects8);
    std::cout << "objects8: " << objects8.size() << std::endl;

//    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
//    std::cout << "proposals: " << proposals.size() << std::endl;


    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(objects8);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(objects8, picked, nms_threshold);
    std::cout << "nms picked: " << picked.size() << std::endl;


    // collect final result after nms
    const int count = picked.size();

//    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
//    for (int i = 0; i < count; i++) {
//        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
//    }

//    ncnn::Mat mask_pred_result;
//    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);
//    std::cout << "mask_pred_result: " << mask_pred_result.w << " " << mask_pred_result.h << std::endl;


    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = objects8[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

//        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
//        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
//        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}

//int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects) {
//    // load image, resize and letterbox pad to multiple of MAX_STRIDE
//    int img_w = bgr.cols;
//    int img_h = bgr.rows;
//
//    // letterbox pad to multiple of MAX_STRIDE
//    int w = img_w;
//    int h = img_h;
//    float scale = 1.f;
//    if (w > h) {
//        scale = (float)target_size / w;
//        w = target_size;
//        h = h * scale;
//    }
//    else {
//        scale = (float)target_size / h;
//        h = target_size;
//        w = w * scale;
//    }
//
//    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
//
//    // pad to target_size rectangle
//    // yolov5/utils/datasets.py letterbox
//    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
//    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
//    ncnn::Mat in_pad;
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
//
//    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
//    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
//    in_pad.substract_mean_normalize(0, norm_vals);
//
//    // yolov5 model inference
//    ncnn::Extractor ex = yolo_nas_seg_ncnn_net.create_extractor();
//    ex.input(yolo_nas_seg_ncnn_in_blob.c_str(), in_pad);
//
//    ncnn::Mat out0;
//    ncnn::Mat out1;
//    ncnn::Mat out2;
//    ex.extract(yolo_nas_seg_ncnn_out1_blob.c_str(), out0);
//    ex.extract(yolo_nas_seg_ncnn_out2_blob.c_str(), out1);
//    ex.extract(yolo_nas_seg_ncnn_out3_blob.c_str(), out2);
//    /*
//    The out blob would be a 3-dim tensor with w=dynamic h=dynamic c=255=85*3
//    We view it as [grid_w,grid_h,85,3] for 3 anchor ratio types
//
//                |<--   dynamic anchor grids     -->|
//                |   larger image yields more grids |
//                +-------------------------- // ----+
//               /| center-x                         |
//              / | center-y                         |
//             /  | box-w                            |
//     anchor-0   | box-h                            |
//      +-----+   | box score(1)                     |
//      |     |   +----------------                  |
//      |     |   | per-class scores(80)             |
//      +-----+\  |   .                              |
//              \ |   .                              |
//               \|   .                              |
//                +-------------------------- // ----+
//               /| center-x                         |
//              / | center-y                         |
//             /  | box-w                            |
//     anchor-1   | box-h                            |
//      +-----+   | box score(1)                     |
//      |     |   +----------------                  |
//      +-----+   | per-class scores(80)             |
//             \  |   .                              |
//              \ |   .                              |
//               \|   .                              |
//                +-------------------------- // ----+
//               /| center-x                         |
//              / | center-y                         |
//             /  | box-w                            |
//     anchor-2   | box-h                            |
//      +--+      | box score(1)                     |
//      |  |      +----------------                  |
//      |  |      | per-class scores(80)             |
//      +--+   \  |   .                              |
//              \ |   .                              |
//               \|   .                              |
//                +-------------------------- // ----+
//    */
//
//    ncnn::Mat mask_proto;
//    ex.extract(yolo_nas_seg_ncnn_seg_blob.c_str(), mask_proto);
//
//    std::vector<Object> proposals;
//
//    // anchor setting from yolov5/models/yolov5s.yaml
//
//    // stride 8
//    {
//        ncnn::Mat anchors(6);
//        anchors[0] = 10.f;
//        anchors[1] = 13.f;
//        anchors[2] = 16.f;
//        anchors[3] = 30.f;
//        anchors[4] = 33.f;
//        anchors[5] = 23.f;
//
//        std::vector<Object> objects;
//        generate_proposals(anchors, 8, in_pad, out0, prob_threshold, objects);
//
//        proposals.insert(proposals.end(), objects.begin(), objects.end());
//    }
//
//    // stride 16
//    {
//        ncnn::Mat anchors(6);
//        anchors[0] = 30.f;
//        anchors[1] = 61.f;
//        anchors[2] = 62.f;
//        anchors[3] = 45.f;
//        anchors[4] = 59.f;
//        anchors[5] = 119.f;
//
//        std::vector<Object> objects;
//        generate_proposals(anchors, 16, in_pad, out1, prob_threshold, objects);
//
//        proposals.insert(proposals.end(), objects.begin(), objects.end());
//    }
//
//    // stride 32
//    {
//        ncnn::Mat anchors(6);
//        anchors[0] = 116.f;
//        anchors[1] = 90.f;
//        anchors[2] = 156.f;
//        anchors[3] = 198.f;
//        anchors[4] = 373.f;
//        anchors[5] = 326.f;
//
//        std::vector<Object> objects;
//        generate_proposals(anchors, 32, in_pad, out2, prob_threshold, objects);
//
//        proposals.insert(proposals.end(), objects.begin(), objects.end());
//    }
//
//    // sort all proposals by score from highest to lowest
//    qsort_descent_inplace(proposals);
//
//    // apply nms with nms_threshold
//    std::vector<int> picked;
//    nms_sorted_bboxes(proposals, picked, nms_threshold);
//
//    // collect final result after nms
//    int count = picked.size();
//
//    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
//    for (int i = 0; i < count; i++) {
//        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
//    }
//
//    ncnn::Mat mask_pred_result;
//    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);
//
//    objects.resize(count);
//    for (int i = 0; i < count; i++) {
//        objects[i] = proposals[picked[i]];
//
//        // adjust offset to original unpadded
//        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
//        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
//        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
//        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
//
//        // clip
//        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
//
//        objects[i].rect.x = x0;
//        objects[i].rect.y = y0;
//        objects[i].rect.width = x1 - x0;
//        objects[i].rect.height = y1 - y0;
//
//        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
//        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
//        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
//    }
//
//    return 0;
//}

void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int mode) {
    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f (%s)\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, class_names[obj.label].c_str());

        if(mode == 0)
            color_index = obj.label;
        const unsigned char* color = colors[color_index];
        cv::Scalar cc(color[0], color[1], color[2]);
        if(mode == 1)
            color_index++;

//        draw_segment(bgr, obj.cv_mask, color);

        cv::rectangle(bgr, obj.rect, cc, 1);

        std::string text = class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%";

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}



//void image(cv::Mat in, std::string outputPath) {
//    std::vector<Object> objects;
//    if (dynamic)
//        detect_dynamic(in, objects);
//    else
//        detect(in, objects);
//    cv::Mat out = in.clone();
//
//    draw_objects(out, objects, 1);
//    cv::imshow("Detect", out);
//    cv::waitKey();
//
//    if (save) {
//        cv::imwrite(outputPath, out);
//        std::cout << "\nOutput saved at " << outputPath;
//    }
//}

//void video(cv::VideoCapture capture) {
//    if (capture.isOpened()) {
//        std::cout << "Object Detection Started...." << std::endl;
//
//        cv::Mat frame, out;
//        std::vector<Object> objects;
//
//        do {
//            capture >> frame; //extract frame by frame
//            if (dynamic)
//                detect_dynamic(frame, objects);
//            else
//                detect(frame, objects);
//            out = frame.clone();
//            draw_objects(out, objects, 0);
//            cv::imshow("Detect", out);
//
//            char key = (char)cv::pollKey();
//
//            if (key == 27 || key == 'q' || key == 'Q') // Press q or esc to exit from window
//                break;
//        } while (!frame.empty());
//    }
//    else {
//        std::cout << "Could not Open Camera/Video";
//    }
//}


void test_yolo_nas_ncnn() {
    std::cout << "yolo-nas detecting." << std::endl;
    std::string image_file("/Users/yang/CLionProjects/test_ncnn/data/traffic_road.jpg");
    std::string param_file("/Users/yang/CLionProjects/test_ncnn/yolo-nas/yolo-nas-s.ncnn.param");
    std::string bin_file("/Users/yang/CLionProjects/test_ncnn/yolo-nas/yolo-nas-s.ncnn.bin");
//
    int res = load(bin_file, param_file);
    std::cout << "init res: " << res << std::endl;
    cv::Mat image = cv::imread(image_file, 1);
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 640, 640);
//    ncnn::Extractor ex = yolo_nas_ncnn_net.create_extractor();
//    ex.input("in0", input);
//    for(auto &in_name : yolo_nas_ncnn_net.input_names()) {
//        ncnn::Mat in;
//        ex.extract(in_name, in);
//        std::cout << in_name << ": " << in.w << " " << in.h << " " << in.d << " " << in.c << std::endl;
//
//    }
//    for(auto &out_name : yolo_nas_ncnn_net.output_names()) {
//        ncnn::Mat out;
//        ex.extract(out_name, out);
//        std::cout << out_name << ": " << out.w << " " << out.h << " " << out.d << " " << out.c << std::endl;
//
//    }

    get_blob_name("in0","out0","out1","out2","out3","out1");
    std::vector<Object> objects;
    detect(image, objects);
    draw_objects(image, objects, 1);

    cv::imshow("a", image);
    cv::waitKey(0);
//    cv::imwrite("../data/traffic_road_detect_v8.jpg", image);



}