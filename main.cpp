#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov8/yolov8_ncnn.cpp"
#include "byte_tracker/include/byte_tracker.h"

// #include "yolov5-seg/yolov5_seg_ncnn.cpp"
//#include "yolov5_v60_v61_v62_v70/yolov5_ncnn.cpp"
// #include "yolov8-seg/yolov8_seg_ncnn.cpp"
//#include "yolov8-pose/yolov8_pose_ncnn.cpp"
//#include "yolov8-pose/yolov8_pose2_ncnn.cpp"
//#include "yolov8/yolov8_ncnn.cpp"
// #include "yolov8-pose/yolov8_pose_with_post_process.cpp"
//#include "yolo-nas/yolo_nas_ncnn.cpp"
//#include "sherpa/sherpa_ncnn.cpp"
//#include "realsr/realsr_ncnn.cpp"
//#include "real_esrgan/realesrgan_ncnn.cpp"
// #include "yolov8_obb/yolov8_obb_with_post_process_ncnn.cpp"
// #include "yolov8-seg/yolov8_seg_with_post_process_ncnn.cpp"
//#include "yolov8-cls/yolov8_cls_ncnn.cpp"
//#include "yolov11/yolov11_ncn.cpp";
// #include "yolov11-pose/yolov11_pose_with_post_process.cpp";
// #include "yolov11-seg/yolov11_seg_ncnn.cpp";
// #include "yolov11-obb/yolov11_obb_ncnn.cpp";
// #include "yolov11-cls/yolov11_cls_ncnn.cpp"
//#include "ppocrv5/ppocrv5.cpp"
//#include "yolov8-worldv2/yolov8_worldv2.cpp"
// #include "yoloe/yoloe_seg_ncnn.cpp" // 已删除yoloe目录，注释掉

// COCO类别名
static const char* coco_labels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main() {
    // 加载yolov8模型
    std::string param_file = "yolov8/yolov8s_ncnn.param";
//    std::string param_file = "yolov8/yolov8n.withPostProcess.ncnn.param";
    std::string bin_file = "yolov8/yolov8s_ncnn.bin";
//    std::string bin_file = "yolov8/yolov8n.withPostProcess.ncnn.bin";
    int res = load(bin_file, param_file);
    if (res != 0) {
        std::cerr << "Failed to load yolov8 model!" << std::endl;
        return -1;
    }
    get_blob_name("in0", "216", "", "", "", ""); // 注意：输出节点名需根据实际模型调整

    // 初始化BYTETracker
    byte_tracker::BYTETracker tracker(30, 0.3f, 0.1f, 0.8f, 20);

    // 尝试多种方式初始化摄像头
    cv::VideoCapture cap;
    cap.open("data/video/track2.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: data/video/track1.mp4!" << std::endl;
        return -1;
    }

    int frame_id = 0;
    // 颜色表
    std::vector<cv::Scalar> track_colors = {
        {244,  67,  54}, {233,  30,  99}, {156,  39, 176}, {103,  58, 183},
        { 63,  81, 181}, { 33, 150, 243}, {  3, 169, 244}, {  0, 188, 212},
        {  0, 150, 136}, { 76, 175,  80}, {139, 195,  74}, {205, 220,  57},
        {255, 235,  59}, {255, 193,   7}, {255, 152,   0}, {255,  87,  34},
        {121,  85,  72}, {158, 158, 158}, { 96, 125, 139}
    };
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frame_id++;
        std::cout << "\n[FRAME] " << frame_id << std::endl;

        // 只对检测做旋转和镜像，展示原始画面
        cv::Mat frame_for_detect = frame.clone();
        // 已去除旋转和翻转操作

        // 1. yolov8检测
        std::vector<common::Object> objects;
        detect(frame_for_detect, objects);
        std::cout << "[DEBUG] objects.size() = " << objects.size() << std::endl;
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto& obj = objects[i];
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            std::cout << "  [DETECT] " << i << ": class=" << obj.label << " (" << (obj.label >= 0 && obj.label < 80 ? coco_labels[obj.label] : "unknown") << ") prob=" << obj.prob
                      << " rect=[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
        }

        // 2. 转换为byte_tracker输入格式（frame_for_detect坐标系）
        // 将检测到的目标转换为BYTETracker所需的输入格式：
        // detections为每个目标的[x, y, w, h]，scores为置信度，classes为类别
        std::vector<std::vector<float>> detections;
        std::vector<float> scores;
        std::vector<int> classes;
        for (const auto& obj : objects) {
            float x1 = obj.rect.x;
            float y1 = obj.rect.y;
            float x2 = obj.rect.x + obj.rect.width;
            float y2 = obj.rect.y + obj.rect.height;
            // BYTETracker要求输入为[x, y, w, h]，其中x, y为左上角坐标，w, h为宽高
            std::vector<float> det = {x1, y1, obj.rect.width, obj.rect.height};
            detections.push_back(det); // 存储每个目标的位置信息
            scores.push_back(obj.prob); // 存储每个目标的置信度
            classes.push_back(obj.label); // 存储每个目标的类别
        }

        // 3. 跟踪（frame_for_detect坐标系）
        // 调用BYTETracker进行多目标跟踪，返回track_results
        // track_results的每一项为: [x1, y1, x2, y2, track_id, score, class]
        auto track_results = tracker.update(detections, scores, classes, frame.cols, frame.rows);
        std::cout << "[DEBUG] track_results.size() = " << track_results.size() << std::endl;
        for (size_t i = 0; i < track_results.size(); ++i) {
            const auto& tr = track_results[i];
            int track_id = static_cast<int>(tr[4]); // 跟踪ID
            int cls = static_cast<int>(tr[6]);      // 目标类别
            // 输出跟踪结果信息
            std::cout << "  [TRACK] " << i << ": id=" << track_id << " class=" << cls << " (" << (cls >= 0 && cls < 80 ? coco_labels[cls] : "unknown") << ")"
                      << " score=" << tr[5] << " rect=[" << tr[0] << "," << tr[1] << "," << tr[2] << "," << tr[3] << "]" << std::endl;
        }

        // 4. 绘制检测框和track id（直接用BYTETracker输出的rect，不做反变换）
        // 遍历所有跟踪结果，画出检测框和track id、类别、置信度
        int img_w = frame.cols;
        int img_h = frame.rows;
        for (const auto& tr : track_results) {
            float x1 = std::max(0.f, std::min(tr[0], float(img_w - 1)));
            float y1 = std::max(0.f, std::min(tr[1], float(img_h - 1)));
            float x2 = std::max(0.f, std::min(tr[2], float(img_w - 1)));
            float y2 = std::max(0.f, std::min(tr[3], float(img_h - 1)));
            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2)); // 目标框
            int track_id = static_cast<int>(tr[4]);              // 跟踪ID
            float score = tr[5];                                 // 置信度
            int cls = static_cast<int>(tr[6]);                   // 类别
            cv::Scalar color = track_colors[track_id % track_colors.size()]; // 根据track_id选择颜色
            cv::rectangle(frame, rect, color, 2); // 画框
            std::string label = std::to_string(track_id) + ": ";
            if (cls >= 0 && cls < 80) label += coco_labels[cls];
            else label += "unknown";
            label += cv::format(" %.2f", score); // 拼接类别和置信度
            cv::putText(frame, label, cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2); // 画标签

            // 画轨迹：遍历tracker.tracked_stracks_，对每个激活track画轨迹
            for (const auto& track_ptr : tracker.get_tracked_stracks()) {
                if (!track_ptr->is_activated()) continue;
                const auto& traj = track_ptr->get_trajectory();
                if (traj.size() > 1) {
                    std::vector<cv::Point> pts;
                    for (const auto& pt : traj) pts.emplace_back(pt[0], pt[1]);
                    cv::polylines(frame, pts, false, track_colors[track_ptr->get_track_id() % track_colors.size()], 2);
                }
            }
        }

        cv::imshow("YOLOv8 + BYTETracker", frame);
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
