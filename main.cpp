#include <iostream>

//#alsa "yolov5-seg/yolov5_seg_ncnn.cpp"
//#alsa "yolov8-seg/yolov8_seg_ncnn.cpp"
//#alsa "yolov8-pose/yolov8_pose_ncnn.cpp"
//#alsa "yolov8/yolov8_ncnn.cpp"
//#alsa "yolo-nas/yolo_nas_ncnn.cpp"
//#include "sherpa/sherpa_ncnn.cpp"
#include "realsr/realsr_ncnn.cpp"

int main() {
//    typedef struct DetectResults {
//        float score;
//        int label;
//        int* box_pred;
//        float* mask_pred;
//    } DetectResults;
//    int a[] = {1,2,3,4} ;
//    int*b = a;
//    for(int i = 0; i<4; i++){
//        std::cout << b+i << " " << *(b+i) << std::endl;
//    }
//    test_yolov5_seg_ncnn();
//    test_yolov8_seg_ncnn();
//    test_yolov8_pose_ncnn();
//    test_yolov8_ncnn();
//    test_yolo_nas_ncnn();
//    test_sherpa_ncnn();
    test_realsr_ncnn();

    return 0;
}
