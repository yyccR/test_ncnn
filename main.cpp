#include <iostream>

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
#include "ppocrv5/ppocrv5.cpp"





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
    // test_yolov5_seg_ncnn();
    // test_yolov8_seg_ncnn();
    // test_yolov8_obb_with_post_process_ncnn();
//    test_yolov8_cls();
//    test_yolov8_pose_ncnn();
//    test_yolov8_ncnn();
//     test_yolov8_pose_with_post_process_ncnn();
//    test_yolo_nas_ncnn();
//    test_sherpa_ncnn();
//    test_realsr_ncnn();
//    test_realesrgan_ncnn();
//    test_yolov8_pose2_ncnn();
//    test_yolov5_v60_v61_v62_v70_ncnn();
//    test_yolov11_ncnn();
    // test_yolov11_pose_with_post_process_ncnn();

    // test_yolov11_seg_with_post_process_ncnn();
    // test_yolov11_obb_with_post_process_ncnn();
    // test_yolov11_cls();
    test_ppocrv5_ncnn();

    return 0;
}
