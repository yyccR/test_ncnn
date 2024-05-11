
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "realsr.h"
#include "stb_image_write.h"


void test_realsr_ncnn() {
    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.param";
    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.bin";
    std::string imgFile = "/Users/yang/CLionProjects/test_ncnn2/data/cat.png";
    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/cat_sr.jpg";
    auto realSr = new RealSR(-1, false,4);
    int res = realSr->load(param, bin);
    realSr->scale = 4;
    realSr->tilesize = 20;
    realSr->prepadding = 10;
    std::cout << "load res: " << res << std::endl;
    cv::Mat image = cv::imread(imgFile, cv::IMREAD_UNCHANGED);
    std::cout << "image.channel: " << image.channels() << std::endl;
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    ncnn::Mat out = ncnn::Mat(realSr->scale * image.cols,realSr->scale * image.rows, (size_t)3, 3);;
    realSr->process(in,out);

//    cv::Mat cv_img(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3);
//    out.to_pixels(cv_img.data, ncnn::Mat::PIXEL_RGB);
////    // 如果需要转换为BGR格式以供OpenCV处理
//    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
//    // 保存图像
//    cv::imwrite(outFile, cv_img);
    stbi_write_jpg(outFile.c_str(),realSr->scale * image.cols, realSr->scale * image.rows, out.elempack, (const unsigned char*)out.data,100);
    return;
}