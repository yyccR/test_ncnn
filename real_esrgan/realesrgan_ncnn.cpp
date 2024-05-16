
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "realesrgan.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


void test_realesrgan_ncnn() {
    std::string param = "/Users/yang/CLionProjects/test_ncnn2/real_esrgan/esrgan.ncnn.param";
    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/real_esrgan/esrgan.ncnn.bin";
    std::string imgFile = "/Users/yang/CLionProjects/test_ncnn2/data/cat.png";
    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/cat_esrgan_sr.jpg";
    auto realesrgan = new RealESRGAN(-1, false);
    int res = realesrgan->load(param, bin);
    realesrgan->scale = 4;
    realesrgan->tilesize = 200;
    realesrgan->prepadding = 10;
    std::cout << "load res: " << res << std::endl;
    cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
//    cv::Mat image;
//    imOrigin.convertTo(image, CV_32FC3);
    int w = image.cols;
    int h = image.rows;
    int c = image.channels();
//    std::cout << "w: " << w << " h: " << h << " c: " << c << std::endl;
//    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
//    ncnn::Mat out = ncnn::Mat(realesrgan->scale * w,realesrgan->scale * h, (size_t)c, c);;
//    realesrgan->process(in,out);
//    stbi_write_jpg(outFile.c_str(),w, h, c, (const unsigned char*)in.data,100);

//    cv::Mat cv_img(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3);
//    out.to_pixels(cv_img.data, ncnn::Mat::PIXEL_RGB);
////    // 如果需要转换为BGR格式以供OpenCV处理
//    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
//    // 保存图像
//    cv::imwrite(outFile, cv_img);

//    cv::Mat cv_mat(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3, out.data);
//    cv::Mat cv_mat(h, w, CV_32FC3, in.data);
//    cv::Mat cv_mat(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3);
//    cv::Mat cv_mat(h, w, CV_32FC3);
//    out.to_pixels(cv_mat.data, ncnn::Mat::PIXEL_RGB);
//    cv::imwrite(outFile, cv_mat);
    ncnn::Mat in = ncnn::Mat(w, h, (void*)image.data, (size_t)c, c);
    ncnn::Mat out = ncnn::Mat(realesrgan->scale * w,realesrgan->scale * h, (size_t)c, c);
    auto start = std::chrono::high_resolution_clock::now();
    realesrgan->process(in,out);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "代码执行耗时: " << elapsed.count() << " ms" << std::endl;
    cv::Mat cv_mat(realesrgan->scale * image.rows, realesrgan->scale * image.cols, CV_8UC3, out.data);
    cv::imwrite(outFile, cv_mat);

    return;
}