
#include <chrono>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "realsr.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

void test_realsr_ncnn() {
//    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.param";
    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/esrgan_anime.ncnn.param";
//    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/dped.ncnn.param";
//    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/esrgan.ncnn.param";
//    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.bin";
    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/esrgan_anime.ncnn.bin";
//    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/dped.ncnn.bin";
//    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/esrgan.ncnn.bin";
    std::string imgFile = "/Users/yang/CLionProjects/test_ncnn2/data/mxd.png";
//    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/dog_sr.jpg";
//    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/dog_esrgan_anime.jpg";
//    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/dog_dped_sr.jpg";
    std::string outFile = "/Users/yang/CLionProjects/test_ncnn2/data/mxd_esrgan_anime.jpg";
    int num_threads = 4;
    auto realSr = new RealSR(-1, false, num_threads);
    int res = realSr->load(param, bin);
    realSr->scale = 4;
    realSr->tilesize = 400;
    realSr->prepadding = 10;
    std::cout << "load res: " << res << std::endl;
    cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
    int w = image.cols;
    int h = image.rows;
    int c = image.channels();
//    std::cout << "w: " << w << " h: " << h << " c: " << c << std::endl;
//    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
//    ncnn::Mat out = ncnn::Mat(realSr->scale * w,realSr->scale * h, (size_t)c, c);;
//    realSr->process(in,out);
//    stbi_write_jpg(outFile.c_str(),realSr->scale * w, realSr->scale * h, c, (const unsigned char*)out.data,100);
//    cv::Mat cv_img(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3);
//    out.to_pixels(cv_img.data, ncnn::Mat::PIXEL_RGB);
////    // 如果需要转换为BGR格式以供OpenCV处理
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//    // 保存图像
//    cv::imwrite(outFile, cv_img);

//    cv::Mat cv_mat(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3, out.data);
//    cv::Mat cv_mat(h, w, CV_32FC3, in.data);
//    cv::Mat cv_mat(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3);
//    cv::Mat cv_mat(h, w, CV_32FC3);
//    out.to_pixels(cv_mat.data, ncnn::Mat::PIXEL_RGB);
//    cv::imwrite(outFile, cv_mat);

//    unsigned char* pixeldata = 0;
//    int w;
//    int h;
//    int c;
//    FILE* fp = fopen(imgFile.c_str(), "rb");
//    unsigned char* filedata = 0;
//    int length = 0;
//    {
//        fseek(fp, 0, SEEK_END);
//        length = ftell(fp);
//        rewind(fp);
//        filedata = (unsigned char*)malloc(length);
//        if (filedata)
//        {
//            fread(filedata, 1, length, fp);
//        }
//        fclose(fp);
//    }
//    pixeldata = webp_load(filedata, length, &w, &h, &c);
//    pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
//    std::cout << length << " " << w << " " << h << " " << c << std::endl;
//    if (pixeldata)
//    {
//        // stb_image auto channel
//        if (c == 1)
//        {
//            // grayscale -> rgb
//            stbi_image_free(pixeldata);
//            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
//            c = 3;
//        }
//        else if (c == 2)
//        {
//            // grayscale + alpha -> rgba
//            stbi_image_free(pixeldata);
//            pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 4);
//            c = 4;
//        }
//    }
    ncnn::Mat in = ncnn::Mat(w, h, (void*)image.data, (size_t)c, c);
    ncnn::Mat out = ncnn::Mat(realSr->scale * w,realSr->scale * h, (size_t)c, c);
    auto start = std::chrono::high_resolution_clock::now();
    realSr->process(in,out);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "代码执行耗时: " << elapsed.count() << " ms" << std::endl;
    cv::Mat cv_mat(realSr->scale * image.rows, realSr->scale * image.cols, CV_8UC3, out.data);
    cv::imwrite(outFile, cv_mat);
//    stbi_write_jpg(outFile.c_str(),realSr->scale * w, realSr->scale * h, c, (const unsigned char*)out.data,100);

    return;
}