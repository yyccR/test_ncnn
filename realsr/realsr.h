
#ifndef REALSR_H
#define REALSR_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class RealSR
{
public:
    RealSR(int gpuid, bool tta_mode = false, int num_threads = 1);
    ~RealSR();

    int load(const std::string& parampath, const std::string& modelpath);

    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

    int process_cpu(const ncnn::Mat& inimage, ncnn::Mat& outimage) const;

public:
    // realsr parameters
//    int num_threads = num_threads;
    int scale = 4;
    int tilesize = 100;
    int prepadding = 10;

private:
    ncnn::VulkanDevice* vkdev;
//    int vkdev;
    ncnn::Net net;
//    std::vector<std::shared_ptr<ncnn::Net>> nets;
    ncnn::Pipeline* realsr_preproc;
    ncnn::Pipeline* realsr_postproc;
    ncnn::Layer* bicubic_4x;
    bool tta_mode;
};

#endif // REALSR_H
