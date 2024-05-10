
#include <string>
#include "realsr.h"


void test_realsr_ncnn() {
    std::string param = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.param";
    std::string bin = "/Users/yang/CLionProjects/test_ncnn2/realsr/df2k.ncnn.bin";
    auto realSr = RealSR(-1, true,1);
//    realSr->load(param, bin);
    return;
}