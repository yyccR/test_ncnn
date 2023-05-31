

#include "common.h"


const unsigned char colors[81][3] = {
        {56,  0,   255},
        {226, 255, 0},
        {0,   94,  255},
        {0,   37,  255},
        {0,   255, 94},
        {255, 226, 0},
        {0,   18,  255},
        {255, 151, 0},
        {170, 0,   255},
        {0,   255, 56},
        {255, 0,   75},
        {0,   75,  255},
        {0,   255, 169},
        {255, 0,   207},
        {75,  255, 0},
        {207, 0,   255},
        {37,  0,   255},
        {0,   207, 255},
        {94,  0,   255},
        {0,   255, 113},
        {255, 18,  0},
        {255, 0,   56},
        {18,  0,   255},
        {0,   255, 226},
        {170, 255, 0},
        {255, 0,   245},
        {151, 255, 0},
        {132, 255, 0},
        {75,  0,   255},
        {151, 0,   255},
        {0,   151, 255},
        {132, 0,   255},
        {0,   255, 245},
        {255, 132, 0},
        {226, 0,   255},
        {255, 37,  0},
        {207, 255, 0},
        {0,   255, 207},
        {94,  255, 0},
        {0,   226, 255},
        {56,  255, 0},
        {255, 94,  0},
        {255, 113, 0},
        {0,   132, 255},
        {255, 0,   132},
        {255, 170, 0},
        {255, 0,   188},
        {113, 255, 0},
        {245, 0,   255},
        {113, 0,   255},
        {255, 188, 0},
        {0,   113, 255},
        {255, 0,   0},
        {0,   56,  255},
        {255, 0,   113},
        {0,   255, 188},
        {255, 0,   94},
        {255, 0,   18},
        {18,  255, 0},
        {0,   255, 132},
        {0,   188, 255},
        {0,   245, 255},
        {0,   169, 255},
        {37,  255, 0},
        {255, 0,   151},
        {188, 0,   255},
        {0,   255, 37},
        {0,   255, 0},
        {255, 0,   170},
        {255, 0,   37},
        {255, 75,  0},
        {0,   0,   255},
        {255, 207, 0},
        {255, 0,   226},
        {255, 245, 0},
        {188, 255, 0},
        {0,   255, 18},
        {0,   255, 75},
        {0,   255, 151},
        {255, 56,  0},
        {245, 255, 0}
};

void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color) {
    for (int y = 0; y < bgr.rows; y++) {
        uchar* image_ptr = bgr.ptr(y);
        const float* mask_ptr = mask.ptr<float>(y);
        for (int x = 0; x < bgr.cols; x++) {
            if (mask_ptr[x] >= 0.5) {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
            }
            image_ptr += 3;
        }
    }
}

void draw_pose(cv::Mat& bgr, std::vector<cv::Point3f> key_points){
    for(auto& kp: key_points){
        if(kp.z > 0.5){
            cv::circle(bgr, cv::Point(kp.x,kp.y), 4, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        }
    }
    cv::Point neck((key_points[5].x + key_points[6].x)/2.0, (key_points[5].y + key_points[6].y)/2.0);

    if(key_points[0].z > 0.5 && key_points[1].z > 0.5){
        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), cv::Point(key_points[1].x, key_points[1].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[0].z > 0.5 && key_points[2].z > 0.5){
        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), cv::Point(key_points[2].x, key_points[2].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[1].z > 0.5 && key_points[3].z > 0.5){
        cv::line(bgr, cv::Point(key_points[1].x, key_points[1].y), cv::Point(key_points[3].x, key_points[3].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[2].z > 0.5 && key_points[4].z > 0.5){
        cv::line(bgr, cv::Point(key_points[2].x, key_points[2].y), cv::Point(key_points[4].x, key_points[4].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[2].z > 0.5 && key_points[4].z > 0.5){
        cv::line(bgr, cv::Point(key_points[2].x, key_points[2].y), cv::Point(key_points[4].x, key_points[4].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[5].z > 0.5 && key_points[6].z > 0.5){

        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), neck, cv::Scalar(127,255,0), 2);
        cv::line(bgr, neck, cv::Point(key_points[5].x, key_points[5].y), cv::Scalar(127,255,0), 2);
        cv::line(bgr, neck, cv::Point(key_points[6].x, key_points[6].y), cv::Scalar(127,255,0), 2);

        if(key_points[11].z > 0.5){
            cv::line(bgr, neck, cv::Point(key_points[11].x, key_points[11].y), cv::Scalar(127,255,0), 2);
        }
        if(key_points[12].z > 0.5){
            cv::line(bgr, neck, cv::Point(key_points[12].x, key_points[12].y), cv::Scalar(127,255,0), 2);
        }
    }

    if(key_points[5].z > 0.5 && key_points[7].z > 0.5){
        cv::line(bgr, cv::Point(key_points[5].x, key_points[5].y), cv::Point(key_points[7].x, key_points[7].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[7].z > 0.5 && key_points[9].z > 0.5){
        cv::line(bgr, cv::Point(key_points[7].x, key_points[7].y), cv::Point(key_points[9].x, key_points[9].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[6].z > 0.5 && key_points[8].z > 0.5){
        cv::line(bgr, cv::Point(key_points[6].x, key_points[6].y), cv::Point(key_points[8].x, key_points[8].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[8].z > 0.5 && key_points[10].z > 0.5){
        cv::line(bgr, cv::Point(key_points[8].x, key_points[8].y), cv::Point(key_points[10].x, key_points[10].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[8].z > 0.5 && key_points[10].z > 0.5){
        cv::line(bgr, cv::Point(key_points[8].x, key_points[8].y), cv::Point(key_points[10].x, key_points[10].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[11].z > 0.5 && key_points[13].z > 0.5){
        cv::line(bgr, cv::Point(key_points[11].x, key_points[11].y), cv::Point(key_points[13].x, key_points[13].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[13].z > 0.5 && key_points[15].z > 0.5){
        cv::line(bgr, cv::Point(key_points[13].x, key_points[13].y), cv::Point(key_points[15].x, key_points[15].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[12].z > 0.5 && key_points[14].z > 0.5){
        cv::line(bgr, cv::Point(key_points[12].x, key_points[12].y), cv::Point(key_points[14].x, key_points[14].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[14].z > 0.5 && key_points[16].z > 0.5){
        cv::line(bgr, cv::Point(key_points[14].x, key_points[14].y), cv::Point(key_points[16].x, key_points[16].y), cv::Scalar(127,255,0), 2);
    }
}

void matPrint(const ncnn::Mat& m){
    for (int q = 0; q < m.c; q++){
        const float* ptr = m.channel(q);
        for (int z = 0; z < m.d; z++){
            for (int y = 0; y < m.h; y++){
                for (int x = 0; x < m.w; x++){
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void matVisualize(const char* title, const ncnn::Mat& m, bool save) {
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i = 0; i < m.c; i++){
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < m.h; y++){
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < m.w; x++){
                float v = tp[x];
                if (v != v){
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }
                sp += 3;
            }
        }
        if (!save) {
            cv::imshow(title, normed_feats[i]);
            cv::waitKey();
        }
    }

    if (save) {
        int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
        int th = (m.c - 1) / tw + 1;

        cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
        show_map = cv::Scalar(127);

        // tile
        for (int i = 0; i < m.c; i++)
        {
            int ty = i / tw;
            int tx = i % tw;

            normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
        }
        cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
        cv::imshow(title, show_map);
        cv::waitKey();
        cv::imwrite(title, show_map);
    }
}

void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1);// order_type

    op->load_param(pd);
    op->create_pipeline(opt);
    // forward
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

void softmax(ncnn::Mat& bottom) {

    ncnn::Layer* sfm = ncnn::create_layer("Softmax");

    ncnn::ParamDict pd;
    pd.set(0, 1); // axis
    pd.set(1, 1);
    sfm->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    sfm->create_pipeline(opt);
    sfm->forward_inplace(bottom, opt);

    sfm->destroy_pipeline(opt);

    delete sfm;
}


void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);  //start
    pd.set(10, ends);   //end
    pd.set(11, axes);   //axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);       // resize_type
    pd.set(1, scale);   // height_scale
    pd.set(2, scale);   // width_scale
    pd.set(3, out_h);   // height
    pd.set(4, out_w);   // width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);           //start
    pd.set(1, h);           //end
    if (d > 0)
        pd.set(11, d);      //axes
    pd.set(2, c);           //axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
void sigmoid(ncnn::Mat& bottom) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward
    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}
void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob) {
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    op->destroy_pipeline(opt);

    delete op;
}

void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                 const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                 ncnn::Mat& mask_pred_result){
    ncnn::Mat masks;
    ncnn::Mat reshape_proto = mask_proto.reshape(mask_proto.w*mask_proto.h,mask_proto.c);

    matmul(std::vector<ncnn::Mat>{mask_feat, reshape_proto}, masks);
//    std::cout << "--" << reshape_proto.w << " " << reshape_proto.h  << " " <<  reshape_proto.d  << " " <<reshape_proto.c << std::endl;
//    std::cout << "--" << mask_feat.w << " " << mask_feat.h  << " " <<  mask_feat.d  << " " <<mask_feat.c << std::endl;
//    std::cout << "--" << mask_proto.w << " " << mask_proto.h << " " << mask_proto.d << " " <<  mask_proto.c << std::endl;
//    std::cout << "matmul:" << masks.w << " " << masks.h << " " << masks.d << " " <<   masks.c << std::endl;

    sigmoid(masks);

//    std::cout << "sigmoid:" << masks.w << " " << masks.h << " " <<  masks.d << " " <<   masks.c << std::endl;
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
//    std::cout << "reshape:" << masks.w << " " << masks.h << " " <<  masks.d << " " <<   masks.c << std::endl;
//    std::cout << "reshape:" << masks.h << " " << in_pad.h << " " <<  masks.d << " " <<   in_pad.w << std::endl;
    interp(masks, 4.0, 0, 0, masks);
//    std::cout << "interp:" << masks.w << " " << masks.h << " " <<  masks.d << " " <<   masks.c << std::endl;
    slice(masks, mask_pred_result, wpad / 2, in_pad.w - wpad / 2, 2);
    slice(mask_pred_result, mask_pred_result, hpad / 2, in_pad.h - hpad / 2, 1);
    interp(mask_pred_result, 1.0, img_w, img_h, mask_pred_result);
}

inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}



inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    //return static_cast<float>(1.f / (1.f + exp(-x)));
    return 1.0f / (1.0f + fast_exp(-x));
}

#if PERMUTE
void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid = feat_blob.h;
    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.w - 5 - 32;

    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                float box_score = feat_blob.channel(q).row(i * num_grid_x + j)[4];
                float box_confidence = sigmoid(box_score);
                if (box_confidence >= prob_threshold) {
                    // find class_index with max class_score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = feat_blob.channel(q).row(i * num_grid_x + j)[5 + k];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    // combined score = box score * class score
                    float score = sigmoid(box_score) * sigmoid(class_score); // apply sigmoid first to get normed 0~1 value

                    // filter candidate boxes with combined score >= prob_threshold
                    if (score >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[0]);
                        float dy = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[1]);
                        float dw = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[2]);
                        float dh = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[3]);

                        float cx = (dx * 2.f - 0.5f + j) * stride;  //center x coordinate
                        float cy = (dy * 2.f - 0.5f + i) * stride;  //cennter y coordinate
                        float bw = pow(dw * 2.f, 2) * anchor_w;     //box width
                        float bh = pow(dh * 2.f, 2) * anchor_h;     //box height

                        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
                        float x0 = cx - bw * 0.5f;
                        float y0 = cy - bh * 0.5f;
                        float x1 = cx + bw * 0.5f;
                        float y1 = cy + bh * 0.5f;

                        // collect candidates
                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = score;
                        obj.mask_feat.resize(32);
                        std::copy(feat_blob.channel(q).row(i * num_grid_x + j) + 5 + num_class, feat_blob.channel(q).row(i * num_grid_x + j) + 5 + num_class + 32, obj.mask_feat.begin());

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}
#else
void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.c / num_anchors - 5 - 32;

    const int feat_offset = num_class + 5 + 32;

    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];
                float box_confidence = sigmoid(box_score);
                if (box_confidence >= prob_threshold) {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    // combined score = box score * class score
                    float score = sigmoid(box_score) * sigmoid(class_score); // apply sigmoid first to get normed 0~1 value

                    // filter candidate boxes with combined score >= prob_threshold
                    if (score >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                        float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                        float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                        float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                        float cx = (dx * 2.f - 0.5f + j) * stride;  //center x coordinate
                        float cy = (dy * 2.f - 0.5f + i) * stride;  //cennter y coordinate
                        float bw = pow(dw * 2.f, 2) * anchor_w;     //box width
                        float bh = pow(dh * 2.f, 2) * anchor_h;     //box height

                        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
                        float x0 = cx - bw * 0.5f;
                        float y0 = cy - bh * 0.5f;
                        float x1 = cx + bw * 0.5f;
                        float y1 = cy + bh * 0.5f;

                        // collect candidates
                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = score;
                        for (int t = 0; t < 32; t++) {
                            float val = (float)feat_blob.channel(q * feat_offset + 5 + num_class + t).row(i)[j];
                            //std::cout << val << std::endl;
                            obj.mask_feat.push_back(val);
                        }
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}
#endif