
#include "common.h"

namespace common {

    const int coco_color_list[80][3] =
            {
                    //{255 ,255 ,255}, //bg
                    {216 , 82 , 24},
                    {236 ,176 , 31},
                    {125 , 46 ,141},
                    {118 ,171 , 47},
                    { 76 ,189 ,237},
                    {238 , 19 , 46},
                    { 76 , 76 , 76},
                    {153 ,153 ,153},
                    {255 ,  0 ,  0},
                    {255 ,127 ,  0},
                    {190 ,190 ,  0},
                    {  0 ,255 ,  0},
                    {  0 ,  0 ,255},
                    {170 ,  0 ,255},
                    { 84 , 84 ,  0},
                    { 84 ,170 ,  0},
                    { 84 ,255 ,  0},
                    {170 , 84 ,  0},
                    {170 ,170 ,  0},
                    {170 ,255 ,  0},
                    {255 , 84 ,  0},
                    {255 ,170 ,  0},
                    {255 ,255 ,  0},
                    {  0 , 84 ,127},
                    {  0 ,170 ,127},
                    {  0 ,255 ,127},
                    { 84 ,  0 ,127},
                    { 84 , 84 ,127},
                    { 84 ,170 ,127},
                    { 84 ,255 ,127},
                    {170 ,  0 ,127},
                    {170 , 84 ,127},
                    {170 ,170 ,127},
                    {170 ,255 ,127},
                    {255 ,  0 ,127},
                    {255 , 84 ,127},
                    {255 ,170 ,127},
                    {255 ,255 ,127},
                    {  0 , 84 ,255},
                    {  0 ,170 ,255},
                    {  0 ,255 ,255},
                    { 84 ,  0 ,255},
                    { 84 , 84 ,255},
                    { 84 ,170 ,255},
                    { 84 ,255 ,255},
                    {170 ,  0 ,255},
                    {170 , 84 ,255},
                    {170 ,170 ,255},
                    {170 ,255 ,255},
                    {255 ,  0 ,255},
                    {255 , 84 ,255},
                    {255 ,170 ,255},
                    { 42 ,  0 ,  0},
                    { 84 ,  0 ,  0},
                    {127 ,  0 ,  0},
                    {170 ,  0 ,  0},
                    {212 ,  0 ,  0},
                    {255 ,  0 ,  0},
                    {  0 , 42 ,  0},
                    {  0 , 84 ,  0},
                    {  0 ,127 ,  0},
                    {  0 ,170 ,  0},
                    {  0 ,212 ,  0},
                    {  0 ,255 ,  0},
                    {  0 ,  0 , 42},
                    {  0 ,  0 , 84},
                    {  0 ,  0 ,127},
                    {  0 ,  0 ,170},
                    {  0 ,  0 ,212},
                    {  0 ,  0 ,255},
                    {  0 ,  0 ,  0},
                    { 36 , 36 , 36},
                    { 72 , 72 , 72},
                    {109 ,109 ,109},
                    {145 ,145 ,145},
                    {182 ,182 ,182},
                    {218 ,218 ,218},
                    {  0 ,113 ,188},
                    { 80 ,182 ,188},
                    {127 ,127 ,  0},
            };

    void common::draw_coco_bboxes(const cv::Mat& bgr, const std::vector<common::YoloBoxInfo>& bboxes, common::object_rect effect_roi)
    {
        static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
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

        cv::Mat image = bgr;
        int src_w = image.cols;
        int src_h = image.rows;
        int dst_w = effect_roi.width;
        int dst_h = effect_roi.height;
        float width_ratio = (float)src_w / (float)dst_w;
        float height_ratio = (float)src_h / (float)dst_h;


        for (size_t i = 0; i < bboxes.size(); i++)
        {
            const common::YoloBoxInfo& bbox = bboxes[i];
            cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                          coco_color_list[bbox.label][1],
                                          coco_color_list[bbox.label][2]);
            //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
            //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

            cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                          cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int x = (bbox.x1 - effect_roi.x) * width_ratio;
            int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          color, -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }

        cv::imshow("image", image);
    }

}