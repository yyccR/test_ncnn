#include <iostream>
#include "byte_tracker.h"
#include <algorithm>
#include <numeric>

namespace byte_tracker {

// STrack类：表示单个跟踪目标，包含卡尔曼滤波状态、ID、类别等
STrack::STrack(const std::vector<float>& xywh, float score, int cls, size_t trajectory_max_length)
    : score_(score), cls_(cls), track_id_(-1), frame_id_(0), start_frame_(0), state_(TrackState::New), is_activated_(false), trajectory_max_length_(trajectory_max_length) {
    // xywh为目标的中心点坐标和宽高
    tlwh_ = {xywh[0], xywh[1], xywh[2], xywh[3]};
    mean_ = std::vector<float>(8, 0.0f); // 卡尔曼滤波均值
    covariance_ = std::vector<float>(64, 0.0f); // 卡尔曼滤波协方差
    kalman_filter_ = nullptr;
    // 初始化轨迹点
    std::vector<float> center = {xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2};
    trajectory_.push_back(center);
}

// STrack::predict：利用卡尔曼滤波预测目标新状态
void STrack::predict() {
    if (!kalman_filter_) return; // 如果没有卡尔曼滤波器则直接返回
    if (state_ != TrackState::Tracked) {
        mean_[7] = 0; // 非跟踪状态时速度设为0
    }
    auto [new_mean, new_covariance] = kalman_filter_->predict(mean_, covariance_); // 预测新状态
    mean_ = new_mean;
    covariance_ = new_covariance;
}

// STrack::update：用新观测更新卡尔曼滤波状态
void STrack::update(const STrack& new_track, int frame_id) {
    frame_id_ = frame_id; // 更新帧号
    if (!kalman_filter_) return;
    auto [new_mean, new_covariance] = kalman_filter_->update(mean_, covariance_, new_track.get_tlwh()); // 更新卡尔曼滤波状态
    mean_ = new_mean;
    covariance_ = new_covariance;
    state_ = TrackState::Tracked; // 状态设为跟踪
    is_activated_ = true; // 激活
    score_ = new_track.get_score(); // 更新分数
    cls_ = new_track.get_cls(); // 更新类别
    // 记录轨迹点
    auto xywh = get_tlwh();
    std::vector<float> center = {xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2};
    trajectory_.push_back(center);
    if (trajectory_.size() > trajectory_max_length_) trajectory_.pop_front(); // 超出长度则移除最早点
}

// STrack::activate：激活新轨迹，分配ID并初始化卡尔曼滤波
void STrack::activate(const KalmanFilterXYAH& kalman_filter, int frame_id) {
    kalman_filter_ = std::make_shared<KalmanFilterXYAH>(kalman_filter); // 复制卡尔曼滤波器
    track_id_ = next_id(); // 分配全局唯一ID
    auto [mean, covariance] = kalman_filter_->initiate(get_tlwh()); // 初始化卡尔曼滤波状态
    mean_ = mean;
    covariance_ = covariance;
    state_ = TrackState::Tracked; // 状态设为跟踪
    is_activated_ = true; // 激活
    frame_id_ = frame_id; // 记录帧号
    start_frame_ = frame_id; // 记录起始帧
    // 记录轨迹点
    auto xywh = get_tlwh();
    std::vector<float> center = {xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2};
    trajectory_.push_back(center);
    if (trajectory_.size() > trajectory_max_length_) trajectory_.pop_front();
    // 打印 mean_、covariance_ 的内容，确认其有效性
    std::cout << "STrack::activate: mean_ = [";
    for (size_t i = 0; i < mean_.size(); i++) {
        std::cout << mean_[i] << (i < mean_.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "STrack::activate: covariance_ = [";
    for (size_t i = 0; i < covariance_.size(); i++) {
        std::cout << covariance_[i] << (i < covariance_.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

// STrack::re_activate：重新激活轨迹（如目标重新出现）
void STrack::re_activate(const STrack& new_track, int frame_id, bool new_id) {
    if (!kalman_filter_) return;
    auto [mean, covariance] = kalman_filter_->update(mean_, covariance_, new_track.get_tlwh()); // 更新卡尔曼滤波状态
    mean_ = mean;
    covariance_ = covariance;
    state_ = TrackState::Tracked; // 状态设为跟踪
    is_activated_ = true; // 激活
    frame_id_ = frame_id; // 更新帧号
    if (new_id) {
        track_id_ = next_id(); // 需要新ID时分配
    }
    score_ = new_track.get_score(); // 更新分数
    cls_ = new_track.get_cls(); // 更新类别
    // 记录轨迹点
    auto xywh = get_tlwh();
    std::vector<float> center = {xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2};
    trajectory_.push_back(center);
    if (trajectory_.size() > trajectory_max_length_) trajectory_.pop_front();
}

// STrack::mark_lost：标记轨迹为丢失
void STrack::mark_lost() {
    state_ = TrackState::Lost;
}

// STrack::mark_removed：标记轨迹为移除
void STrack::mark_removed() {
    state_ = TrackState::Removed;
}

// STrack::get_tlwh：获取当前目标的左上角坐标和宽高
std::vector<float> STrack::get_tlwh() const {
    if (mean_.size() != 8 || mean_ == std::vector<float>(8, 0.0f)) {
        return tlwh_; // 如果未初始化卡尔曼滤波，直接返回原始框
    }
    std::vector<float> ret(4);
    ret[0] = mean_[0] - mean_[2] / 2; // x1 = cx - w/2
    ret[1] = mean_[1] - mean_[3] / 2; // y1 = cy - h/2
    ret[2] = mean_[2]; // w
    ret[3] = mean_[3]; // h
    // 打印 mean_ 的内容，确认其有效性
    std::cout << "STrack::get_tlwh: mean_ = [";
    for (size_t i = 0; i < mean_.size(); i++) {
        std::cout << mean_[i] << (i < mean_.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    return ret;
}

// STrack::get_xyxy：返回左上右下坐标
std::vector<float> STrack::get_xyxy() const {
    std::vector<float> ret = get_tlwh();
    ret[2] += ret[0]; // x2 = x1 + w
    ret[3] += ret[1]; // y2 = y1 + h
    return ret;
}

// STrack::get_xywh：返回中心点坐标和宽高
std::vector<float> STrack::get_xywh() const {
    std::vector<float> ret = tlwh_;
    ret[0] += ret[2] / 2; // cx = x1 + w/2
    ret[1] += ret[3] / 2; // cy = y1 + h/2
    return ret;
}

// STrack::get_result：返回最终输出格式[x1, y1, x2, y2, id, score, cls]
std::vector<float> STrack::get_result(int img_w, int img_h) const {
    std::vector<float> ret = get_xyxy();
    ret[0] = std::max(0.f, std::min(ret[0], float(img_w - 1))); // 限制在图像范围内
    ret[1] = std::max(0.f, std::min(ret[1], float(img_h - 1)));
    ret[2] = std::max(0.f, std::min(ret[2], float(img_w - 1)));
    ret[3] = std::max(0.f, std::min(ret[3], float(img_h - 1)));
    ret.push_back(static_cast<float>(track_id_)); // 添加ID
    ret.push_back(score_); // 添加分数
    ret.push_back(static_cast<float>(cls_)); // 添加类别
    return ret;
}

// BYTETracker类：多目标跟踪主流程
BYTETracker::BYTETracker(int track_buffer, float track_high_thresh, float track_low_thresh, float match_thresh, size_t trajectory_length)
    : frame_id_(0), track_buffer_(track_buffer), track_high_thresh_(track_high_thresh), track_low_thresh_(track_low_thresh), match_thresh_(match_thresh), trajectory_length_(trajectory_length) {
    kalman_filter_ = std::make_shared<KalmanFilterXYAH>(); // 创建卡尔曼滤波器
}

// BYTETracker::update：主跟踪流程，输入为检测框、分数、类别，输出为跟踪结果
std::vector<std::vector<float>> BYTETracker::update(const std::vector<std::vector<float>>& detections, const std::vector<float>& scores, const std::vector<int>& classes, int img_w, int img_h) {
    frame_id_++; // 帧号加一
    std::vector<std::shared_ptr<STrack>> activated_stracks; // 本帧激活的轨迹
    std::vector<std::shared_ptr<STrack>> refind_stracks; // 本帧重新找到的轨迹
    std::vector<std::shared_ptr<STrack>> lost_stracks; // 本帧丢失的轨迹
    std::vector<std::shared_ptr<STrack>> removed_stracks; // 本帧移除的轨迹

    // 1. 高分低分两阶段关联
    std::vector<int> remain_inds;
    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] >= track_high_thresh_) {
            remain_inds.push_back(i); // 只保留高分检测
        }
    }

    std::vector<std::vector<float>> dets;
    std::vector<float> scores_keep;
    std::vector<int> cls_keep;
    for (int i : remain_inds) {
        dets.push_back(detections[i]);
        scores_keep.push_back(scores[i]);
        cls_keep.push_back(classes[i]);
    }

    // 初始化检测目标为STrack对象
    std::vector<std::shared_ptr<STrack>> detections_strack = init_track(dets, scores_keep, cls_keep);
    std::vector<std::shared_ptr<STrack>> unconfirmed;
    std::vector<std::shared_ptr<STrack>> tracked_stracks;
    for (auto& track : tracked_stracks_) {
        if (!track->is_activated()) {
            unconfirmed.push_back(track); // 未激活的轨迹
        } else {
            tracked_stracks.push_back(track); // 已激活的轨迹
        }
    }

    std::cout << "init_track done, detections_strack.size(): " << detections_strack.size() << std::endl;
    std::cout << "tracked_stracks_ before split: " << tracked_stracks_.size() << std::endl;
    std::cout << "tracked_stracks after split: " << tracked_stracks.size() << std::endl;
    std::cout << "unconfirmed after split: " << unconfirmed.size() << std::endl;
    // 2. 卡尔曼预测，对已激活轨迹做状态预测
    for (auto& track : tracked_stracks) {
        if (track->is_activated())
            track->predict(); // 预测新状态
    }
    std::cout << "after predict" << std::endl;
    // 3. 匈牙利算法关联，计算距离矩阵并分配检测与轨迹
    std::cout << "tracked_stracks.size(): " << tracked_stracks.size() << std::endl;
    std::cout << "detections_strack.size(): " << detections_strack.size() << std::endl;
    std::vector<std::vector<float>> dists = get_dists(tracked_stracks, detections_strack); // 计算距离矩阵
    std::cout << "after get_dists" << std::endl;
    std::vector<std::pair<int, int>> matches = linear_assignment(dists, match_thresh_); // 匈牙利算法匹配
    std::cout << "after linear_assignment" << std::endl;

    std::cout << "matches.size(): " << matches.size() << std::endl;
    for (size_t i = 0; i < matches.size(); ++i) {
        std::cout << "match[" << i << "]: (" << matches[i].first << ", " << matches[i].second << ")" << std::endl;
    }
    std::cout << "before match update loop" << std::endl;
    std::vector<int> u_track, u_detection;
    for (size_t i = 0; i < tracked_stracks.size(); i++) {
        u_track.push_back(i); // 所有轨迹索引
    }
    for (size_t i = 0; i < detections_strack.size(); i++) {
        u_detection.push_back(i); // 所有检测索引
    }
    for (const auto& match : matches) {
        u_track.erase(std::remove(u_track.begin(), u_track.end(), match.first), u_track.end()); // 移除已匹配轨迹
        u_detection.erase(std::remove(u_detection.begin(), u_detection.end(), match.second), u_detection.end()); // 移除已匹配检测
        tracked_stracks[match.first]->update(*detections_strack[match.second], frame_id_); // 用检测更新轨迹
        activated_stracks.push_back(tracked_stracks[match.first]); // 加入激活列表
    }
    std::cout << "after match update loop" << std::endl;

    // 4. 轨迹管理，未匹配的轨迹标记为丢失，未匹配的检测尝试激活新轨迹
    for (int i : u_track) {
        tracked_stracks[i]->mark_lost(); // 标记为丢失
        lost_stracks.push_back(tracked_stracks[i]);
    }

    for (int i : u_detection) {
        if (detections_strack[i]->get_score() >= track_high_thresh_) {
            detections_strack[i]->activate(*kalman_filter_, frame_id_); // 激活新轨迹
            activated_stracks.push_back(detections_strack[i]);
        }
    }

    // 5. 更新状态，保存当前帧的激活、丢失、移除轨迹
    tracked_stracks_ = activated_stracks;
    lost_stracks_ = lost_stracks;
    removed_stracks_ = removed_stracks;

    // 6. 输出结果，返回所有激活轨迹的[x1, y1, x2, y2, id, score, cls]
    std::vector<std::vector<float>> results;
    for (const auto& track : tracked_stracks_) {
        if (track->is_activated()) {
            results.push_back(track->get_result(img_w, img_h));
        }
    }
    std::cout << "tracked_stracks_.size(): " << tracked_stracks_.size() << std::endl;
    for (size_t i = 0; i < tracked_stracks_.size(); ++i) {
        std::cout << "track[" << i << "] is_activated: " << (tracked_stracks_[i] ? tracked_stracks_[i]->is_activated() : -1) << std::endl;
    }
    return results;
}

// BYTETracker::init_track：将检测结果初始化为STrack对象
std::vector<std::shared_ptr<STrack>> BYTETracker::init_track(const std::vector<std::vector<float>>& detections, const std::vector<float>& scores, const std::vector<int>& classes) {
    std::vector<std::shared_ptr<STrack>> tracks;
    for (size_t i = 0; i < detections.size(); i++) {
        auto track = std::make_shared<STrack>(detections[i], scores[i], classes[i], trajectory_length_); // 创建STrack对象
        track->activate(*kalman_filter_, frame_id_); // 激活
        tracks.push_back(track);
    }
    return tracks;
}

// BYTETracker::get_dists：计算轨迹与检测之间的距离（IOU距离）
std::vector<std::vector<float>> BYTETracker::get_dists(const std::vector<std::shared_ptr<STrack>>& tracks, const std::vector<std::shared_ptr<STrack>>& detections) {
    std::vector<STrack> tracks_raw, detections_raw;
    for (const auto& t : tracks) tracks_raw.push_back(*t); // 拷贝轨迹
    for (const auto& d : detections) detections_raw.push_back(*d); // 拷贝检测
    std::cout << "get_dists: tracks_raw.size()=" << tracks_raw.size() << ", detections_raw.size()=" << detections_raw.size() << std::endl;
    return iou_distance(tracks_raw, detections_raw); // 计算IOU距离
}

// BYTETracker::multi_predict：批量预测所有轨迹
void BYTETracker::multi_predict(std::vector<std::shared_ptr<STrack>>& tracks) {
    for (auto& track : tracks) {
        if (track->is_activated())
            track->predict(); // 预测
    }
}

// BYTETracker::joint_stracks：合并两个轨迹列表，去重
std::vector<std::shared_ptr<STrack>> BYTETracker::joint_stracks(const std::vector<std::shared_ptr<STrack>>& tlista, const std::vector<std::shared_ptr<STrack>>& tlistb) {
    std::vector<std::shared_ptr<STrack>> res = tlista;
    for (const auto& t : tlistb) {
        if (std::find_if(res.begin(), res.end(), [&t](const std::shared_ptr<STrack>& track) { return track->get_track_id() == t->get_track_id(); }) == res.end()) {
            res.push_back(t); // 只添加未出现过的轨迹
        }
    }
    return res;
}

// BYTETracker::sub_stracks：从tlista中去除tlistb中存在的轨迹
std::vector<std::shared_ptr<STrack>> BYTETracker::sub_stracks(const std::vector<std::shared_ptr<STrack>>& tlista, const std::vector<std::shared_ptr<STrack>>& tlistb) {
    std::vector<std::shared_ptr<STrack>> res;
    for (const auto& t : tlista) {
        if (std::find_if(tlistb.begin(), tlistb.end(), [&t](const std::shared_ptr<STrack>& track) { return track->get_track_id() == t->get_track_id(); }) == tlistb.end()) {
            res.push_back(t); // 只保留tlistb中没有的轨迹
        }
    }
    return res;
}

// BYTETracker::remove_duplicate_stracks：去除重复轨迹
std::pair<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> BYTETracker::remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>>& stracksa, const std::vector<std::shared_ptr<STrack>>& stracksb) {
    std::vector<STrack> stracksa_raw, stracksb_raw;
    for (const auto& t : stracksa) stracksa_raw.push_back(*t);
    for (const auto& t : stracksb) stracksb_raw.push_back(*t);
    std::vector<std::vector<float>> pdist = iou_distance(stracksa_raw, stracksb_raw); // 计算IOU距离
    std::vector<int> dupa, dupb;
    for (size_t i = 0; i < pdist.size(); i++) {
        for (size_t j = 0; j < pdist[i].size(); j++) {
            if (pdist[i][j] < 0.15) {
                dupa.push_back(i); // 记录重复索引
                dupb.push_back(j);
            }
        }
    }
    std::vector<std::shared_ptr<STrack>> resa, resb;
    for (size_t i = 0; i < stracksa.size(); i++) {
        if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) {
            resa.push_back(stracksa[i]); // 保留未重复的
        }
    }
    for (size_t i = 0; i < stracksb.size(); i++) {
        if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) {
            resb.push_back(stracksb[i]); // 保留未重复的
        }
    }
    return {resa, resb};
}

} // namespace byte_tracker 