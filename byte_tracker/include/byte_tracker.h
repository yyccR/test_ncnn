#pragma once

#include <vector>
#include <memory>
#include "kalman_filter.h"
#include "matching.h"
#include <deque>

namespace byte_tracker {

enum class TrackState {
    New,
    Tracked,
    Lost,
    Removed
};

class STrack {
public:
    STrack(const std::vector<float>& xywh, float score, int cls, size_t trajectory_max_length = 60);
    void predict();
    void update(const STrack& new_track, int frame_id);
    void activate(const KalmanFilterXYAH& kalman_filter, int frame_id);
    void re_activate(const STrack& new_track, int frame_id, bool new_id = false);
    void mark_lost();
    void mark_removed();

    // Getters
    std::vector<float> get_tlwh() const;
    std::vector<float> get_xyxy() const;
    std::vector<float> get_xywh() const;
    std::vector<float> get_result(int img_w, int img_h) const;
    int get_track_id() const { return track_id_; }
    float get_score() const { return score_; }
    int get_cls() const { return cls_; }
    TrackState get_state() const { return state_; }
    bool is_activated() const { return is_activated_; }
    const std::deque<std::vector<float>>& get_trajectory() const { return trajectory_; }

    // Static methods
    static int next_id() {
        static int _id = 0;
        return _id++;
    }

private:
    std::vector<float> tlwh_;  // [x, y, w, h]
    float score_;
    int cls_;
    int track_id_;
    int frame_id_;
    int start_frame_;
    TrackState state_;
    std::shared_ptr<KalmanFilterXYAH> kalman_filter_;
    std::vector<float> mean_;
    std::vector<float> covariance_;
    bool is_activated_;
    std::deque<std::vector<float>> trajectory_;
    size_t trajectory_max_length_;
};

class BYTETracker {
public:
    BYTETracker(int track_buffer = 30, float track_high_thresh = 0.6f, float track_low_thresh = 0.1f, float match_thresh = 0.8f, size_t trajectory_length = 20);
    std::vector<std::vector<float>> update(const std::vector<std::vector<float>>& detections, const std::vector<float>& scores, const std::vector<int>& classes, int img_w, int img_h);
    const std::vector<std::shared_ptr<STrack>>& get_tracked_stracks() const { return tracked_stracks_; }

private:
    std::vector<std::shared_ptr<STrack>> tracked_stracks_;
    std::vector<std::shared_ptr<STrack>> lost_stracks_;
    std::vector<std::shared_ptr<STrack>> removed_stracks_;
    int frame_id_;
    int track_buffer_;
    float track_high_thresh_;
    float track_low_thresh_;
    float match_thresh_;
    std::shared_ptr<KalmanFilterXYAH> kalman_filter_;
    size_t trajectory_length_;

    std::vector<std::shared_ptr<STrack>> init_track(const std::vector<std::vector<float>>& detections, const std::vector<float>& scores, const std::vector<int>& classes);
    std::vector<std::vector<float>> get_dists(const std::vector<std::shared_ptr<STrack>>& tracks, const std::vector<std::shared_ptr<STrack>>& detections);
    void multi_predict(std::vector<std::shared_ptr<STrack>>& tracks);
    std::vector<std::shared_ptr<STrack>> joint_stracks(const std::vector<std::shared_ptr<STrack>>& tlista, const std::vector<std::shared_ptr<STrack>>& tlistb);
    std::vector<std::shared_ptr<STrack>> sub_stracks(const std::vector<std::shared_ptr<STrack>>& tlista, const std::vector<std::shared_ptr<STrack>>& tlistb);
    std::pair<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>>& stracksa, const std::vector<std::shared_ptr<STrack>>& stracksb);
};

} // namespace byte_tracker 