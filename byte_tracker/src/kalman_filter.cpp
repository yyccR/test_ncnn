#include "kalman_filter.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace byte_tracker {

namespace {

// 矩阵乘法: C = A * B
// A: rA x cA
// B: cA x cB
// C: rA x cB
void matrix_multiply(std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& B, int rA, int cA, int cB) {
    C.assign(rA * cB, 0.0f); // 初始化结果矩阵为0
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            for (int k = 0; k < cA; ++k) {
                C[i * cB + j] += A[i * cA + k] * B[k * cB + j]; // 累加乘积
            }
        }
    }
}

// 矩阵加法: C = A + B
void matrix_add(std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& B, int rows, int cols) {
    C.assign(rows * cols, 0.0f); // 初始化结果矩阵为0
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] + B[i]; // 元素相加
    }
}

// 矩阵减法: C = A - B
void matrix_subtract(std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& B, int rows, int cols) {
    C.assign(rows * cols, 0.0f); // 初始化结果矩阵为0
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] - B[i]; // 元素相减
    }
}

// 矩阵转置: T = A^T
void matrix_transpose(std::vector<float>& T, const std::vector<float>& A, int rows, int cols) {
    T.assign(rows * cols, 0.0f); // 初始化结果矩阵为0
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T[j * rows + i] = A[i * cols + j]; // 行列互换
        }
    }
}

// 4x4矩阵求逆，使用伴随矩阵法
// 来源: http://rodolphe-vaillant.fr/entry/7/c-code-for-4x4-matrix-inversion
bool invert_4x4(const std::vector<float>& m, std::vector<float>& invOut) {
    invOut.assign(16, 0.0f); // 初始化结果矩阵为0
    std::vector<float> inv(16);
    float det;

    // 计算代数余子式
    inv[0] = m[5]  * m[10] * m[15] - m[5]  * m[11] * m[14] - m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] + m[13] * m[6]  * m[11] - m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + m[4]  * m[11] * m[14] + m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - m[12] * m[6]  * m[11] + m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9]  * m[15] - m[4]  * m[11] * m[13] - m[8]  * m[5]  * m[15] + 
             m[8]  * m[7]  * m[13] + m[12] * m[5]  * m[11] - m[12] * m[7]  * m[9];

    inv[12] = -m[4]  * m[9]  * m[14] + m[4]  * m[10] * m[13] + m[8]  * m[5]  * m[14] - 
               m[8]  * m[6]  * m[13] - m[12] * m[5]  * m[10] + m[12] * m[6]  * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + m[1]  * m[11] * m[14] + m[9]  * m[2]  * m[15] - 
              m[9]  * m[3]  * m[14] - m[13] * m[2]  * m[11] + m[13] * m[3]  * m[10];

    inv[5] = m[0]  * m[10] * m[15] - m[0]  * m[11] * m[14] - m[8]  * m[2]  * m[15] + 
             m[8]  * m[3]  * m[14] + m[12] * m[2]  * m[11] - m[12] * m[3]  * m[10];

    inv[9] = -m[0]  * m[9]  * m[15] + m[0]  * m[11] * m[13] + m[8]  * m[1]  * m[15] - 
              m[8]  * m[3]  * m[13] - m[12] * m[1]  * m[11] + m[12] * m[3]  * m[9];

    inv[13] = m[0]  * m[9]  * m[14] - m[0]  * m[10] * m[13] - m[8]  * m[1]  * m[14] + 
              m[8]  * m[2]  * m[13] + m[12] * m[1]  * m[10] - m[12] * m[2]  * m[9];

    inv[2] = m[1]  * m[6]  * m[15] - m[1]  * m[7]  * m[14] - m[5]  * m[2]  * m[15] + 
             m[5]  * m[3]  * m[14] + m[13] * m[2]  * m[7] - m[13] * m[3]  * m[6];

    inv[6] = -m[0]  * m[6]  * m[15] + m[0]  * m[7]  * m[14] + m[4]  * m[2]  * m[15] - 
              m[4]  * m[3]  * m[14] - m[12] * m[2]  * m[7] + m[12] * m[3]  * m[6];

    inv[10] = m[0]  * m[5]  * m[15] - m[0]  * m[7]  * m[13] - m[4]  * m[1]  * m[15] + 
              m[4]  * m[3]  * m[13] + m[12] * m[1]  * m[7] - m[12] * m[3]  * m[5];

    inv[14] = -m[0]  * m[5]  * m[14] + m[0]  * m[6]  * m[13] + m[4]  * m[1]  * m[14] - 
               m[4]  * m[2]  * m[13] - m[12] * m[1]  * m[6] + m[12] * m[2]  * m[5];

    inv[3] = -m[1] * m[6] * m[11] +  m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - m[9] * m[2] * m[7] +  m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false; // 行列式为0，不可逆

    det = 1.0f / det;

    for (int i = 0; i < 16; i++)
        invOut[i] = inv[i] * det; // 归一化

    return true;
}

} // anonymous namespace

// KalmanFilterXYAH：卡尔曼滤波器实现，跟踪目标的中心、宽高和速度
KalmanFilterXYAH::KalmanFilterXYAH() : std_weight_position_(1.f / 20.f), std_weight_velocity_(1.f / 40.f) {
    // 初始化状态转移矩阵和更新矩阵
    motion_mat_ = std::vector<float>(64, 0.0f);
    update_mat_ = std::vector<float>(32, 0.0f);
    for (int i = 0; i < 8; i++) {
        motion_mat_[i * 8 + i] = 1.0f;
        if (i < 4) {
            motion_mat_[i * 8 + i + 4] = 1.0f;
        }
    }
    for (int i = 0; i < 4; i++) {
        update_mat_[i * 8 + i] = 1.0f;
    }
}

// predict：卡尔曼预测步骤，预测目标新状态
std::pair<std::vector<float>, std::vector<float>> KalmanFilterXYAH::predict(const std::vector<float>& mean, const std::vector<float>& covariance) {
    std::vector<float> new_mean(8, 0.0f);
    std::vector<float> new_covariance(64, 0.0f);

    // 状态转移
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            new_mean[i] += motion_mat_[i * 8 + j] * mean[j]; // 线性预测
        }
    }

    // 协方差更新
    std::vector<float> p_temp(64, 0.0f);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                p_temp[i * 8 + j] += motion_mat_[i * 8 + k] * covariance[k * 8 + j];
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                new_covariance[i * 8 + j] += p_temp[i * 8 + k] * motion_mat_[j * 8 + k];
            }
        }
    }

    // 添加过程噪声
    float w = mean[2];
    float h = mean[3];
    new_covariance[0 * 8 + 0] += std::pow(std_weight_position_ * w, 2);
    new_covariance[1 * 8 + 1] += std::pow(std_weight_position_ * h, 2);
    new_covariance[2 * 8 + 2] += std::pow(std_weight_position_ * w, 2);
    new_covariance[3 * 8 + 3] += std::pow(std_weight_position_ * h, 2);
    new_covariance[4 * 8 + 4] += std::pow(std_weight_velocity_ * w, 2);
    new_covariance[5 * 8 + 5] += std::pow(std_weight_velocity_ * h, 2);
    new_covariance[6 * 8 + 6] += std::pow(std_weight_velocity_ * w, 2);
    new_covariance[7 * 8 + 7] += std::pow(std_weight_velocity_ * h, 2);

    return {new_mean, new_covariance};
}

// update：卡尔曼更新步骤，用观测修正预测
std::pair<std::vector<float>, std::vector<float>> KalmanFilterXYAH::update(const std::vector<float>& mean, const std::vector<float>& covariance, const std::vector<float>& measurement) {
    // Kalman滤波更新步骤
    // H: 观测矩阵 (4x8)
    // R: 观测噪声协方差 (4x4)
    // P: 误差协方差 (8x8)
    // K: 卡尔曼增益 (8x4)
    // S: 创新协方差 (4x4)
    
    // 创新协方差: S = H * P * H^T + R
    std::vector<float> H_T;
    matrix_transpose(H_T, update_mat_, 4, 8); // H_T is 8x4

    std::vector<float> P_HT;
    matrix_multiply(P_HT, covariance, H_T, 8, 8, 4); // P*H^T is 8x4

    std::vector<float> H_P_HT;
    matrix_multiply(H_P_HT, update_mat_, P_HT, 4, 8, 4); // H*P*H^T is 4x4

    std::vector<float> R(16, 0.0f);
    float w = measurement[2];
    float h = measurement[3];
    R[0] = std::pow(std_weight_position_ * w, 2);
    R[5] = std::pow(std_weight_position_ * h, 2);
    R[10] = std::pow(std_weight_position_ * w, 2);
    R[15] = std::pow(std_weight_position_ * h, 2);

    std::vector<float> S;
    matrix_add(S, H_P_HT, R, 4, 4); // S = H*P*H^T + R

    // 创新协方差的逆: S^-1
    std::vector<float> S_inv(16);
    if (!invert_4x4(S, S_inv)) {
        // 如果矩阵不可逆，直接返回原状态
        return {mean, covariance};
    }

    // 卡尔曼增益: K = P * H^T * S^-1
    std::vector<float> K;
    matrix_multiply(K, P_HT, S_inv, 8, 4, 4);

    // 更新状态估计: x' = x + K * (z - H*x)
    std::vector<float> H_x;
    matrix_multiply(H_x, update_mat_, mean, 4, 8, 1);
    
    std::vector<float> converted_measurement = {
        measurement[0] + measurement[2] / 2.0f, // cx
        measurement[1] + measurement[3] / 2.0f, // cy
        measurement[2],
        measurement[3]
    };
    
    std::vector<float> innovation;
    matrix_subtract(innovation, converted_measurement, H_x, 4, 1); // 创新量

    std::vector<float> K_innovation;
    matrix_multiply(K_innovation, K, innovation, 8, 4, 1);

    std::vector<float> new_mean;
    matrix_add(new_mean, mean, K_innovation, 8, 1); // 新状态
    
    // 更新误差协方差: P' = (I - K * H) * P
    std::vector<float> K_H;
    matrix_multiply(K_H, K, update_mat_, 8, 4, 8);

    std::vector<float> I(64, 0.0f);
    for(int i=0; i<8; ++i) I[i*8+i] = 1.0f; // 单位阵

    std::vector<float> I_KH;
    matrix_subtract(I_KH, I, K_H, 8, 8);

    std::vector<float> new_covariance;
    matrix_multiply(new_covariance, I_KH, covariance, 8, 8, 8);

    return {new_mean, new_covariance};
}

// initiate：用观测初始化卡尔曼滤波状态
std::pair<std::vector<float>, std::vector<float>> KalmanFilterXYAH::initiate(const std::vector<float>& measurement) {
    std::vector<float> mean(8, 0.0f);
    std::vector<float> covariance(64, 0.0f);

    // 初始化状态: measurement is (tlx, tly, w, h)
    mean[0] = measurement[0] + measurement[2] / 2.0f; // cx
    mean[1] = measurement[1] + measurement[3] / 2.0f; // cy
    mean[2] = measurement[2]; // w
    mean[3] = measurement[3]; // h
    for (int i = 4; i < 8; i++) {
        mean[i] = 0.0f; // 速度初始化为0
    }

    // 初始化协方差
    float w = measurement[2];
    float h = measurement[3];
    for (int i = 0; i < 4; i++) {
        covariance[i * 8 + i] = std::pow(std_weight_position_ * (i % 2 == 0 ? w : h), 2); // 位置相关
    }
    for (int i = 4; i < 8; i++) {
        covariance[i * 8 + i] = std::pow(std_weight_velocity_ * (i % 2 == 0 ? w : h), 2); // 速度相关
    }

    return {mean, covariance};
}

} // namespace byte_tracker 