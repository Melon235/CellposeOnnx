#include "Visualize.h"
#include <map>
#include <random>

/**
 * @brief 内部函数：为给定的实例ID生成一个随机但区分度高的颜色。
 * * 使用HSV颜色空间生成颜色，保证较高的亮度和饱和度，背景颜色为黑色。
 * * @param label 实例ID (大于 0)。
 * @return cv::Scalar BGR 格式的颜色。
 */
static cv::Scalar get_random_color(int label) {
    if (label == 0) {
        return cv::Scalar(0, 0, 0); // 背景为黑色
    }

    // 使用实例ID作为随机种子，确保每次运行时颜色一致
    // 随机数生成器用于生成 H (色相)
    std::mt19937 gen(label);
    std::uniform_int_distribution<> distrib_h(0, 179); // OpenCV HUE 范围 0-179

    int h = distrib_h(gen);

    // 固定 Saturation (饱和度) 和 Value (亮度) 来确保颜色鲜艳
    int s = 200 + (label % 55); // S 范围 0-255
    int v = 200 + (label % 55); // V 范围 0-255

    // 创建 HSV 颜色
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));

    // 转换回 BGR 颜色空间
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    // 提取 BGR Scalar
    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(color[0], color[1], color[2]);
}

/**
 * @brief 将 Cellpose 的掩膜 (Masks) 转换为彩色分割图像。
 */
cv::Mat masks_to_color_image(const cv::Mat& masks) {
    if (masks.empty()) {
        return cv::Mat();
    }

    // 确定输入类型，我们期望是 32S 或 16U
    if (masks.type() != CV_32SC1 && masks.type() != CV_16UC1) {
        std::cerr << "Error: Masks input must be CV_32SC1 or CV_16UC1." << std::endl;
        return cv::Mat();
    }

    // 准备输出彩色图像 (CV_8UC3)
    cv::Mat color_image(masks.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // 默认背景黑色

    // 使用 map 存储已生成的颜色，避免重复计算
    std::map<int, cv::Scalar> color_map;

    // 迭代遍历所有像素
    for (int y = 0; y < masks.rows; ++y) {
        // 根据输入类型选择指针
        const int* mask_ptr_32 = nullptr;
        const unsigned short* mask_ptr_16 = nullptr;

        if (masks.type() == CV_32SC1) {
            mask_ptr_32 = masks.ptr<int>(y);
        }
        else {
            mask_ptr_16 = masks.ptr<unsigned short>(y);
        }

        cv::Vec3b* color_ptr = color_image.ptr<cv::Vec3b>(y);

        for (int x = 0; x < masks.cols; ++x) {
            int label = (masks.type() == CV_32SC1) ? mask_ptr_32[x] : mask_ptr_16[x];

            if (label > 0) { // 实例ID > 0
                // 查找或生成颜色
                if (color_map.find(label) == color_map.end()) {
                    color_map[label] = get_random_color(label);
                }

                // 设置像素颜色
                cv::Scalar color = color_map[label];
                color_ptr[x][0] = (uchar)color[0]; // B
                color_ptr[x][1] = (uchar)color[1]; // G
                color_ptr[x][2] = (uchar)color[2]; // R
            }
        }
    }

    return color_image;
}

/**
 * @brief 将彩色分割图叠加到原始图像上。
 */
cv::Mat overlay_masks_on_image(const cv::Mat& original_img, const cv::Mat& color_mask, double alpha) {
    if (original_img.empty() || color_mask.empty()) {
        return cv::Mat();
    }

    if (original_img.size() != color_mask.size() || original_img.type() != CV_8UC3 || color_mask.type() != CV_8UC3) {
        std::cerr << "Error: Images must be same size and CV_8UC3 for overlay." << std::endl;
        return original_img.clone();
    }

    // 确保 alpha 在 [0, 1] 范围内
    alpha = std::min(1.0, std::max(0.0, alpha));
    double beta = 1.0 - alpha; // beta 是原始图像的权重

    cv::Mat overlay_result;

    // 使用 cv::addWeighted 进行透明叠加：result = alpha * color_mask + beta * original_img
    cv::addWeighted(color_mask, alpha, original_img, beta, 0.0, overlay_result);

    // 对于背景为黑色的区域，我们不想叠加，只想显示原始图像。
    // 但是 cv::addWeighted 已经完成了全局叠加，如果想要更精细的控制，需要手动遍历。

    // **【优化】只在掩膜区域进行叠加（更符合 Cellpose 可视化习惯）**
    // 假设 color_mask 中所有黑色像素 (0, 0, 0) 都是背景（Label 0）。

    cv::Mat final_result = original_img.clone(); // 从原始图像开始

    for (int y = 0; y < final_result.rows; ++y) {
        const cv::Vec3b* color_mask_ptr = color_mask.ptr<cv::Vec3b>(y);
        cv::Vec3b* final_result_ptr = final_result.ptr<cv::Vec3b>(y);
        const cv::Vec3b* original_img_ptr = original_img.ptr<cv::Vec3b>(y);

        for (int x = 0; x < final_result.cols; ++x) {
            // 检查彩色掩膜的像素是否非黑色 (即它是一个实例，不是背景)
            if (color_mask_ptr[x][0] != 0 || color_mask_ptr[x][1] != 0 || color_mask_ptr[x][2] != 0) {
                // 执行加权平均 (叠加)
                for (int c = 0; c < 3; ++c) {
                    final_result_ptr[x][c] = (uchar)(alpha * color_mask_ptr[x][c] + beta * original_img_ptr[x][c]);
                }
            }
            // 否则，保持原始图像的像素值 (final_result_ptr[x] 已经是 original_img_ptr[x])
        }
    }

    return final_result;
}

cv::Mat draw_contours_on_image(const cv::Mat& original_img, const cv::Mat& masks,
    cv::Scalar color, int thickness) {
    if (original_img.empty() || masks.empty()) {
        return cv::Mat();
    }

    cv::Mat result = original_img.clone(); // 从原始图像开始

    // 1. 获取所有实例ID
    std::set<int> unique_labels;
    // 遍历掩膜，找出所有的非零实例ID
    cv::Mat flat_masks = masks.reshape(1, 1);
    for (int i = 0; i < flat_masks.cols; ++i) {
        int label = (masks.type() == CV_32SC1) ? flat_masks.at<int>(0, i) : flat_masks.at<unsigned short>(0, i);
        if (label > 0) {
            unique_labels.insert(label);
        }
    }

    // 2. 逐个实例提取轮廓并绘制
    for (int label : unique_labels) {
        // 创建一个只包含当前实例的二值掩膜
        cv::Mat current_mask = (masks == label);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(current_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 绘制轮廓
        cv::drawContours(result, contours, -1, color, thickness);
    }

    return result;
}