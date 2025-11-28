#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/opencv.hpp>

/**
 * @brief 将 Cellpose 的掩膜 (Masks) 转换为彩色分割图像。
 *
 * @param masks 输入掩膜 (Cellpose 结果), 期望类型为 CV_32SC1 或 CV_16UC1。
 * 像素值为 0 为背景，>0 为实例ID。
 * @return cv::Mat 彩色分割图像，类型为 CV_8UC3。每个实例ID被赋予一个独特的随机颜色。
 */
cv::Mat masks_to_color_image(const cv::Mat& masks);


/**
 * @brief 将彩色分割图叠加到原始图像上，用于最终显示。
 *
 * @param original_img 原始彩色图像 (CV_8UC3)。
 * @param color_mask 彩色分割图像 (CV_8UC3)。
 * @param alpha 叠加的透明度，范围 [0.0, 1.0]。
 * @return cv::Mat 叠加后的图像。
 */
cv::Mat overlay_masks_on_image(const cv::Mat& original_img, const cv::Mat& color_mask, double alpha = 0.5);

/**
 * @brief 在原始图像上绘制 Cellpose 掩膜的轮廓。
 *
 * @param original_img 原始彩色图像 (CV_8UC3)。
 * @param masks Cellpose 结果掩膜 (CV_32SC1 或 CV_16UC1)。
 * @param color 轮廓的颜色，例如 cv::Scalar(0, 255, 0) 为绿色。
 * @param thickness 轮廓线的粗细。
 * @return cv::Mat 带有轮廓线的图像。
 */
cv::Mat draw_contours_on_image(const cv::Mat& original_img, const cv::Mat& masks,
    cv::Scalar color = cv::Scalar(0, 255, 0), int thickness = 2);
#endif // VISUALIZE_H

