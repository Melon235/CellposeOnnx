#ifndef CELLPOSE_CORE_HPP
#define CELLPOSE_CORE_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// -------------------------------------------------
// 模型输出定义 
// -------------------------------------------------

// run_net 函数的返回结果
struct RunNetResult {
    cv::Mat flows_and_prob; // 流场信息和细胞概率 [Ly x Lx x 3]
    cv::Mat style;          // 风格向量 style [1 x 256]
};

// -------------------------------------------------
// 函数声明
// -------------------------------------------------

/**
 * @brief 在图像上运行神经网络模型
 * @param onnx_model_path ONNX 模型的路径
 * @param imgi_hwc 输入图像，格式为 H*W*C ，数据类型为CV_32FC3
 * @param batch_size 批处理大小
 * @param augment 是否使用测试时数据增强
 * @param tile_overlap 分块重叠率
 * @param bsize 分块大小
 * @param rsz 图像缩放系数, rsz = 30 / user_diameter
 * @return RunNetResult
 */
RunNetResult run_net(const std::string& onnx_model_path, const cv::Mat& imgi_hwc, int batch_size = 8,
    bool augment = false, float tile_overlap = 0.1, int bsize = 224, float rsz = 1.0);


/**
 * @brief 通过迭代追踪像素点在流场中的运动轨迹
 * @param dP 流场，一个 2 通道的 CV_32F Mat [Ly x Lx]，通道顺序为 (Y-flow, X-flow)
 * @param inds 初始像素点的整数坐标 (x, y) 的向量
 * @param niter 迭代次数
 * @return 一个 (n_points, 2) 的 Mat，包含每个点最终的浮点坐标 (y, x)
 */
cv::Mat steps_interp(const cv::Mat& dP, const std::vector<cv::Point>& inds, int niter = 200, torch::Device device = torch::kCPU);


/**
 * @brief steps_interp 封装，执行完整的像素追踪
 */
cv::Mat follow_flows(const cv::Mat& dP, const std::vector<cv::Point>& inds, int niter = 200);

/**
 * @brief 最大池化
 */
cv::Mat max_pool_nd_opencv(const cv::Mat& input, int kernel_size);

/**
 * @brief 去除过大掩膜
 */
void remove_large_masks_and_renumber(cv::Mat& M0, double max_size_fraction = 0.4f);

/**
 * @brief 填充单个掩膜中的孔洞 (使用 OpenCV FloodFill 或形态学闭合)
 * @param mask_binary CV_8U, 单通道二值掩膜 (前景为 1 或 255)
 * @return CV_8U, 填充孔洞后的掩膜
 */
cv::Mat fill_holes_single_mask(const cv::Mat& mask_binary);

/**
 * @brief 填充空洞 去除小掩膜
 * @param masks 标签图 (CV_16U 或 CV_32S)
 * @param min_size 最小像素数
 * @return 经过填充和移除小掩膜的标签图 (CV_32S)
 */
cv::Mat fill_holes_and_remove_small_masks(const cv::Mat& masks_in, int min_size);

/**
 * @brief 根据像素收敛点生成最终的实例掩码
 * @param p_final `follow_flows` 的输出，像素的最终位置
 * @param inds 参与计算的原始像素索引
 * @param original_shape 原始图像的尺寸
 * @param max_size_fraction
 * @return 实例分割掩码图，CV_16U 类型，每个实例有唯一标签
 */
cv::Mat get_masks(const cv::Mat& p_final_input, const std::vector<cv::Point>& inds,
    cv::Size original_shape, double max_size_fraction = 0.4);


/**
 * @brief 掩码计算函数
 * @param dP run_net 输出的流场 [Ly x Lx x 2]
 * @param cellprob run_net 输出的细胞概率图 [Ly x Lx x 1]
 * @param cellprob_threshold 细胞概率阈值
 * @param niter 动力学迭代次数
 * @param max_size_fraction
 * @return 实例分割掩码图，CV_16U 类型
 */
cv::Mat compute_masks(const cv::Mat& dP, const cv::Mat& cellprob, float cellprob_threshold = 0.0, int niter = 200, float max_size_fraction = 0.4, int min_size = -1);


/**
 * @brief 掩码计算与后处理，移除小掩码（认为是噪声）
 * @param dP 流场
 * @param cellprob 细胞概率
 * @param min_size threshold 移除小于此尺寸的掩码
 * @return 最终的实例分割掩码图，CV_16U 类型
 */
cv::Mat resize_and_compute_masks(const cv::Mat& dP, const cv::Mat& cellprob, int min_size = 15);

/**
 * @brief 最高层封装pipeline img2mask
 * @param onnx_model_path 模型路径
 * @param img 分割图像
 * @param rsz 缩放系数
 * @return masks 分割掩码，CV_16U 类型
 */
cv::Mat Cellpose_img2mask(const std::string& onnx_model_path, const cv::Mat& img, float rsz = 1);

#endif //CELLPOSE_CORE_HPP