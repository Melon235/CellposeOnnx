#include "Cellpose_core.h"
#include <vector>
#include <torch/torch.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map>
#include <functional>
#include <limits>

// =================================================================================
// Internal Helper Functions
// =================================================================================

namespace {

    // 计算图像所需的填充量
    void get_pad_yx(int Ly, int Lx, int min_size_y, int min_size_x,
        int& ypad1, int& ypad2, int& xpad1, int& xpad2) {
        int ypad = std::max(0, min_size_y - Ly);
        int xpad = std::max(0, min_size_x - Lx);
        ypad1 = ypad / 2;
        ypad2 = ypad - ypad1;
        xpad1 = xpad / 2;
        xpad2 = xpad - xpad1;
    }

    // 生成边缘渐变的掩膜，用于平滑拼接
    cv::Mat _taper_mask(int ly = 224, int lx = 224, double sig = 7.5) {
        int bsize = std::max({ 224, ly, lx });
        cv::Mat xm = cv::Mat::zeros(1, bsize, CV_64F);

        for (int i = 0; i < bsize; ++i) {
            xm.at<double>(0, i) = static_cast<double>(i);
        }
        xm = cv::abs(xm - (bsize - 1) / 2.0);

        cv::Mat arg;
        cv::Mat offset_term = xm - (bsize / 2.0 - 20.0);
        cv::divide(offset_term, sig, arg);

        cv::Mat mask_line;
        cv::exp(arg, mask_line); // mask_line = exp(arg)
        mask_line = 1.0 / (1.0 + mask_line);

        cv::Mat mask = mask_line.t() * mask_line;

        cv::Rect crop_region(bsize / 2 - lx / 2, bsize / 2 - ly / 2, lx, ly);
        return mask(crop_region).clone();
    }

    // 将分块结果平滑地拼接成一张大图
    cv::Mat average_tiles(const std::vector<cv::Mat>& y, const std::vector<std::pair<int, int>>& ysub,
        const std::vector<std::pair<int, int>>& xsub, int Ly, int Lx) {
        if (y.empty()) return cv::Mat();

        int nchan = y[0].channels();
        cv::Mat Navg = cv::Mat::zeros(Ly, Lx, CV_64F);
        cv::Mat yf = cv::Mat::zeros(Ly, Lx, CV_64FC(nchan));

        cv::Mat taper = _taper_mask(y[0].rows, y[0].cols);
        cv::Mat taper_mask_multi_channel;
        std::vector<cv::Mat> taper_channels(nchan, taper);
        cv::merge(taper_channels, taper_mask_multi_channel);

        for (size_t j = 0; j < ysub.size(); ++j) {
            cv::Rect roi(xsub[j].first, ysub[j].first, xsub[j].second - xsub[j].first, ysub[j].second - ysub[j].first);
            cv::Mat y_tapered;
            cv::multiply(y[j], taper_mask_multi_channel, y_tapered);
            yf(roi) += y_tapered;
            Navg(roi) += taper;
        }

        cv::Mat Navg_inv;
        cv::Mat Navg_safe = Navg.clone();

        cv::max(Navg_safe, 1e-6, Navg_safe);
        cv::divide(1.0, Navg_safe, Navg_inv);

        std::vector<cv::Mat> yf_channels;
        cv::split(yf, yf_channels);
        for (int i = 0; i < nchan; ++i) {
            cv::multiply(yf_channels[i], Navg_inv, yf_channels[i]);
        }
        cv::merge(yf_channels, yf);

        return yf;
    }

    inline double get_value_safe(const cv::Mat& mat, int r, int c) {
        if (mat.type() == CV_64F) {
            return mat.at<double>(r, c);
        }
        else if (mat.type() == CV_32F) {
            // 从 float 读取并转换为 double
            return static_cast<double>(mat.at<float>(r, c));
        }
        // 默认或错误类型处理
        return 0.0;
    }
}

// =================================================================================
// Inference
// =================================================================================

RunNetResult run_net(const std::string& onnx_model_path, const cv::Mat& imgi_hwc, int batch_size,
    bool augment, float tile_overlap, int bsize, float rsz) {

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "run_net");
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ORT_SEQUENTIAL); // 强制顺序执行

    std::wstring widestr = std::wstring(onnx_model_path.begin(), onnx_model_path.end());
    const wchar_t* widecstr = widestr.c_str();
    Ort::Session session(env, widecstr, session_options);

    cv::Mat img_norm;
    if (imgi_hwc.type() != CV_32FC3 && imgi_hwc.type() != CV_32FC1) {
        imgi_hwc.convertTo(img_norm, CV_32F, 1.0 / 255.0);
    }
    else {
        img_norm = imgi_hwc.clone();
    }

    img_norm = img_norm - 0.5;

    cv::Mat img_resized;
    if (rsz != 1.0) cv::resize(img_norm, img_resized, cv::Size(), rsz, rsz, cv::INTER_LINEAR);
    else img_resized = img_norm;

    int ypad1, ypad2, xpad1, xpad2;
    get_pad_yx(img_resized.rows, img_resized.cols, bsize, bsize, ypad1, ypad2, xpad1, xpad2);

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, ypad1, ypad2, xpad1, xpad2, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // --- 分块逻辑 ---
    int Ly = img_padded.rows;
    int Lx = img_padded.cols;
    int ny = (Ly <= bsize) ? 1 : static_cast<int>(ceil((1.0 + 2.0 * tile_overlap) * Ly / bsize));
    int nx = (Lx <= bsize) ? 1 : static_cast<int>(ceil((1.0 + 2.0 * tile_overlap) * Lx / bsize));

    std::vector<cv::Rect> rois;
    std::vector<std::pair<int, int>> ysub, xsub;
    for (int j = 0; j < ny; ++j) {
        int yl = (ny > 1) ? std::min(Ly - bsize, static_cast<int>(round(j * (float)(Ly - bsize) / (ny - 1)))) : 0;
        for (int i = 0; i < nx; ++i) {
            int xl = (nx > 1) ? std::min(Lx - bsize, static_cast<int>(round(i * (float)(Lx - bsize) / (nx - 1)))) : 0;
            rois.emplace_back(xl, yl, bsize, bsize);
            ysub.push_back({ yl, yl + bsize });
            xsub.push_back({ xl, xl + bsize });
        }
    }

    // --- 推理 ---
    std::vector<float> zeros_channel(bsize * bsize, 0.0f);
    std::vector<cv::Mat> y_tiles, style_tiles;
    for (size_t j = 0; j < rois.size(); j += batch_size) {
        size_t current_batch_size = std::min((size_t)batch_size, rois.size() - j);

        std::vector<int64_t> input_shape = { (int64_t)current_batch_size, 2, (int64_t)bsize, (int64_t)bsize };
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(current_batch_size * 2 * bsize * bsize);

        for (size_t b = 0; b < current_batch_size; ++b) {
            cv::Mat tile = img_padded(rois[j + b]);
            cv::Mat channel0;
            if (tile.channels() == 1 && tile.type() == CV_32F) {
                channel0 = tile;
            }
            else if (tile.channels() >= 3 && tile.type() == CV_32FC3) {
                cv::cvtColor(tile, channel0, cv::COLOR_BGR2GRAY);
            }
            else {
                std::cerr << "RUN_NET ERROR: Tile has invalid channels or type (Expected CV_32F)." << std::endl;
                continue;
            }
            //  ONNX 输入通道 0 (图像数据)
            input_tensor_values.insert(input_tensor_values.end(),
                (float*)channel0.datastart, (float*)channel0.dataend);

            //  ONNX 输入通道 1 （置0）
            input_tensor_values.insert(input_tensor_values.end(), zeros_channel.begin(), zeros_channel.end());
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = { "input" };
        const char* output_names[] = { "output", "style" };
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        float* style_data = output_tensors[1].GetTensorMutableData<float>();

        for (size_t b = 0; b < current_batch_size; ++b) {
            std::vector<cv::Mat> out_channels;
            for (int c = 0; c < 3; ++c) {
                out_channels.push_back(cv::Mat(bsize, bsize, CV_32F, output_data + (b * 3 + c) * bsize * bsize));
            }
            cv::Mat y_tile_f32;
            cv::merge(out_channels, y_tile_f32);
            cv::Mat y_tile_f64;
            y_tile_f32.convertTo(y_tile_f64, CV_64FC3);
            y_tiles.push_back(y_tile_f64.clone());

            style_tiles.push_back(cv::Mat(1, 256, CV_32F, style_data + b * 256).clone());
        }
    }

    // --- 后处理 ---
    cv::Mat yfi_padded = average_tiles(y_tiles, ysub, xsub, Ly, Lx);
    cv::Rect crop_roi(xpad1, ypad1, img_resized.cols, img_resized.rows);
    cv::Mat yfi = yfi_padded(crop_roi);

    cv::Mat style_cat;
    if (!style_tiles.empty()) cv::vconcat(style_tiles, style_cat);
    cv::Mat style_final;
    if (!style_cat.empty()) cv::reduce(style_cat, style_final, 0, cv::REDUCE_AVG);

    std::cout << "yfi" << "rows: " << yfi.rows << ", cols: " << yfi.cols
        << ", channels: " << yfi.channels()
        << ", type: " << yfi.type() << std::endl;

    return { yfi, style_final };
}

// =================================================================================
// Dynamics Simulation
// =================================================================================


cv::Mat steps_interp(const cv::Mat& dP, const std::vector<cv::Point>& inds, int niter, torch::Device device) {
    if (inds.empty()) return cv::Mat();

    int Ly = dP.rows;
    int Lx = dP.cols;
    const float Ly_size = static_cast<float>(Ly);
    const float Lx_size = static_cast<float>(Lx);
    const int ndim = 2;
    const int N_seeds = inds.size();
    const float shape[2] = { Lx_size - 1.f, Ly_size - 1.f };

    // 1. 初始化 pt (种子点坐标) [1, 1, N, 2]
    torch::Tensor pt_tensor = torch::zeros({ 1, 1, N_seeds, ndim }, torch::kFloat32).to(device);
    // 使用 accessor 加速赋值
    auto pt_acc = pt_tensor.accessor<float, 4>();
    for (int i = 0; i < N_seeds; ++i) {
        float x = static_cast<float>(inds[i].x);
        float y = static_cast<float>(inds[i].y);
        // 归一化到 [-1, 1] 供 grid_sample 使用
        pt_acc[0][0][i][0] = (x / shape[0]) * 2.0f - 1.0f; // X
        pt_acc[0][0][i][1] = (y / shape[1]) * 2.0f - 1.0f; // Y
    }

    // 2. 初始化 Flow Tensor [1, 2, Ly, Lx]

    std::vector<cv::Mat> dP_channels;
    cv::split(dP, dP_channels); // 0:dY, 1:dX

    // 转为 Tensor
    torch::Tensor dX = torch::from_blob(dP_channels[1].data, { Ly, Lx }, dP.depth() == CV_64F ? torch::kFloat64 : torch::kFloat32).to(device).clone();
    torch::Tensor dY = torch::from_blob(dP_channels[0].data, { Ly, Lx }, dP.depth() == CV_64F ? torch::kFloat64 : torch::kFloat32).to(device).clone();

    if (dX.dtype() == torch::kFloat64) {
        dX = dX.to(torch::kFloat32);
        dY = dY.to(torch::kFloat32);
    }

    // 归一化 Flow 
    dX.mul_(2.0f / shape[0]);
    dY.mul_(2.0f / shape[1]);

    torch::Tensor flow_tensor = torch::zeros({ 1, ndim, Ly, Lx }, torch::kFloat32).to(device);
    flow_tensor.index_put_({ 0, 0 }, dX);
    flow_tensor.index_put_({ 0, 1 }, dY); 

    // 3. 动力学迭代 
    for (int t = 0; t < niter; ++t) {
        torch::Tensor dPt = torch::nn::functional::grid_sample(
            flow_tensor,
            pt_tensor,
            torch::nn::functional::GridSampleFuncOptions().align_corners(false).padding_mode(torch::kBorder)
        );
        dPt = dPt.permute({ 0, 2, 3, 1 });

        pt_tensor.add_(dPt);
        pt_tensor.clamp_(-1.0f, 1.0f);
    }

    // 4. 反归一化
    pt_tensor.add_(1.0f).mul_(0.5f);
    pt_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0 }).mul_(shape[0]);
    pt_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1 }).mul_(shape[1]);

    // 5. 格式化输出
    cv::Mat final_points(N_seeds, 1, CV_32FC2);
    torch::Tensor final_cpu = pt_tensor.squeeze().cpu().contiguous(); // [N, 2] (X, Y)


    float* ptr = reinterpret_cast<float*>(final_points.data);
    const float* tensor_ptr = final_cpu.data_ptr<float>();

    for (int i = 0; i < N_seeds; ++i) {
        ptr[2 * i] = tensor_ptr[2 * i + 1];   // Y
        ptr[2 * i + 1] = tensor_ptr[2 * i];   // X
    }

    return final_points;
}


cv::Mat follow_flows(const cv::Mat& dP, const std::vector<cv::Point>& inds, int niter) {
    return steps_interp(dP, inds, niter);
}

// =================================================================================
// Mask Generation & Postprocessing
// =================================================================================
// 用于存储种子信息
struct SeedInfoLocal {
    cv::Point coord;
    int hval;
};


cv::Mat max_pool_nd_opencv(const cv::Mat& input, int ksize) {
    if (input.empty()) return input;

    // 5x5 的最大池化
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(ksize, ksize),
        cv::Point(ksize / 2, ksize / 2) 
    );

    cv::Mat output;
    // cv::dilate 在单通道矩阵上执行最大值滤波操作
    cv::dilate(input, output, kernel);
    return output;
}

// 移除过大掩膜并重编号 
void remove_large_masks_and_renumber(cv::Mat& M0_32S, double max_size_fraction) {
    if (M0_32S.empty()) return;

    std::map<int, int> counts;
    for (int r = 0; r < M0_32S.rows; ++r) {
        const int* data = M0_32S.ptr<const int>(r);
        for (int c = 0; c < M0_32S.cols; ++c) {
            if (data[c] != 0) {
                counts[data[c]]++;
            }
        }
    }

    // 2. 移除大掩膜 和 小掩膜
    double total_pixels = M0_32S.rows * M0_32S.cols;
    int max_pixels = static_cast<int>(total_pixels * max_size_fraction);
    int min_pixels = 10; 
    std::set<int> labels_to_remove;

    for (const auto& pair : counts) {
        if (pair.second > max_pixels || (pair.second < min_pixels && pair.first != 0)) {
            labels_to_remove.insert(pair.first);
        }
    }

    // 清除大/小掩膜
    for (int r = 0; r < M0_32S.rows; ++r) {
        int* data = M0_32S.ptr<int>(r);
        for (int c = 0; c < M0_32S.cols; ++c) {
            if (labels_to_remove.count(data[c])) {
                data[c] = 0;
            }
        }
    }

    // 3. 重编号
    std::map<int, int> label_map;
    int new_label = 1;
    cv::Mat M0_copy = M0_32S.clone();

    for (int r = 0; r < M0_32S.rows; ++r) {
        const int* old_data = M0_copy.ptr<const int>(r);
        int* new_data = M0_32S.ptr<int>(r);
        for (int c = 0; c < M0_32S.cols; ++c) {
            int old_label = old_data[c];
            if (old_label != 0) {
                if (label_map.find(old_label) == label_map.end()) {
                    label_map[old_label] = new_label++;
                }
                new_data[c] = label_map[old_label];
            }
            else {
                new_data[c] = 0;
            }
        }
    }
}

// ====================================================================
// 核心函数 get_masks
// ====================================================================

cv::Mat get_masks(const cv::Mat& p_final_input, const std::vector<cv::Point>& inds,
    cv::Size original_shape, double max_size_fraction) {

    if (p_final_input.empty() || inds.empty()) {
        return cv::Mat::zeros(original_shape, CV_16UC1);
    }

    const int rpad = 20;
    cv::Size padded_shape(original_shape.width + 2 * rpad, original_shape.height + 2 * rpad);

    cv::Mat h1 = cv::Mat::zeros(padded_shape, CV_32S);
    std::vector<cv::Point> p_final_padded_int(inds.size());

    // --- 1. 直方图统计 ---
    for (size_t i = 0; i < inds.size(); ++i) {
        // 直接读取 float (CV_32FC2)
        cv::Vec2f pos_vec;
        if (p_final_input.depth() == CV_32F) {
            pos_vec = p_final_input.at<cv::Vec2f>(i, 0);
        }
        else {
            cv::Vec2d v = p_final_input.at<cv::Vec2d>(i, 0);
            pos_vec = cv::Vec2f((float)v[0], (float)v[1]);
        }

        float final_y = pos_vec[0]; // Y (from steps_interp modified)
        float final_x = pos_vec[1]; // X

        if (std::isnan(final_y) || std::isinf(final_y) || std::isnan(final_x) || std::isinf(final_x)) {
            p_final_padded_int[i] = cv::Point(-1, -1);
            continue;
        }

        int y = static_cast<int>(std::floor(final_y) + rpad);
        int x = static_cast<int>(std::floor(final_x) + rpad);

        // 边界检查
        if (y < 0 || y >= padded_shape.height || x < 0 || x >= padded_shape.width) {
            p_final_padded_int[i] = cv::Point(-1, -1);
            continue;
        }

        h1.at<int>(y, x)++;
        p_final_padded_int[i] = cv::Point(x, y);
    }

    // --- 2. 寻找种子点 (Seeds) ---
    cv::Mat h1_float;
    h1.convertTo(h1_float, CV_32F);

    cv::Mat hmax1 = max_pool_nd_opencv(h1_float, 5); // Kernel=5

    std::vector<SeedInfoLocal> seeds_info;
    const float* hmax_ptr = (float*)hmax1.data;
    const int* h1_ptr = (int*)h1.data;
    const float* h1_f_ptr = (float*)h1_float.data;

    int total_pixels = padded_shape.height * padded_shape.width;

    // 展平
    for (int i = 0; i < total_pixels; ++i) {
        if (h1_ptr[i] > 10) { // Threshold > 10
            if (h1_f_ptr[i] - hmax_ptr[i] > -1e-6f) { // Local Max
                int r = i / padded_shape.width;
                int c = i % padded_shape.width;
                seeds_info.push_back({ cv::Point(c, r), h1_ptr[i] });
            }
        }
    }

    if (seeds_info.empty()) {
        return cv::Mat::zeros(original_shape, CV_16UC1);
    }

    // --- 2.1 种子去重 ---
    std::sort(seeds_info.begin(), seeds_info.end(), [](const SeedInfoLocal& a, const SeedInfoLocal& b) {
        return a.hval > b.hval;
        });

    std::vector<cv::Point> seeds1;
    int min_dist2 = 30 * 30; 

    for (const auto& s : seeds_info) {
        bool keep = true;
        for (const auto& fs : seeds1) {
            int dx = s.coord.x - fs.x;
            int dy = s.coord.y - fs.y;
            if (dx * dx + dy * dy < min_dist2) {
                keep = false;
                break;
            }
        }
        if (keep) seeds1.push_back(s.coord);
    }

    // --- 3. Voronoi 聚类 ---
    int n_seeds = (int)seeds1.size();
    std::vector<int> pixel_label(inds.size(), 0);

    // 最近邻搜索
    for (size_t i = 0; i < inds.size(); ++i) {
        const cv::Point& pf = p_final_padded_int[i];
        if (pf.x < 0) continue; // Skip invalid

        int best_k = -1;
        int min_d2 = std::numeric_limits<int>::max();

        for (int k = 0; k < n_seeds; ++k) {
            int dx = pf.x - seeds1[k].x;
            int dy = pf.y - seeds1[k].y;
            int d2 = dx * dx + dy * dy;
            if (d2 < min_d2) {
                min_d2 = d2;
                best_k = k;
            }
        }
        if (best_k >= 0) pixel_label[i] = best_k + 1;
    }

    // --- 4. 映射回原图 ---
    cv::Mat M0_32S = cv::Mat::zeros(original_shape, CV_32SC1);
    for (size_t i = 0; i < inds.size(); ++i) {
        int lbl = pixel_label[i];
        if (lbl > 0) {
            M0_32S.at<int>(inds[i].y, inds[i].x) = lbl;
        }
    }

    // --- 5. 移除大区域并重编号 ---
    remove_large_masks_and_renumber(M0_32S, max_size_fraction);

    cv::Mat M0_16U;
    M0_32S.convertTo(M0_16U, CV_16UC1);
    return M0_16U;
}

cv::Mat fill_holes_single_mask(const cv::Mat& mask_binary) {
    if (mask_binary.empty()) return cv::Mat();

    cv::Mat mask_u8;
    if (mask_binary.type() != CV_8U) {
        mask_binary.convertTo(mask_u8, CV_8U, 255.0 / 1.0); // 转换为 0/255
    }
    else {
        mask_u8 = mask_binary;
    }

    // 1. 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_u8, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    // 2. 填充内部轮廓
    cv::Mat filled_mask = cv::Mat::zeros(mask_u8.size(), CV_8U);
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(filled_mask, contours, (int)i, cv::Scalar(255), cv::FILLED);
    }
    return filled_mask;
}

cv::Mat fill_holes_and_remove_small_masks(const cv::Mat& masks_in, int min_size) {
    if (masks_in.empty()) return masks_in;

    cv::Mat masks;
    masks_in.convertTo(masks, CV_32S);

    // 1. 统计各标签大小
    std::map<int, int> counts;
    for (int r = 0; r < masks.rows; ++r) {
        const int* row = masks.ptr<int>(r);
        for (int c = 0; c < masks.cols; ++c) {
            int v = row[c];
            if (v > 0) counts[v]++;
        }
    }

    // 2. 根据大小筛选、重新编号
    std::map<int, int> newID;
    int id = 1;
    for (auto& kv : counts) {
        if (kv.second >= min_size)
            newID[kv.first] = id++;
        else
            newID[kv.first] = 0;
    }

    // 3. 应用重编号
    cv::Mat out = cv::Mat::zeros(masks.size(), CV_32S);
    for (int r = 0; r < masks.rows; ++r) {
        const int* src = masks.ptr<int>(r);
        int* dst = out.ptr<int>(r);
        for (int c = 0; c < masks.cols; ++c) {
            int v = src[c];
            if (v > 0) dst[c] = newID[v];
        }
    }

    return out;
}

cv::Mat compute_masks(const cv::Mat& dP, const cv::Mat& cellprob, float cellprob_threshold, int niter, float max_size_fraction, int min_size) {
    // 1. 阈值化 Cellprob
    cv::Mat mask_pixels_8u;
    cv::threshold(cellprob, mask_pixels_8u, cellprob_threshold, 255, cv::THRESH_BINARY);
    mask_pixels_8u.convertTo(mask_pixels_8u, CV_8U);

    std::vector<cv::Point> inds;
    cv::findNonZero(mask_pixels_8u, inds);

    if (inds.empty()) {
        return cv::Mat::zeros(cellprob.size(), CV_16U);
    }

    // 2. Flow 预处理
    cv::Mat dP_f;
    if (dP.depth() != CV_32F) dP.convertTo(dP_f, CV_32FC2);
    else dP_f = dP.clone();

    std::vector<cv::Mat> dP_ch;
    cv::split(dP_f, dP_ch);

    cv::Mat mask_f;
    mask_pixels_8u.convertTo(mask_f, CV_32F, 1.0 / 255.0);

    for (int k = 0; k < 2; ++k) {
        cv::multiply(dP_ch[k], 0.2f, dP_ch[k]); // Div 5.0
        cv::multiply(dP_ch[k], mask_f, dP_ch[k]);
    }

    cv::merge(dP_ch, dP_f);

    // 3. 动力学迭代
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
    }

    cv::Mat p_final = steps_interp(dP_f, inds, niter, device);

    // 4. 生成 Mask
    cv::Mat mask_16U = get_masks(p_final, inds, cellprob.size(), max_size_fraction);

    // 5. 后处理：填补孔洞 & 移除小区域
    
    cv::Mat final_mask = fill_holes_and_remove_small_masks(mask_16U, min_size);

    return final_mask;
}

cv::Mat resize_and_compute_masks(const cv::Mat& dP, const cv::Mat& cellprob, int min_size) {
    cv::Mat mask = compute_masks(dP, cellprob, 0, 200, 0.4, -1);

    if (min_size <= 0) return mask;

    // 后处理: 移除小掩码
    std::map<ushort, int> label_counts;
    for (int r = 0; r < mask.rows; ++r) {
        for (int c = 0; c < mask.cols; ++c) {
            ushort label = mask.at<ushort>(r, c);
            if (label > 0) label_counts[label]++;
        }
    }
    std::cout << "mask1" << "rows: " << mask.rows << ", cols: " << mask.cols
        << ", channels: " << mask.channels()
        << ", type: " << mask.type() << std::endl;
    for (int r = 0; r < mask.rows; ++r) {
        for (int c = 0; c < mask.cols; ++c) {
            ushort& label = mask.at<ushort>(r, c);
            if (label > 0 && label_counts[label] < min_size) {
                label = 0; 
            }
        }
    }
    std::cout << "mask2" << "rows: " << mask.rows << ", cols: " << mask.cols
        << ", channels: " << mask.channels()
        << ", type: " << mask.type() << std::endl;
    return mask;
}

cv::Mat Cellpose_img2mask(const std::string& onnx_model_path, const cv::Mat& img, float rsz)
{
    // 网络推理
    cv::Mat flows, style;
    RunNetResult result = run_net(onnx_model_path, img, 1, false, 0.1, 224, 1.0);
    flows = result.flows_and_prob;

    // 掩膜计算
    std::vector<cv::Mat> flows_ch;
    cv::split(flows, flows_ch); // flows_ch[0]=Y-flow, flows_ch[1]=X-flow, flows_ch[2]=Cell-prob
    std::vector<cv::Mat> dP_channels = { flows_ch[0], flows_ch[1] };
    cv::Mat dP;
    cv::merge(dP_channels, dP); // dP: CV_32FC2 (H x W)
    cv::Mat cellprob = flows_ch[2]; // cellprob: CV_32F (H x W)
    cv::Mat masks = resize_and_compute_masks(dP, cellprob, 15);

    return masks;
}