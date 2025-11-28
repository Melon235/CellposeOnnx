#include <opencv2/opencv.hpp>
#include <filesystem>
#include "Cellpose_core.h"
#include "Visualize.h"

int main() {
    std::filesystem::path root_path = R"(C:\Users\86180\Desktop\Cellpose\CellposeOnnx)";
    std::filesystem::path img_full_path = root_path / "images" / "test.png";
    std::filesystem::path model_full_path = root_path / "models" / "cyto3.onnx";
    std::string img_path = img_full_path.string();
    std::string model_path = model_full_path.string();

    // 读取图像
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Error reading image: " << img_path << std::endl;
        return -1;
    }

    // 模型运行
    cv::Mat masks;
    masks = Cellpose_img2mask(model_path, img, 1);
   
    // 可视化结果
    cv::Mat original_color_img = img.clone();
    if (original_color_img.channels() == 1) {
        cv::cvtColor(original_color_img, original_color_img, cv::COLOR_GRAY2BGR);
    }

    // --- 彩色填充叠加 ---
    cv::Mat color_mask_img = masks_to_color_image(masks);
    cv::Mat fill_overlay_result = overlay_masks_on_image(original_color_img, color_mask_img, 0.6);

    // --- 轮廓叠加  ---
    cv::Scalar contour_color(0, 255, 0); 
    cv::Mat contour_overlay_result = draw_contours_on_image(original_color_img, masks, contour_color, 2);

    cv::imshow("1. Original Image", original_color_img);
    cv::imshow("2. Color Fill Overlay (Alpha=0.6)", fill_overlay_result);
    cv::imshow("3. Contour Overlay (Green, 2px)", contour_overlay_result);
    cv::waitKey(0);


    return 0;
}
