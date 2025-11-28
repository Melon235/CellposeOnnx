## 🚀  CellposeOnnx

### I. 项目介绍

**CellposeOnnx** 是 **Cellpose** 细胞分割算法核心的 C++ 移植版本。本项目旨在提供一个**低依赖、简化算法**的解决方案，用于将 Cellpose 的全部功能集成到 C++ 原生应用程序、高通量系统中。

本项目利用 **LibTorch** 和 **OpenCV**，实现了从 ONNX 模型推理到流场动力学模拟和掩膜重建的完整分割流程，确保在高性能 C++ 环境中实现与 Python 原版算法一致的分割效果。

---

### II. 项目结构 

本项目的代码和资源文件按以下结构组织：
```dictionary
CellposeOnnx/
├── images/            # 存放用于测试的输入图像文件
├── models/            # 存放 ONNX 模型文件
├── Cellpose_core.cpp  # 核心算法实现文件 (包含主要的分割逻辑)
├── Cellpose_core.h    # 核心函数声明 (头文件)
├── main.cpp           # 程序入口文件，负责加载图像和调用核心分割函数
└── CMakeLists.txt     # CMake 构建配置文件
```
### III. 如何运行

本项目使用 **CMake** 作为构建系统。在執行前，請確保您的環境已安裝 **CMake**、**OpenCV**、**LibTorch** 和 **ONNX Runtime** C++ 庫。

#### 步骤 1: 环境配置和准备

1.  **安裝依赖：** 确保所有必需的 C++ 库（LibTorch, OpenCV, ONNX Runtime）已正确安装并配置在系统路径中。
2.  **模型设置：** 将Cellpose ONNX 模型文件放入 `models/` 目录中。
3.  **环境变量：** 如果您的 LibTorch 或其他依賴沒有全局安裝，请设置全局变量

#### 步骤 2: 编译

在项目根目录下执行以下命令：

1.  **创建并进入 `build` 目录：**
    ```bash
    mkdir build
    cd build
    ```

2.  **运行 CMake 配置：**
    ```bash
    # 推荐使用 Release 模式以获得最佳性能
    cmake .. -DCMAKE_BUILD_TYPE=Release 
    ```

3.  **编译項目：**
    ```bash
    # 使用 -j 选项利用多核加速编译
    cmake --build . --config Release -- 
    ```

#### 步骤 3: 运行程序

编译成功后，可执行文件 `CellposeOnnx` 將生成在 `build` 目錄下。

1.  **运行测试：**
    ```bash
    ./CellposeOnnx
    ```
2.  程序将自动加载 `models/` 中的 ONNX 模型，对 `images/` 中的指定图像进行分割，并输出結果。
