cmake_minimum_required(VERSION 3.13)

# -------------------------------------------------------------------------
# 检查 MSVC 热重载策略
# -------------------------------------------------------------------------
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("CellposeOnnx") 

# -------------------------------------------------------------------------
# 1. 库路径定义和查找
# -------------------------------------------------------------------------

# --- OpenCV ---
set(OpenCV_DIR "D:/lib/opencv/build")
find_package(OpenCV 4.10.0 REQUIRED)
set(OpenCV_DLL_DIR "${OpenCV_DIR}/x64/vc16/bin") # OpenCV DLL 路径

# --- ONNX Runtime 和 LibTorch  ---
find_package(Torch REQUIRED
            NO_MODULE
            PATHS "D:/lib/libtorch"
            NO_DEFAULT_PATH)

set(LIBTORCH_ONNXRUNTIME_ROOT "D:/lib/onnxruntime-win-x64-1.15.0") 
set(LIBTORCH_DLL_DIR "D:/lib/libtorch/lib") # LibTorch 和 ONNX Runtime DLL 路径

# --- C++ 运行时 DLL 路径 (此路径用于 Debug 配置，Release 配置下忽略) ---
set(VCRUNTIME_DLL_DIR "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/v143/debug_nonredist/x64/Microsoft.VC143.DebugCRT")

# =========================================================================
# 2. 目标定义
# =========================================================================

add_executable(CellposeOnnx
    "Cellpose_core.h" 
    "main.cpp" "Visualize.h" "Visualize.cpp"  "Cellpose_core.cpp")
    
if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET CellposeOnnx PROPERTY CXX_STANDARD 20)
endif()

# 链接外部库
target_link_libraries(CellposeOnnx PRIVATE
    ${OpenCV_LIBS}
    "${TORCH_LIBRARIES}"
    "${LIBTORCH_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"  # 使用 onnxruntime.lib
)

# 包含 LibTorch 和 ONNX Runtime 的根 include 路径
target_include_directories(CellposeOnnx PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    "${LIBTORCH_ONNXRUNTIME_ROOT}/include"  # 包含 onnxruntime 的头文件
)

# -------------------------------------------------------------------------
# 3. 改进 DLL 复制方案：只复制 Release 配置所需的特定 DLL
# -------------------------------------------------------------------------

# 定义一个更可靠的函数来复制单个文件
function(copy_specific_dll dll_name source_dir target_name)
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${source_dir}/${dll_name}" "$<TARGET_FILE_DIR:${target_name}>/${dll_name}"
        COMMENT "Copying x64 DLL: ${dll_name}"
    )
endfunction()

# 复制 LibTorch 和 ONNX Runtime 的 DLL 文件
copy_specific_dll("c10.dll" "${LIBTORCH_DLL_DIR}" CellposeOnnx)
copy_specific_dll("torch.dll" "${LIBTORCH_DLL_DIR}" CellposeOnnx)

# 复制 ONNX Runtime 的 DLL
copy_specific_dll("onnxruntime.dll" "${LIBTORCH_ONNXRUNTIME_ROOT}/lib" CellposeOnnx)

# -------------------------------------------------------------------------
# 4. 强制复制 C++ Debug 运行时 DLL (仅在 Debug 配置下需要)
# -------------------------------------------------------------------------
if (EXISTS ${VCRUNTIME_DLL_DIR})
    file(GLOB VCRUNTIME_DLL_FILES "${VCRUNTIME_DLL_DIR}/*.dll")
    foreach(dll ${VCRUNTIME_DLL_FILES})
        get_filename_component(dll_name "${dll}" NAME)
        add_custom_command(
            TARGET Cellpose_cpp POST_BUILD
            COMMAND $<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:CellposeOnnx>/${dll_name}">
            COMMENT "Copying Conditional Debug DLL: ${dll_name}"
        )
    endforeach()
endif()

# -------------------------------------------------------------------------
# 5. 最终 x64 平台检查 (保持不变)
# -------------------------------------------------------------------------
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "此项目应在 64 位 (x64) 平台上编译。当前平台可能为 x86。请使用 -A x64 选项生成项目。")
endif()
