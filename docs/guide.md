# 项目构建指南

## 项目简介

本项目基于 [ncnn](https://github.com/Tencent/ncnn) 推理框架，结合 OpenCV、BYTETracker 等，实现了多种 YOLO 系列模型（如 YOLOv8）在 PC 端的高效推理与多目标跟踪。支持多种模型格式和后处理方式，适合目标检测、分割、姿态估计等任务。

## 依赖环境准备

### 1. 基础依赖
- **C++17** 编译器（如 g++/clang++ 7.0 及以上）
- **CMake** 3.17 及以上
- **OpenCV** 4.x（需包含 core、imgproc、highgui 模块）
- **ncnn** 推理库（已集成于 `lib/ncnn-20240410-macos-vulkan/` 目录）
- **libomp**（Mac 下需安装 OpenMP 支持）

#### Mac 下安装 OpenCV 和 libomp
```bash
brew install opencv libomp
```

#### Ubuntu 下安装 OpenCV
```bash
sudo apt update
sudo apt install libopencv-dev
```

### 2. Python 相关（模型转换/导出用）
如需自行导出 YOLO/Real-ESRGAN 等模型为 ncnn 格式，需安装：
- Python 3.7+
- torch、ultralytics、ncnn、pnnx 等

参考 [README.md](../README.md) 中的模型导出说明。

## 模型文件准备

- YOLOv8 示例模型已包含于 `yolov8/` 目录：
  - `yolov8s_ncnn.param`、`yolov8s_ncnn.bin`
- 其他模型（如 yolov8n、yolov8n.withPostProcess 等）可按需替换。
- 测试视频位于 `data/video/` 目录（如 `track2.mp4`）。

如需自定义模型，请参考 [README.md](../README.md) 的"模型导出"部分，将导出的 `.param` 和 `.bin` 文件放入对应目录，并在 `main.cpp` 中修改模型路径。

## 构建步骤

### 1. 拉取代码
```bash
git clone <your_repo_url>
cd test_ncnn2
```

### 2. 配置与编译

#### 推荐方式：CMake 构建
```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```
编译完成后，主程序为 `build/test_ncnn`。

#### 其他方式
- 也可直接在 CLion、VSCode 等 IDE 中以 CMake 项目方式打开。
- 若有 Makefile，可尝试 `make`（但推荐使用 CMake）。

### 3. 运行测试

确保 `data/video/track2.mp4` 存在，或修改 `main.cpp` 中视频路径。

```bash
./test_ncnn
```

运行后会弹出窗口，显示检测与跟踪结果。

## 常见问题与 FAQ

### Q1: OpenCV/ncnn 找不到？
- 检查 `CMakeLists.txt` 中 `find_package(OpenCV REQUIRED ...)` 是否能找到 OpenCV。
- ncnn 已集成于 `lib/ncnn-20240410-macos-vulkan/`，无需单独安装。
- Mac 下需安装 `libomp`，并确保 `/opt/homebrew/opt/libomp/` 路径存在。

### Q2: 如何更换/导出自己的模型？
- 参考 [README.md](../README.md) 的模型导出部分。
- 将导出的 `.param` 和 `.bin` 文件放入对应目录，并在 `main.cpp` 中修改模型路径。

### Q3: 运行时找不到视频文件？
- 检查 `data/video/` 目录下是否有对应视频。
- 可修改 `main.cpp`，更换为自己的测试视频。

### Q4: 其他依赖问题？
- 确保所有子模块（如 `byte_tracker/`、`yolov8/` 等）均已编译。
- 若遇到链接错误，检查 CMake 输出，确认所有依赖库路径正确。

## 参考与扩展

- 更多模型导出、后处理、定制化用法请参考 [README.md](../README.md)。
- 支持的模型类型、转换流程、常见问题等均有详细说明。

---

如有问题欢迎提 issue 或联系作者。 