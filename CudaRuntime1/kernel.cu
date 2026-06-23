#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <windows.h>    // 引入 Windows API
#include <commdlg.h>    // 引入通用对话框 API

#include <chrono>  // 引入 chrono 精细计时器

//图片操作库，仅头文件
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ========== 类型定义 ==========
struct ImageData {
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 3;

    ImageData() = default;
    ImageData(const std::string& path) { load(path); }
    ~ImageData() { if (data) stbi_image_free(data); }

    bool load(const std::string& path) {
        if (data) { stbi_image_free(data); data = nullptr; }
        data = stbi_load(path.c_str(), &width, &height, &channels, 3);
        channels = 3;
        return data != nullptr;
    }

    bool save(const std::string& path) const {
        return stbi_write_png(path.c_str(), width, height, 3, data, width * 3) != 0;
    }

    size_t size() const { return width * height * 3; }
    size_t byteSize() const { return size() * sizeof(unsigned char); }
};

// ========== 类型定义 ==========
struct PerformanceResult {
    float gpu_time_ms = 0;
    float cpu_time_ms = 0;
    double speedup = 0;
    double gpu_gflops = 0;
    double cpu_gflops = 0;

    void print(const std::string& operation) const {
        printf("\n--- %s 性能对比 ---\n", operation.c_str());
        printf("GPU运行时间: %.3f ms\n", gpu_time_ms);
        printf("CPU运行时间: %.3f ms\n", cpu_time_ms);
        printf("GPU加速比: %.2fx\n", speedup);
        printf("GPU浮点运算性能: %.2f GFLOPS\n", gpu_gflops);
        printf("CPU浮点运算性能: %.2f GFLOPS\n", cpu_gflops);
        printf("----------------\n");
    }
};

// ========== 工具函数：选择输入文件 ==========
std::string OpenFileDialog() {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileName(&ofn)) {
        return std::string(szFile);
    }
    return "";
}

// ========== 工具函数：选择输出文件 ==========
std::string SaveFileDialog() {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "PNG Files\0*.png\0JPEG Files\0*.jpg\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

    if (GetSaveFileName(&ofn)) {
        return std::string(szFile);
    }
    return "";
}

// ========== 工具核函数 浮点转整数 ==========
__global__ void float_to_uchar_kernel(
    const float* input, unsigned char* output,
    int width, int height) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 3;

    if (idx < total) {
        float val = input[idx];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        output[idx] = static_cast<unsigned char>(val);
    }
}

// ========== 工具函数 ==========
__device__ int calculateOutputSizeDevice(int input_size, int kernel_size, int stride) {
    return (input_size - kernel_size) / stride + 1;
}


int calculateOutputSize(int input_size, int kernel_size, int stride) {
    return (input_size - kernel_size) / stride + 1;
}

void checkCudaError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(error));
    }
}

// ========== CUDA 核函数 RGB卷积 ==========
__global__ void convolution_rgb_kernel(
    const unsigned char* input,         //输入
    float* output,                      //输出
    int width,//输入的宽
    int height,//输入的高
    const float* kernel, //卷积核
    int kernel_size,//卷积核大小
    int stride, //步长
    int channel//当前处理的通道（0-R, 1-G, 2-B）
) {
    //通过块索引*块大小+线程索引确定单个线程在整个块图的坐标x,y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // (原始宽 - 核长) / 步长 + 1  是核在图片中移动到最后时能处理并输出的像素
    int out_width = calculateOutputSizeDevice(width, kernel_size, stride);
    int out_height = calculateOutputSizeDevice(height, kernel_size, stride);
    //边界不处理
    if (x < out_width && y < out_height) {

        //累加器
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                //  输出目标位置 (x, y) * 步长 + 核内偏移（核长决定） -> 获取原图应该处理的像素坐标
                int in_x = x * stride + kx;
                int in_y = y * stride + ky;

                //跳过 y 行 偏移 x列到达目标位置，RGB图像要*3.依据处理通道取出对应像素通道
                int idx = (in_y * width + in_x) * 3 + channel;
                //卷积计算，对像素 * 卷积
                sum += static_cast<float>(input[idx]) * kernel[ky * kernel_size + kx];
            }
        }
        //输出图(x,y)的 "channel" 数据为sum
        output[(y * out_width + x) * 3 + channel] = sum;
    }
}

// ========== CUDA 核函数 RGB最大池化 ==========
__global__ void max_pooling_rgb_kernel(
    const float* input, float* output,
    int in_width, int in_height,
    int out_width, int out_height,
    int pool_size, int stride) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        for (int c = 0; c < 3; ++c) {
            float max_val = -1e38f;
            for (int py = 0; py < pool_size; ++py) {
                for (int px = 0; px < pool_size; ++px) {
                    int in_x = x * stride + px;
                    int in_y = y * stride + py;
                    if (in_x < in_width && in_y < in_height) {
                        float val = input[(in_y * in_width + in_x) * 3 + c];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            output[(y * out_width + x) * 3 + c] = max_val;
        }
    }
}

// ========== CUDA 核函数 RGB平均池化 ==========
__global__ void avg_pooling_rgb_kernel(
    const float* input, float* output,
    int in_width, int in_height,
    int out_width, int out_height,
    int pool_size, int stride) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            int count = 0;
            for (int py = 0; py < pool_size; ++py) {
                for (int px = 0; px < pool_size; ++px) {
                    int in_x = x * stride + px;
                    int in_y = y * stride + py;
                    if (in_x < in_width && in_y < in_height) {
                        sum += input[(in_y * in_width + in_x) * 3 + c];
                        count++;
                    }
                }
            }
            output[(y * out_width + x) * 3 + c] = sum / count;
        }
    }
}

// ========== CPU 实现函数 ==========
void convolution_rgb_cpu(const unsigned char* input, float* output,
    int width, int height,
    const float* kernel, int kernel_size,
    int stride) {

    int out_width = calculateOutputSize(width, kernel_size, stride);
    int out_height = calculateOutputSize(height, kernel_size, stride);

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_x = x * stride + kx;
                        int in_y = y * stride + ky;
                        if (in_x < 0) in_x = 0;
                        if (in_x >= width) in_x = width - 1;
                        if (in_y < 0) in_y = 0;
                        if (in_y >= height) in_y = height - 1;
                        int idx = (in_y * width + in_x) * 3 + c;
                        sum += static_cast<float>(input[idx]) * kernel[ky * kernel_size + kx];
                    }
                }
                output[(y * out_width + x) * 3 + c] = sum;
            }
        }
    }
}

void max_pooling_rgb_cpu(const float* input, float* output,
    int in_width, int in_height,
    int out_width, int out_height,
    int pool_size, int stride) {

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float max_val = -1e38f;
                for (int py = 0; py < pool_size; ++py) {
                    for (int px = 0; px < pool_size; ++px) {
                        int in_x = x * stride + px;
                        int in_y = y * stride + py;
                        if (in_x < in_width && in_y < in_height) {
                            float val = input[(in_y * in_width + in_x) * 3 + c];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                output[(y * out_width + x) * 3 + c] = max_val;
            }
        }
    }
}

void avg_pooling_rgb_cpu(const float* input, float* output,
    int in_width, int in_height,
    int out_width, int out_height,
    int pool_size, int stride) {

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                int count = 0;
                for (int py = 0; py < pool_size; ++py) {
                    for (int px = 0; px < pool_size; ++px) {
                        int in_x = x * stride + px;
                        int in_y = y * stride + py;
                        if (in_x < in_width && in_y < in_height) {
                            sum += input[(in_y * in_width + in_x) * 3 + c];
                            count++;
                        }
                    }
                }
                output[(y * out_width + x) * 3 + c] = sum / count;
            }
        }
    }
}

// ========== 通用性能测试框架 ==========
template<typename GPUFunc, typename CPUFunc>
PerformanceResult benchmarkOperation(
    const unsigned char* input_image,
    int width, int height,
    int out_width, int out_height,
    GPUFunc gpu_func,
    CPUFunc cpu_func,
    double ops_per_pixel,
    unsigned char*& gpu_output_image,
    bool verbose = true) {

    PerformanceResult result;

    // ===== GPU实现 =====
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // 分配GPU内存
    size_t input_size = width * height * 3 * sizeof(unsigned char);
    size_t output_size = out_width * out_height * 3 * sizeof(float);

    unsigned char* d_input;
    float* d_output;
    unsigned char* d_result;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_result, out_width * out_height * 3);

    cudaMemcpy(d_input, input_image, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size);

    // 执行GPU操作
    cudaEventRecord(start_gpu);
    gpu_func(d_input, d_output, width, height, out_width, out_height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cudaEventElapsedTime(&result.gpu_time_ms, start_gpu, stop_gpu);

    // 转换结果
    int total_threads = out_width * out_height * 3;
    int blocks = (total_threads + 255) / 256;
    float_to_uchar_kernel << <blocks, 256 >> > (d_output, d_result, out_width, out_height);
    cudaDeviceSynchronize();

    // 分配内存并复制GPU结果
    gpu_output_image = new unsigned char[out_width * out_height * 3];
    cudaMemcpy(gpu_output_image, d_result, out_width * out_height * 3,
        cudaMemcpyDeviceToHost);

    // ===== CPU实现 =====
    float* cpu_input_float = new float[width * height * 3];
    for (int i = 0; i < width * height * 3; ++i) {
        cpu_input_float[i] = static_cast<float>(input_image[i]);
    }

    float* cpu_output = new float[out_width * out_height * 3];

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_func(cpu_input_float, cpu_output, width, height, out_width, out_height);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    result.cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ===== 计算性能指标 =====
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    double total_ops = (double)out_width * out_height * 3 * ops_per_pixel;
    result.gpu_gflops = total_ops / (result.gpu_time_ms / 1000.0) / 1e9;
    result.cpu_gflops = total_ops / (result.cpu_time_ms / 1000.0) / 1e9;

    // 清理
    delete[] cpu_input_float;
    delete[] cpu_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return result;
}

// ========== 卷积操作封装 ==========
bool applyConvolutionWithBenchmark(const std::string& input_path,
    const std::string& output_path,
    const std::vector<float>& kernel,
    int stride = 1,
    bool verbose = true) {

    ImageData image(input_path);
    if (!image.data) {
        printf("读取图片失败: %s\n", input_path.c_str());
        return false;
    }

    int kernel_size = (int)sqrt(kernel.size());
    int out_width = calculateOutputSize(image.width, kernel_size, stride);
    int out_height = calculateOutputSize(image.height, kernel_size, stride);

    if (verbose) {
        printf("图片: %dx%d, 卷积核: %dx%d, 输出: %dx%d\n",
            image.width, image.height, kernel_size, kernel_size, out_width, out_height);
    }

    // 准备GPU函数
    auto gpu_conv = [&](unsigned char* d_input, float* d_output,
        int width, int height, int out_w, int out_h) {

            float* d_kernel;
            size_t kernel_size_bytes = kernel.size() * sizeof(float);
            cudaMalloc(&d_kernel, kernel_size_bytes);
            cudaMemcpy(d_kernel, kernel.data(), kernel_size_bytes, cudaMemcpyHostToDevice);

            const int BLOCK_SIZE = 16;
            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocksPerGrid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

            for (int c = 0; c < 3; ++c) {
                convolution_rgb_kernel << <blocksPerGrid, threadsPerBlock >> > (
                    d_input, d_output, width, height, d_kernel, kernel_size, stride, c);
            }
            cudaDeviceSynchronize();

            cudaFree(d_kernel);
        };

    // 准备CPU函数
    auto cpu_conv = [&](float* cpu_input, float* cpu_output,
        int width, int height, int out_w, int out_h) {

            unsigned char* cpu_input_uchar = new unsigned char[width * height * 3];
            for (int i = 0; i < width * height * 3; ++i) {
                cpu_input_uchar[i] = static_cast<unsigned char>(cpu_input[i]);
            }

            convolution_rgb_cpu(cpu_input_uchar, cpu_output, width, height,
                kernel.data(), kernel_size, stride);

            delete[] cpu_input_uchar;
        };

    // 执行性能测试
    double ops_per_pixel = kernel_size * kernel_size * 2; // 乘和加
    unsigned char* gpu_result = nullptr;
    PerformanceResult result = benchmarkOperation(
        image.data, image.width, image.height,
        out_width, out_height,
        gpu_conv, cpu_conv, ops_per_pixel, gpu_result, verbose);

    if (verbose) {
        result.print("卷积");
    }

    // 保存GPU结果到文件
    if (gpu_result) {
        bool saved = stbi_write_png(output_path.c_str(), out_width, out_height, 3,
            gpu_result, out_width * 3) != 0;
        if (saved) {
            printf("GPU处理结果已保存到: %s\n", output_path.c_str());
        }
        else {
            printf("保存GPU结果失败!\n");
        }
        delete[] gpu_result;
    }

    return true;
}

// ========== 池化操作封装 ==========
bool applyMaxPoolingWithBenchmark(const std::string& input_path,
    const std::string& output_path,
    int pool_size = 2,
    int stride = 2,
    bool verbose = true) {

    ImageData image(input_path);
    if (!image.data) {
        printf("读取图片失败: %s\n", input_path.c_str());
        return false;
    }

    int out_width = calculateOutputSize(image.width, pool_size, stride);
    int out_height = calculateOutputSize(image.height, pool_size, stride);

    if (verbose) {
        printf("池化: %dx%d -> %dx%d, 池化窗口: %dx%d\n",
            image.width, image.height, out_width, out_height, pool_size, pool_size);
    }

    // 准备GPU函数
    auto gpu_pool = [&](unsigned char* d_input, float* d_output,
        int width, int height, int out_w, int out_h) {

            // 将输入转换为float
            float* d_input_float;
            size_t float_size = width * height * 3 * sizeof(float);
            cudaMalloc(&d_input_float, float_size);

            // 将unsigned char转换为float并复制到GPU
            unsigned char* h_temp = new unsigned char[width * height * 3];
            cudaMemcpy(h_temp, d_input, width * height * 3, cudaMemcpyDeviceToHost);

            float* h_input_float = new float[width * height * 3];
            for (int i = 0; i < width * height * 3; ++i) {
                h_input_float[i] = static_cast<float>(h_temp[i]);
            }

            cudaMemcpy(d_input_float, h_input_float, float_size, cudaMemcpyHostToDevice);

            delete[] h_temp;
            delete[] h_input_float;

            const int BLOCK_SIZE = 16;
            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocksPerGrid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

            max_pooling_rgb_kernel << <blocksPerGrid, threadsPerBlock >> > (
                d_input_float, d_output, width, height, out_w, out_h, pool_size, stride);
            cudaDeviceSynchronize();

            cudaFree(d_input_float);
        };

    // 准备CPU函数
    auto cpu_pool = [&](float* cpu_input, float* cpu_output,
        int width, int height, int out_w, int out_h) {

            max_pooling_rgb_cpu(cpu_input, cpu_output, width, height,
                out_w, out_h, pool_size, stride);
        };

    // 执行性能测试
    double ops_per_pixel = pool_size * pool_size; // 比较操作
    unsigned char* gpu_result = nullptr;
    PerformanceResult result = benchmarkOperation(
        image.data, image.width, image.height,
        out_width, out_height,
        gpu_pool, cpu_pool, ops_per_pixel, gpu_result, verbose);

    if (verbose) {
        result.print("最大池化");
    }

    // 保存GPU结果到文件
    if (gpu_result) {
        bool saved = stbi_write_png(output_path.c_str(), out_width, out_height, 3,
            gpu_result, out_width * 3) != 0;
        if (saved) {
            printf("GPU处理结果已保存到: %s\n", output_path.c_str());
        }
        else {
            printf("保存GPU结果失败!\n");
        }
        delete[] gpu_result;
    }

    return true;
}

// ========== 主函数 ==========
int main(int argc, char* argv[]) {
	int choice = 0;
    printf("=== CUDA图片处理性能对比 ===\n\n");

    printf("正在打开文件选择器...\n");
    std::string input_path = OpenFileDialog();

    if (input_path.empty()) {
        printf("用户取消了文件选择或选择失败。\n");
        system("pause");
        return -1;
    }

    printf("已选择输入图片: %s\n", input_path.c_str());

    size_t lastSlash = input_path.find_last_of("/\\");
    std::string output_dir = (lastSlash == std::string::npos) ? "" : input_path.substr(0, lastSlash + 1);
    printf("\n开始处理...\n\n");


    // ===== 示例1：边缘检测 =====
    printf("--- 1. 边缘检测 ---\n");
    std::vector<float> edge_kernel = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
    applyConvolutionWithBenchmark(input_path, output_dir + "边缘检测.png",
        edge_kernel, 1, true);
    printf("\n");
   
    // ===== 示例2：锐化 =====
    printf("--- 2. 锐化 ---\n");
    std::vector<float> sharpen_kernel = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    applyConvolutionWithBenchmark(input_path, output_dir + "锐化.png",
        sharpen_kernel, 1, true);
    printf("\n");
    // ===== 示例3：模糊 =====
    printf("--- 3. 模糊 ---\n");
    std::vector<float> blur_kernel = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    for (auto& v : blur_kernel) v /= 16.0f;
    applyConvolutionWithBenchmark(input_path, output_dir + "高斯模糊.png",
        blur_kernel, 1, true);
    printf("\n");
    // ===== 示例4：浮雕效果 =====
    printf("--- 4. 浮雕效果 ---\n");
    std::vector<float> emboss_kernel = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };
    applyConvolutionWithBenchmark(input_path, output_dir + "浮雕.png",
        emboss_kernel, 1, true);
    printf("\n");

    // ===== 示例5：池化 =====
    printf("--- 5. 最大值池化 ---\n");
    applyMaxPoolingWithBenchmark(input_path, output_dir + "最大池化.png", 2, 2, true);

    printf("\n所有处理完成!\n");
    printf("输出文件已保存在与原图相同的目录下。\n");

    system("pause");
    return 0;
}