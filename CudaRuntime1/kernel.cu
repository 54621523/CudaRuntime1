#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <windows.h>    // 引入 Windows API
#include <commdlg.h>    // 引入通用对话框 API




#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// ========== 工具函数：选择输入文件 ==========
std::string OpenFileDialog() {
    OPENFILENAME ofn;
    char szFile[260] = { 0 }; // 用于存储文件路径

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;  // 无父窗口
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0"; // 过滤器
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

// ========== 工具函数 浮点转整数（RGB位图格式） ==========
__global__ void float_to_uchar_kernel(
    const float* input, unsigned char* output,
    int width, int height) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 3;

    if (idx < total) {
        float val = input[idx];
        // 限制值在0-255之间
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        output[idx] = static_cast<unsigned char>(val);
    }
}

// ========== 工具函数 计算输出大小 ==========
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
    //通过块索引*块大小+线程索引确定单个线程在输出图的坐标x,y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // (原始宽 - 核长) / 步长 + 1  是核在图片中移动到最后时能处理并输出的像素
    int out_width = (width - kernel_size) / stride + 1;
    int out_height = (height - kernel_size) / stride + 1;
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

            //基于比较的最大值获取方法
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

// ========== CUDA 核函数 RGB平均池化==========
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





// ========== 图像处理函数 ==========
bool applyConvolution(const std::string& input_path, const std::string& output_path,
    const std::vector<float>& kernel, int stride = 1) {

    // 加载图片
    int width, height, channels;
    unsigned char* input_image = stbi_load(input_path.c_str(), &width, &height, &channels, 3);

    if (!input_image) {
        printf("读取图片失败: %s\n", input_path.c_str());
        return false;
    }

    printf("加载图片: %dx%d 像素, %d 通道\n", width, height, 3);

    int kernel_size = (int)sqrt(kernel.size());
    int out_width = calculateOutputSize(width, kernel_size, stride);
    int out_height = calculateOutputSize(height, kernel_size, stride);

    printf("输出大小: %dx%d\n", out_width, out_height);

    // 分配GPU内存
    unsigned char* d_input;
    float* d_output, * d_kernel;
    unsigned char* d_result;

    size_t input_size = width * height * 3 * sizeof(unsigned char);
    size_t output_size = out_width * out_height * 3 * sizeof(float);
    size_t kernel_size_bytes = kernel.size() * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_kernel, kernel_size_bytes);
    cudaMalloc(&d_result, input_size);

    // 拷贝数据到GPU
    cudaMemcpy(d_input, input_image, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernel_size_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size);

    // 配置Kernel参数
    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
        (out_height + blockSize.y - 1) / blockSize.y);

    // 对RGB三个通道分别处理
    for (int c = 0; c < 3; ++c) {
        convolution_rgb_kernel << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height,
            d_kernel, kernel_size,
            stride, c
            );
        cudaDeviceSynchronize();
        checkCudaError("Convolution kernel failed");
    }

    // 转换为unsigned char
    int total_threads = out_width * out_height * 3;
    int blocks = (total_threads + 255) / 256;
    float_to_uchar_kernel << <blocks, 256 >> > (d_output, d_result,
        out_width, out_height);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    unsigned char* output_image = new unsigned char[out_width * out_height * 3];
    cudaMemcpy(output_image, d_result, out_width * out_height * 3 * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    // 保存图片
    stbi_write_png(output_path.c_str(), out_width, out_height, 3,
        output_image, out_width * 3);
    printf("保存处理后的图片: %s\n\n", output_path.c_str());

    // 清理
    stbi_image_free(input_image);
    delete[] output_image;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_result);

    return true;
}

bool applyMaxPooling(const std::string& input_path, const std::string& output_path,
    int pool_size = 2, int stride = 2) {

    int width, height, channels;
    unsigned char* input_image = stbi_load(input_path.c_str(), &width, &height, &channels, 3);

    if (!input_image) {
        printf("Failed to load image: %s\n", input_path.c_str());
        return false;
    }

    int out_width = calculateOutputSize(width, pool_size, stride);
    int out_height = calculateOutputSize(height, pool_size, stride);

    printf("池化: %dx%d -> %dx%d\n", width, height, out_width, out_height);

    // 分配GPU内存
    float* d_input_float, * d_output;
    unsigned char* d_input, * d_result;

    size_t input_size = width * height * 3 * sizeof(unsigned char);
    size_t float_size = width * height * 3 * sizeof(float);
    size_t output_size = out_width * out_height * 3 * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_input_float, float_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_result, out_width * out_height * 3);

    // 拷贝数据并转换为float
    cudaMemcpy(d_input, input_image, input_size, cudaMemcpyHostToDevice);

    // 直接使用unsigned char作为float处理
    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
        (out_height + blockSize.y - 1) / blockSize.y);

    // 将输入复制到float数组
    cudaMemcpy(d_input_float, d_input, input_size, cudaMemcpyDeviceToDevice);

    max_pooling_rgb_kernel << <gridSize, blockSize >> > (
        d_input_float, d_output,
        width, height, out_width, out_height,
        pool_size, stride
        );
    cudaDeviceSynchronize();

    // 转换回unsigned char
    int total_threads = out_width * out_height * 3;
    int blocks = (total_threads + 255) / 256;
    float_to_uchar_kernel << <blocks, 256 >> > (d_output, d_result,
        out_width, out_height);
    cudaDeviceSynchronize();

    unsigned char* output_image = new unsigned char[out_width * out_height * 3];
    cudaMemcpy(output_image, d_result, out_width * out_height * 3,
        cudaMemcpyDeviceToHost);

    stbi_write_png(output_path.c_str(), out_width, out_height, 3,
        output_image, out_width * 3);
    printf("保存池化后的图片: %s\n\n", output_path.c_str());

    stbi_image_free(input_image);
    delete[] output_image;
    cudaFree(d_input);
    cudaFree(d_input_float);
    cudaFree(d_output);
    cudaFree(d_result);

    return true;
}

// ========== 主函数 (修改部分) ==========
int main(int argc, char* argv[]) {
    printf("=== CUDA图片处理 (图形界面版) ===\n\n");

    // 1. 弹窗选择输入图片
    printf("正在打开文件选择器...\n");
    std::string input_path = OpenFileDialog();

    if (input_path.empty()) {
        printf("用户取消了文件选择或选择失败。\n");
        system("pause");
        return -1;
    }

    printf("已选择输入图片: %s\n", input_path.c_str());

    // 2. 定义输出文件名 (基于输入文件名生成，或者也可以用 SaveFileDialog 让用户逐个选，这里为了效率自动生成)
    // 获取输入文件路径的目录部分
    size_t lastSlash = input_path.find_last_of("/\\");
    std::string output_dir = (lastSlash == std::string::npos) ? "" : input_path.substr(0, lastSlash + 1);

    // 定义输出路径
    std::string base_name = output_dir + "edge_detected.png";
    std::string sharp_name = output_dir + "sharpened.png";
    std::string blur_name = output_dir + "blurred.png";
    std::string emboss_name = output_dir + "embossed.png";
    std::string pool_name = output_dir + "pooled.png";

    // 3. 执行处理任务
    printf("\n开始处理...\n");

    // ===== 示例1：边缘检测 =====
    printf("1. 边缘检测...\n");
    std::vector<float> edge_kernel = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
    applyConvolution(input_path, base_name, edge_kernel);

    // ===== 示例2：锐化 =====
    printf("2. 锐化...\n");
    std::vector<float> sharpen_kernel = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    applyConvolution(input_path, sharp_name, sharpen_kernel);

    // ===== 示例3：模糊 =====
    printf("3. 模糊...\n");
    std::vector<float> blur_kernel = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    for (auto& v : blur_kernel) v /= 16.0f;
    applyConvolution(input_path, blur_name, blur_kernel);

    // ===== 示例4：浮雕效果 =====
    printf("4. 浮雕效果...\n");
    std::vector<float> emboss_kernel = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };
    applyConvolution(input_path, emboss_name, emboss_kernel);

    // ===== 示例5：池化 =====
    printf("5. 最大值池化...\n");
    applyMaxPooling(input_path, pool_name, 2, 2);

    printf("\n🎉 所有处理完成!\n");
    printf("输出文件已保存在与原图相同的目录下。\n");

    system("pause");
    return 0;
}