#include <torch/extension.h>

__global__ void setup_kernel(float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index] * 2.0f;
    }
}

torch::Tensor setup_kernel_launch(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    setup_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("setup_kernel_launch", &setup_kernel_launch, "Setup kernel launch function");
}