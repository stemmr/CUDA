from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("setup_kernel.cu").read_text()
    cpp_source = "torch::Tensor setup_kernel_launch(torch::Tensor input);"

    # Load the CUDA kernel as a PyTorch extension
    setup_extension = load_inline(
        name="setup_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["setup"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return setup_extension


def main():
    """
    Use torch cpp inline extension function to compile the kernel in mean_filter_kernel.cu.
    Read input image, convert apply mean filter custom cuda kernel and write result out into output.png.
    """
    ext = compile_extension()

    input_data = torch.randn(1024, device="cuda")

    output_data = ext.setup(input_data)


if __name__ == "__main__":
    main()
