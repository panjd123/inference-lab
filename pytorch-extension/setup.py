# setup.py
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="pytorch_extension",
    packages=["pytorch_extension"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "pytorch_extension._C",
            [
                "pytorch_extension/extension_cpu.cpp",
                "pytorch_extension/extension_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp", "-march=native"],
                "nvcc": ["-O3", "--use_fast_math", "-Wno-deprecated-gpu-targets"],
            },
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
