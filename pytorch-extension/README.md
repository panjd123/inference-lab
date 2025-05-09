## 如何运行

```bash
cd pytorch-extension
pip install . # 安装当前目录下的包，本质上他会执行一遍 python setup.py install，里面写了关于编译的脚本
python examples/quick_start.py # 运行最简单的 CPU 和 CUDA 例子
```

## 如何开发

为了让 VSCode 的 C++ 插件能够正确识别，你需要添加以下内容到你的 `.vscode/settings.json` 文件中：

```json
"C_Cpp.default.includePath": [
    "/opt/miniconda3/lib/python3.12/site-packages/torch/include",
    "/opt/miniconda3/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
    "/opt/miniconda3/include/python3.12"
]
```

## 学习目标

- 通过 `setup.py` 掌握 Pytorch Extension 的写法
- 学习简单的 CPU/GPU 并行计算