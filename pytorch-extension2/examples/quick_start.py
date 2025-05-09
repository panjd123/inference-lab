import torch
import pytorch_extension2


def test_square():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = pytorch_extension2.square(x)

    print("Input:", x)
    print("Squared:", y)

    x_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y_cuda = pytorch_extension2.square(x_cuda)
    print("Input (CUDA):", x_cuda)
    print("Squared (CUDA):", y_cuda)


def test_matmul_rc():
    """
            3 6
            2 5
            1 4
    1 2 3
    4 5 6
    """
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b = torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32)
    print("torch:", torch.matmul(a, b.T))
    print("pytorch_extension:", pytorch_extension2.matmul_rowmajor_columnmajor(a, b))

    a_cuda = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda")
    b_cuda = torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32, device="cuda")
    print("torch (CUDA):", torch.matmul(a_cuda, b_cuda.T))
    print(
        "pytorch_extension (CUDA):",
        pytorch_extension2.matmul_rowmajor_columnmajor(a_cuda, b_cuda),
    )


test_square()
test_matmul_rc()
