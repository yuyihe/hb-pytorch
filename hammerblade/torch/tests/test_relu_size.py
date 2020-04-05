import torch
import torch.nn as nn


def test_relu_size_1():
    x = torch.randn(10)
    x_h = x.hammerblade()
    relu = nn.ReLU()
    x_relu = relu(x)
    x_h_relu = relu(x_h)
    assert x_h_relu.device == torch.device("hammerblade")
    assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_2():
#     x = torch.randn(20)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_3():
#     x = torch.randn(30)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)

# def test_relu_size_4():
#     x = torch.randn(40)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)

# def test_relu_size_5():
#     x = torch.randn(50)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_6():
#     x = torch.randn(60)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_7():
#     x = torch.randn(70)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)

# def test_relu_size_8():
#     x = torch.randn(80)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_9():
#     x = torch.randn(90)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


# def test_relu_size_10():
#     x = torch.randn(100)
#     x_h = x.hammerblade()
#     relu = nn.ReLU()
#     x_relu = relu(x)
#     x_h_relu = relu(x_h)
#     assert x_h_relu.device == torch.device("hammerblade")
#     assert torch.equal(x_h_relu.cpu(), x_relu)


