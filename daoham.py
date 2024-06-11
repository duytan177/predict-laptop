import numpy as np
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Đạo hàm của sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hàm tanh
def tanh(x):
    return np.tanh(x)

# Đạo hàm của tanh
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Hàm ReLU
def relu(x):
    return np.maximum(0, x)

# Đạo hàm của ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Hàm Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# Đạo hàm của Leaky ReLU
def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# Hàm ELU
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Đạo hàm của ELU
def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

# Tạo dữ liệu đầu vào
x = np.linspace(-5, 5, 100)

# Vẽ đồ thị và đạo hàm của các hàm kích hoạt
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, sigmoid_derivative(x), label='Sigmoid derivative')
plt.title('Sigmoid')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, tanh_derivative(x), label='Tanh derivative')
plt.title('Tanh')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, relu_derivative(x), label='ReLU derivative')
plt.title('ReLU')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, leaky_relu_derivative(x), label='Leaky ReLU derivative')
plt.title('Leaky ReLU')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x, elu(x), label='ELU')
plt.plot(x, elu_derivative(x), label='ELU derivative')
plt.title('ELU')
plt.legend()

plt.tight_layout()
plt.show()
