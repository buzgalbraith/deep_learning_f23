import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
gelu = torch.nn.GELU()
relu = torch.nn.ReLU()
leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

soft_plus = torch.nn.Softplus(beta=1)
x_range = torch.linspace(-10, 10, 1000)
ax , fig = plt.subplots(ncols=2,nrows=2)
def relu_deriv(x):
    return (x > 0) * 1.0
def gelu_deriv(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))
def leaky_relu_deriv(x):
    return (x > 0) * 1.0 + (x < 0) * 0.1
def softplus_deriv(x):
    return 1 / (1 + torch.exp(-x))
# gelu_deriv = torch.autograd.grad(gelu_x, x_range, create_graph=True)[0]
fig[0,0].plot(x_range, gelu_deriv(x_range), label='GELU')
fig[0,1].plot(x_range, leaky_relu_deriv(x_range), label='GELU')
fig[1,0].plot(x_range, relu_deriv(x_range), label='RELU')
fig[1,1].plot(x_range, softplus_deriv(x_range), label='GELU')

fig[0,0].set_title('GELU Derivative')
fig[0,1].set_title('Leaky RELU Derivative')
fig[1,0].set_title('RELU Derivative')
fig[1,1].set_title('Softplus Derivative')
plt.show()

