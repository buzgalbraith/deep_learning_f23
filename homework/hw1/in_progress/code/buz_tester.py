from mlp import *
n = 50
X =  torch.rand(size = (n,11))
y  = torch.empty(n,5).random_(2)

MLP_instance = MLP(linear_1_in_features=11, linear_1_out_features=6, f_function=elementwise_relu, 
    g_function=ellementwise_sigmoid, linear_2_in_features=6, linear_2_out_features=5)
y_hat = MLP_instance.forward(X)
test_out = bce_loss(y, y_hat)
MLP_instance.backward(test_out[1])