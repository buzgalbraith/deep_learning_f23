import torch
from typing import Tuple

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = elementwise_relu if f_function=="relu" else (ellementwise_sigmoid if f_function == "sigmoid" else ellementwise_identity)
        self.g_function = elementwise_relu if g_function=="relu" else (ellementwise_sigmoid if g_function == "sigmoid" else ellementwise_identity)
        self.f_prime =  relu_prime if f_function=="relu" else (sigmoid_prime if f_function == "sigmoid" else identity_prime)
        self.g_prime = relu_prime if g_function=="relu" else (sigmoid_prime if g_function == "sigmoid" else identity_prime)
        
        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()
    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        #import ipdb; ipdb.set_trace()
        self.cache["x"] = x  
        c_1 = x @ self.parameters['W1'].T + self.parameters['b1'] ## first forward pass 
        self.cache["c_1"] = c_1
        z_1 = self.f_function(c_1) ## pass to relu
        self.cache['z_1'] = z_1
        c_2 = z_1 @ self.parameters["W2"].T  + self.parameters['b2'] ## second forward pass
        self.cache['c_2'] = c_2
        y_tilde = self.g_function(c_2) ## pass to sigmoid 
        self.cache["y_tilde"] = y_tilde
        return y_tilde 


    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        ### output layer 
        dyhat_dc2 = self.g_prime(self.cache['c_2'])  # derivative of y_hat with respect to c_2
        dc2_dW2 = self.cache['z_1']  # derivative of c_2 with respect to W_2
        batch_size = dJdy_hat.shape[0]
        dc2_db2 = torch.eye(self.parameters['b2'].shape[0])  # derivative of c_2 with respect to b_2
        self.grads['dJdW2'] = (dJdy_hat * dyhat_dc2).T @ dc2_dW2 / batch_size  # Correct derivative of J with respect to W_2
        self.grads['dJdb2'] = dc2_db2 @ torch.sum(dJdy_hat * dyhat_dc2, dim=0) / batch_size # Correct derivative of J with respect to b_2
        ## input layer 
        dc2_dz1 = self.parameters['W2']  # derivative of c_2 with respect to z_1
        dz1_dc1 = self.f_prime(self.cache['c_1'])  # derivative of z_1 with respect to c_1
        dc1_dW1 = self.cache['x']  # derivative of c_1 with respect to W_1
        dc1_db1 = torch.eye(self.parameters['b1'].shape[0])  # derivative of c_1 with respect to b_1
        self.grads['dJdW1'] = (((dJdy_hat * dyhat_dc2)@ dc2_dz1) * dz1_dc1).T @ dc1_dW1 / batch_size  # Correct derivative of J with respect to W_1
        self.grads['dJdb1'] = torch.sum((((dJdy_hat * dyhat_dc2)@ dc2_dz1) * dz1_dc1), dim=0) @ dc1_db1 / batch_size  # Correct derivative of J with respect to W_1


    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y:torch.tensor, y_hat:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = torch.mean((y_hat - y)**2) # mean squared error 
    djdy_hat = -2 * (y - y_hat)  / y.shape[1] ## gradient is $2/n * (~y-y)$
    return J, djdy_hat
    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = torch.mean(-y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)) # binary cross entropy
    djdy_hat = (y_hat - y) / (y_hat * (1 - y_hat)) / y.shape[1] ## here we are scaling by y.shape[1] 
    return J, djdy_hat
def elementwise_relu(x:torch.tensor)->torch.tensor:
    """ applies the relu function elementwise to x
    Args:
        x: tensor of any shape
    Return:
        y: tensor of same shape as x
    """
    return torch.max(torch.zeros_like(x), x) 
def relu_prime(x):
    """applies the derivative of the relu function elementwise to x"""
    return torch.where(x>0, 1.0, 0.0)
def ellementwise_sigmoid(x:torch.tensor)->torch.tensor:
    """ applies the sigmoid function elementwise to x
    Args:
        x: tensor of any shape
    Return:
        y: tensor of same shape as x
    """
    return 1/(1+torch.exp(-x))
def sigmoid_prime(x):
    """applies the derivative of the sigmoid function elementwise to x"""
    return ellementwise_sigmoid(x) * (1 - ellementwise_sigmoid(x))
def ellementwise_identity(x):
    return x 
def identity_prime(x):
    """applies the derivative of the identity function elementwise to x
    """
    return torch.ones_like(x)








