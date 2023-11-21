import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from typing import Callable
from tqdm import tqdm
import torch.optim as optim
import torchvision 
DEVICE = "cpu"  # "cuda"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects [BGR] (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(), iterations=256) ## finds the mean of the tensor at the label
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(dim = 1)[0, label].item()}") 
        print(torch.argsort(out, dim=1, descending=True)[0, :10]) ## prints the top 10 labels
        # import ipdb; ipdb.set_trace()
        break

# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )
def gradient_descent(input, model, loss_fn, iterations=256, learning_rate=0.01, blur_sigma=0.5):
    for _ in range(iterations):

        output = model(input)
        loss = -loss_fn(output)  

        model.zero_grad()
        loss.backward()

        gradient = input.grad.data / input.grad.data.std()

        input.data += learning_rate * gradient

        input.data = torch.clamp(input.data, 0, 1)

        with torch.no_grad():
            input_data_np = input[0].permute(1, 2, 0).cpu().detach().numpy()
            input_data_np = cv.GaussianBlur(input_data_np, (0, 0), blur_sigma)
            input[0] = torch.tensor(input_data_np).permute(2, 0, 1).to(input.device)

        input.data = normalize_and_jitter(input.data)

    return input
#         input.data = normalize_and_jitter(input.data)
#     return input

# def gradient_descent(input, model, loss, iterations=256):
#     lr = 0.01
#     optimizer = torch.optim.SGD([input], lr=lr)

#     gb = transforms.GaussianBlur(kernel_size=3)

#     for i in tqdm(range(iterations)):
#         # with torch.no_grad():
#         #     input.data = gb.forward(input.data)

#         optimizer.zero_grad()

#         normalized_input = normalize_and_jitter(input)

#         output = model(normalized_input)
#         current_loss = - loss(output)
#         #print(current_loss.item())
#         current_loss.backward()

#         # torch.nn.utils.clip_grad_value_(input.grad, 0.1)

#         optimizer.step()

#         input.data.clamp_(0, 1)
#     return input

def gradient_descent(input:torch.Tensor, model:torch.nn.Module, loss:Callable, iterations:int =256)->torch.Tensor:
    """given an input, model and loss preforms gradient descent for a set number of iterations on the image to minimize the loss.  
    
    Args: 
        input (torch.tensor) : image input (image we are trying to optime ) 
        model (torch.nn.Module) : model to optimize
        loss  (Callable ) : loss function to optimize model with respect to. 
        iterations (int, optional) : number of iterations (no early stopping for now )
    Returns: 
        input (torch.Tensor) 
    """
    input.requires_grad_(True)
    iteration_number = 1
    eta = 0.0001 ## our learning rate 
    optimizer =optim.SGD([input], lr = eta)
    built_in = False
    gn_added = torchvision.transforms.GaussianBlur(kernel_size=5)
    input = normalize_and_jitter(input)
    while iteration_number<=iterations:
        model.zero_grad()
        input = gn_added(input) ## add Gaussian noise
        input = torch.clamp(input=input, min=0, max= 1) ## clamp pixel values between zero and one.        
        #import ipdb; ipdb.set_trace()
        y_hat = model(input) 
        loss_value = -loss(y_hat)
        temp = torch.autograd.grad(loss_value, input)[0]
        grad = temp / temp.std()
        print("at iteration {0} loss value is {1}".format(iteration_number, round(loss_value.item(),4)))
        # grad = input.grad.data / input.grad.data.std()
        input = input + eta * grad
        iteration_number += 1
    input = normalize_and_jitter(input)
    return input



def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
