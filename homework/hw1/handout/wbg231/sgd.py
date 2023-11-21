import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm
from typing import Callable
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
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


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
    current_iteration = 1
    eta = 0.01
    optimizer = torch.optim.SGD([input], lr=eta)
    #gb = transforms.GaussianBlur(kernel_size=3)
    use_built_in = False ## for now. 

    while current_iteration <= iterations:
        y_hat = model(input)
        loss_value = -loss(y_hat)  
        model.zero_grad() ## zero the gradient
        loss_value.backward() ## compute the gradient
        gradient = input.grad.data / input.grad.data.std() ## normalize the gradient
        input.data = input.data + eta * gradient ## update the input
        input.data = torch.clamp(input.data, 0, 1) ## clamp the values between 0 and 1
        with torch.no_grad():
            input_data_np = input[0].permute(1, 2, 0).cpu().detach().numpy() ## convert to numpy
            input_data_np = cv.GaussianBlur(input_data_np, (0, 0), .5) ## blur the image
            input[0] = torch.tensor(input_data_np).permute(2, 0, 1).to(input.device) ## convert back to tensor
        input.data = normalize_and_jitter(input.data) ## normalize and jitter the image

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
