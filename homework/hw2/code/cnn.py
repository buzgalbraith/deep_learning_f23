import torch.nn as nn
import torch
class CNN_Solution(torch.nn.Module):

from torch import nn
import torch.nn.functional as torch.nn.functional

class CNN_Solution(nn.Module):

  def __init__(self, num_features=64, num_features_2=128):
    super().__init__()

    self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=3, padding=1)
    self.conv_layer_2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)

    self.conv_layer_3 = nn.Conv2d(in_channels=num_features, out_channels=num_features_2, kernel_size=3, padding=1)
    self.conv_layer_4 = nn.Conv2d(in_channels=num_features_2, out_channels=num_features_2, kernel_size=3, padding=1)

    self.downsample_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=1, stride=4, bias=torch.nn.functionalalse),
        nn.BatchNorm2d(num_features)
    )

    self.downsample_layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=num_features, out_channels=num_features_2, kernel_size=1, stride=4, bias=torch.nn.functionalalse),
        nn.BatchNorm2d(num_features_2)
    )

    self.batch_norm_1 = nn.BatchNorm2d(num_features)
    self.batch_norm_2 = nn.BatchNorm2d(num_features)

    self.batch_norm_3 = nn.BatchNorm2d(num_features_2)
    self.batch_norm_4 = nn.BatchNorm2d(num_features_2)

    self.fc_layer = nn.Linear(num_features_2, 2)

  def forward(self, x):
    z1 = self.conv_layer_1(x)
    a1 = torch.nn.functional.relu(z1)
    bn1 = self.batch_norm_1(a1)
    x1 = torch.nn.functional.max_pool2d(bn1, 2)

    x2 = conv_layer_2(x1)
    x2 = torch.nn.functional.relu(x2)
    x2 = self.batch_norm_2(x2)
    x2 = torch.nn.functional.max_pool2d(x2, 2)

    x2 = x2 + self.downsample_layer_1(x)

    x3 = self.conv_layer_3(x2)
    x3 = torch.nn.functional.relu(x3)
    x3 = self.batch_norm_3(x3)
    x3 = torch.nn.functional.max_pool2d(x3, 2)

    x4 = self.conv_layer_4(x3)
    x4 = torch.nn.functional.relu(x4)
    x4 = self.batch_norm_4(x4)
    x4 = torch.nn.functional.max_pool2d(x4, 2)

    x4 = x4 + self.downsample_layer_2(x2)

    x5 = torch.nn.functional.avg_pool2d(x4, x4.size()[2:])
    x5 = self.fc(torch.squeeze(x5))

    return x5