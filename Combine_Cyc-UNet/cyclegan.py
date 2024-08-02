import torch
import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=False),
#             # 添加更多的卷积层
#             nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=False),
#             # 添加更多的卷积层
#             nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),  # Using inplace=False to avoid inplace operation issues
            # Add more convolution layers if needed
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()  # Tanh activation for the output
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),  # Using inplace=False to avoid inplace operation issues
            # Add more convolution layers if needed
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.model(x)
