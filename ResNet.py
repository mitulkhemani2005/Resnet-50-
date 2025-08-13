import torch
import torch.nn as nn

class bottleneck (nn.Module):
  def __init__(self, input_filter, output_filter, stride_value):
    super().__init__()
    self.conv1 = nn.Conv2d(input_filter, output_filter, kernel_size=1, stride = 1, padding=0)
    self.bn1 = nn.BatchNorm2d(output_filter)
    self.conv2 = nn.Conv2d(output_filter, output_filter, kernel_size=3, stride = stride_value, padding=1)
    self.bn2 = nn.BatchNorm2d(output_filter)
    self.conv3 = nn.Conv2d(output_filter, output_filter*4, kernel_size=1, stride = 1, padding=0)
    self.bn3 = nn.BatchNorm2d(output_filter*4)
    self.relu = nn.ReLU()

    if input_filter != output_filter * 4 or stride_value != 1:
      self.downsample = nn.Sequential(
          nn.Conv2d(input_filter, output_filter * 4, kernel_size=1, stride=stride_value, padding=0),
          nn.BatchNorm2d(output_filter * 4))
    else:
      self.downsample = None


  def forward(self, x):
    start = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      start = self.downsample(start)


    out = out + start
    out = self.relu(out)
    return out
  

class MyResnet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()

    #ResNet stages
    self.layer1 = nn.Sequential(
        bottleneck(64, 64, 1),
        bottleneck(256, 64, 1),
        bottleneck(256, 64, 1)
    )
    self.layer2 = nn.Sequential(
        bottleneck(256, 128, 2),
        bottleneck(512, 128, 1),
        bottleneck(512, 128, 1),
        bottleneck(512, 128, 1)
    )
    self.layer3 = nn.Sequential(
        bottleneck(512, 256, 2),
        bottleneck(1024, 256, 1),
        bottleneck(1024, 256, 1),
        bottleneck(1024, 256, 1),
        bottleneck(1024, 256, 1),
        bottleneck(1024, 256, 1)
    )
    self.layer4 = nn.Sequential(
        bottleneck(1024, 512, 2),
        bottleneck(2048, 512, 1),
        bottleneck(2048, 512, 1)
    )

    #final average pooling
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(2048, 1000)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.mpool1(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out