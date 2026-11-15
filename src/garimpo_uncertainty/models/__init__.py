"""The segmentation models: the ConvNet (smp's DeepLabV3) and a U-Net, both on one ResNet-18."""

# Shared by both models; they still differ in decoder and output stride, 32 against DeepLabV3's 8.
ENCODER = "resnet18"
