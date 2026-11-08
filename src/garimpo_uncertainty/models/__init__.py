"""The segmentation models: the ConvNet (smp's DeepLabV3) and a U-Net, both on one ResNet-18."""

# Shared by both models, so the comparison isolates the decoder: skip connections or none.
ENCODER = "resnet18"
