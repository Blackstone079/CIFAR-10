from models.resnet_cifar import ResNet as CIFARResNet

class ResNet8(CIFARResNet):
    def __init__(self, base_width=16, num_classes=10):
        super().__init__(blocks_per_stage=1, base_width=base_width, num_classes=num_classes)
