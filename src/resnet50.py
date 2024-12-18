import torch
import torchvision.models as models
import torch.nn as nn

net = models.segmentation.fcn_resnet50(
    pretrained=False, num_classes=2, pretrained_backbone=False
)
net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)


def convertBNtoGN(module, num_groups=16):
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        return nn.GroupNorm(
            num_groups, module.num_features, eps=module.eps, affine=module.affine
        )
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

    return module


net = convertBNtoGN(net)
