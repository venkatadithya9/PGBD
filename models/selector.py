from models.wresnet import *
from models.preactresnet import *
from models.resnet import *
from models.combat_models import *
from torchvision.models import vgg19_bn, convnext_tiny, resnet50
import os


def select_model(
    dataset,
    model_name,
    pretrained=False,
    pretrained_models_path=None,
    n_classes=10,
    ml_mmdr=False,
    all2all=False,
):
    print(model_name)
    if model_name == "WRN-16-1":
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name == "WRN-16-2":
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == "WRN-40-1":
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name == "WRN-40-2":
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == "WRN-10-2":
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == "WRN-10-1":
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=n_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=n_classes)
    elif model_name == "vgg19_bn":
        model = vgg19_bn(num_classes=n_classes)
    elif model_name == "convnext_tiny":
        model = convnext_tiny(num_classes=n_classes)
    elif model_name == "preactresnet18":
        model = PreActResNet18(num_classes=n_classes)
    elif model_name == "preactresnet34":
        model = PreActResNet34(num_classes=n_classes)
    elif model_name == "preactresnet50":
        model = PreActResNet50(num_classes=n_classes)
    elif model_name == "combat_gen":
        model = UnetGenerator()
    else:
        raise NotImplementedError

    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print("Loading Model from {}".format(model_path))
        print(ml_mmdr)
        checkpoint = torch.load(model_path, map_location="cpu")
        if ml_mmdr:
            print("Using ML-MMDR model for this run.")
            path = "../Multi-Level-MMD-Regularization/weights/CIFAR10_preactresnet18_badnet_mlmmdr_0.1_all.pt"
            checkpoint = torch.load(path, map_location="cpu")
            print(checkpoint.keys())
        if all2all:
            path = model_path.split(".p")[0] + "_all2all.p" + model_path.split(".p")[1]
            print("Using all2all model", path)
            checkpoint = torch.load(path, map_location="cpu")
        if (
            model_name == "vgg19_bn"
            or model_name == "preactresnet18"
            or model_name == "preactresnet34"
            or model_name == "preactresnet50"
            or model_name == "convnext_tiny"
            or model_name == "resnet50"
        ):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint["state_dict"])
            # print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})".format(model_path, checkpoint['epoch'], checkpoint['best_prec']))
            print(
                "=> loaded checkpoint '{}' (epoch {}) ".format(
                    model_path, checkpoint["epoch"]
                )
            )

    return model


if __name__ == "__main__":

    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(
        torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    )

    t0 = time.time()
    model = select_model("CIFAR10", model_name="WRN-16-2")
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
