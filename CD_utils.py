import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from statistics import mean
import torch.utils.data as tutils
from torchvision.models import resnet18, vgg19_bn
from torchvision.models import resnet50

# from utils_train import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from N3F.feature_extractor.lib.baselines import DINO, get_model

# from N3F.feature_extractor.lib.baselines_v2 import DINOv2, get_model_v2
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import torchvision.transforms as tfs
from utils import Normalizer
from inversion_torch import PixelBackdoor
from PIL import Image
import copy
from tqdm import tqdm
import pickle as pkl
import glob
import time
import sys

term_width = 317

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    cur_time - last_time
    last_time = cur_time
    cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def get_pairs(pois_type, num_csets, diff_pois=False):
    pairs = dict()
    lis = []
    for i in range(num_csets):
        if "trig" in pois_type:
            lis.append((pois_type, f"{pois_type}_clean{i}"))
        else:
            lis.append((pois_type, f"clean{i}"))
    pairs[pois_type] = lis
    return pairs


class EncoMapPreactResnet_18(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapPreactResnet_18, self).__init__()
        self.conv = nn.ConvTranspose2d(64, 512, (3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoMapPreactResnet_18(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapPreactResnet_18, self).__init__()
        self.conv = nn.Conv2d(512, 64, (3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapPreactResnet_18_large(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapPreactResnet_18_large, self).__init__()
        self.conv = nn.ConvTranspose2d(64, 512, (5, 5), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoMapPreactResnet_18_large(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapPreactResnet_18_large, self).__init__()
        self.conv = nn.Conv2d(512, 64, (5, 5), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapPreactResnet_18_non_linear(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapPreactResnet_18_non_linear, self).__init__()
        self.conv = nn.ConvTranspose2d(64, 512, (3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DecoMapPreactResnet_18_non_linear(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapPreactResnet_18_non_linear, self).__init__()
        self.conv = nn.Conv2d(512, 64, (3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class EncoMapWResNet_16_1(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapWResNet_16_1, self).__init__()
        self.conv = nn.ConvTranspose2d(64, 64, (5, 5), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoMapWResNet_16_1(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapWResNet_16_1, self).__init__()
        self.conv = nn.Conv2d(64, 64, (5, 5), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapVGG19BN(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapVGG19BN, self).__init__()
        self.conv = nn.Conv2d(64, 512, (3, 3), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoMapVGG19BN(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapVGG19BN, self).__init__()
        self.conv = nn.ConvTranspose2d(512, 64, (3, 3), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapResNet_50(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapResNet_50, self).__init__()
        self.conv = nn.Conv2d(64, 2048, (7, 7), stride=1)

    def forward(self, x):
        x = self.conv(x)
        # print("Post encoding shape", x.shape)
        return x


class DecoMapResNet_50(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapResNet_50, self).__init__()
        self.conv = nn.ConvTranspose2d(2048, 64, (7, 7), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapPreactResnet_34(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapPreactResnet_34, self).__init__()
        self.conv = nn.Conv2d(64, 512, (3, 3), padding=1)

    def forward(self, x):
        x = self.conv(x)
        # print("Post encoding shape", x.shape)
        return x


class DecoMapPreactResnet_34(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapPreactResnet_34, self).__init__()
        self.conv = nn.ConvTranspose2d(512, 64, (3, 3), padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoMapPreactResnet_50(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(EncoMapPreactResnet_50, self).__init__()
        self.conv = nn.Conv2d(64, 2048, (3, 3), padding=1)

    def forward(self, x):
        x = self.conv(x)
        # print("Post encoding shape", x.shape)
        return x


class DecoMapPreactResnet_50(nn.Module):
    """
    Mainly for mapping to match dims
    """

    def __init__(self):
        super(DecoMapPreactResnet_50, self).__init__()
        self.conv = nn.ConvTranspose2d(2048, 64, (3, 3), padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def create_targets_bd(targets, opt):
    if opt.target_type == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.target_type == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.target_type))
    return bd_targets.to(opt.device)


def low_freq(x, opt):
    image_size = opt.input_height
    ratio = opt.dct_ratio
    mask = torch.zeros_like(x)
    mask[:, :, : int(image_size * ratio), : int(image_size * ratio)] = 1
    x_dct = dct_2d((x + 1) / 2 * 255)
    x_dct *= mask
    x_idct = (idct_2d(x_dct) / 255 * 2) - 1
    return x_idct


def get_mapping_module(model_name, dataset_name="CIFAR10"):
    if model_name == "vgg19_bn":
        downconv_m = EncoMapVGG19BN()
        upconv_m = DecoMapVGG19BN()
    if model_name == "WRN-16-1":
        downconv_m = EncoMapWResNet_16_1()
        upconv_m = DecoMapWResNet_16_1()
    if model_name == "preactresnet18":
        downconv_m = EncoMapPreactResnet_18()
        upconv_m = DecoMapPreactResnet_18()
        # if dataset_name == "tinyImagenet":
        #     downconv_m = EncoMapPreactResnet_18_large()
        #     upconv_m = DecoMapPreactResnet_18_large()
        # downconv_m = EncoMapPreactResnet_18_non_linear()
        # upconv_m = DecoMapPreactResnet_18_non_linear()
    if model_name == "preactresnet34":
        downconv_m = EncoMapPreactResnet_34()
        upconv_m = DecoMapPreactResnet_34()
    if model_name == "preactresnet50":
        downconv_m = EncoMapPreactResnet_50()
        upconv_m = DecoMapPreactResnet_50()
    if model_name == "resnet50":
        downconv_m = EncoMapResNet_50()
        upconv_m = DecoMapResNet_50()
    return upconv_m, downconv_m


# Save class wise prototypes
def save_classwise_protos(model, data, opt, bss=35, use_knn_proto=True):
    model.eval()
    print("new normal protos")
    # for param in model.parameters():
    #     param.requires_grad = False
    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    with torch.no_grad():
        for bottleneck_name in bottlenecks:
            proto_dic = {}
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]

                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out

                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                handle = named_layers[bottleneck_name].register_forward_hook(
                    save_activation_hook
                )
                # print(c, len(X_c_dataloader), len(inds))
                for idx, inp in enumerate(X_c_dataloader):
                    imgs, _ = inp
                    with torch.no_grad():
                        _ = model(imgs.cuda())
                    acts.append(bn_activation)
                # print("acts SHAPE", len(acts), len(acts[0]), acts[0][0])
                # print()
                # return
                acts = torch.concat(acts, axis=0)
                ss = acts.shape
                print(ss)
                flat_dim = 1
                ss_len = len(ss)
                for i in range(1, len(ss)):
                    flat_dim *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    if opt.use_kmedoids:
                        kmeans = KMedoids(n_clusters=opt.knn_k, random_state=0).fit(
                            acts
                        )
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                        else:
                            proto_dic[c] = {}
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (
                        torch.mean(acts, axis=0).unsqueeze(0).detach().cpu().numpy()
                    )
            protos[bottleneck_name] = proto_dic
    # for param in model.parameters():
    #     param.requires_grad = True
    return protos


def save_classwise_protos_delta(
    model, data, opt, prev_protos, bss=35, use_knn_proto=True
):
    model.eval()
    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    with torch.no_grad():
        for bottleneck_name in bottlenecks:
            proto_dic = {}
            prev_proto_dic = prev_protos[bottleneck_name]
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]

                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out

                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                handle = named_layers[bottleneck_name].register_forward_hook(
                    save_activation_hook
                )
                for idx, inp in enumerate(X_c_dataloader):
                    imgs, _ = inp
                    with torch.no_grad():
                        _ = model(imgs.cuda())
                    acts.append(bn_activation)
                acts = torch.concat(acts, axis=0)
                ss = acts.shape
                flat_dim_ss = 1
                for i in range(1, len(ss)):
                    flat_dim_ss *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim_ss))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if len(ss) == 4:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1], ss[2], ss[3]) + (
                                    1 - opt.delta
                                ) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1]) + (1 - opt.delta) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )

                        else:
                            proto_dic[c] = {}
                            if len(ss) == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (opt.delta) * torch.mean(acts, axis=0).unsqueeze(
                        0
                    ).detach().cpu().numpy() + (1 - opt.delta) * (prev_proto_dic[c])
            protos[bottleneck_name] = proto_dic
    return protos


def save_classwise_protos_dino(model, data, opt, bss=35, use_knn_proto=True):
    model.eval()
    print("new")
    # for param in model.parameters():
    #     param.requires_grad = False
    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    with torch.no_grad():
        for bottleneck_name in bottlenecks:
            proto_dic = {}
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]
                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                # handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                # print(c, len(X_c_dataloader), len(inds))
                acts = get_dino_activations(X_c_dataloader, opt.device)
                print(acts.shape, type(acts))

                acts = torch.concat([acts], axis=0)
                # return
                ss = acts.shape
                # print(ss)
                flat_dim = 1
                ss_len = len(ss)
                for i in range(1, len(ss)):
                    flat_dim *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                        else:
                            proto_dic[c] = {}
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (
                        torch.mean(acts, axis=0).unsqueeze(0).detach().cpu().numpy()
                    )
            protos[bottleneck_name] = proto_dic
    # for param in model.parameters():
    #     param.requires_grad = True
    return protos


def save_classwise_protos_dino_mm(
    model, data, opt, bss=35, use_knn_proto=True, upconvs=None, downconvs=None
):
    model.eval()
    print("new ")
    # for param in model.parameters():
    #     param.requires_grad = False

    pairs_lis = dict()
    pairs_lis["clean0"] = [("", "")]

    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    if upconvs == None or downconvs == None:
        upconvs, downconvs = {}, {}
    # with torch.no_grad():
    for bottleneck_name in bottlenecks:
        if len(upconvs) == 0 or len(downconvs) == 0:
            dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"
            student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"
            upconv_m, downconv_m = map_activation_spaces(
                dino_acts_path, student_acts_path, pairs_lis, 150, opt
            )
            upconvs[bottleneck_name] = upconv_m
            downconvs[bottleneck_name] = downconv_m
        else:
            upconv_m = upconvs[bottleneck_name]
            downconv_m = downconvs[bottleneck_name]

        proto_dic = {}
        with torch.no_grad():
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]

                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out

                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                handle = named_layers[bottleneck_name].register_forward_hook(
                    save_activation_hook
                )
                for idx, inp in enumerate(X_c_dataloader):
                    imgs, _ = inp
                    with torch.no_grad():
                        _ = model(imgs.cuda())
                    acts.append(upconv_m(bn_activation))
                # print(acts.shape, type(acts))

                acts = torch.concat(acts, axis=0)
                # return
                ss = acts.shape
                # print(ss)
                flat_dim = 1
                ss_len = len(ss)
                for i in range(1, len(ss)):
                    flat_dim *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                        else:
                            proto_dic[c] = {}
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (
                        torch.mean(acts, axis=0).unsqueeze(0).detach().cpu().numpy()
                    )
        protos[bottleneck_name] = proto_dic
    # for param in model.parameters():
    #     param.requires_grad = True
    return protos, upconvs, downconvs


def save_classwise_protos_dino_delta(
    model, data, opt, prev_protos, bss=35, use_knn_proto=True
):
    model.eval()
    print("new")
    # for param in model.parameters():
    #     param.requires_grad = False
    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    with torch.no_grad():
        for bottleneck_name in bottlenecks:
            proto_dic = {}
            prev_proto_dic = prev_protos[bottleneck_name]
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]
                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                # handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
                # print(c, len(X_c_dataloader), len(inds))
                acts = get_dino_activations(X_c_dataloader, opt.device)
                print(acts.shape, type(acts))

                acts = torch.concat([acts], axis=0)
                # return
                ss = acts.shape
                # print(ss)
                flat_dim = 1
                ss_len = len(ss)
                for i in range(1, len(ss)):
                    flat_dim *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if len(ss) == 4:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1], ss[2], ss[3]) + (
                                    1 - opt.delta
                                ) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1]) + (1 - opt.delta) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )
                        else:
                            proto_dic[c] = {}
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (opt.delta) * torch.mean(acts, axis=0).unsqueeze(
                        0
                    ).detach().cpu().numpy() + (1 - opt.delta) * (prev_proto_dic[c])
                    # proto_dic[c] = torch.mean(acts,axis=0).unsqueeze(0).detach().cpu().numpy()
            protos[bottleneck_name] = proto_dic
    # for param in model.parameters():
    #     param.requires_grad = True
    return protos


def save_classwise_protos_dino_mm_delta(
    model,
    data,
    opt,
    prev_protos,
    bss=35,
    use_knn_proto=True,
    upconvs=None,
    downconvs=None,
):
    model.eval()
    print("new")
    # for param in model.parameters():
    #     param.requires_grad = False
    pairs_lis = dict()
    pairs_lis["clean0"] = [("", "")]

    class_labels = [i for i in range(opt.num_class)]
    protos = {}
    named_layers = dict(model.named_modules())
    bottlenecks = opt.bottlenecks
    if upconvs == None or downconvs == None:
        upconvs, downconvs = {}, {}
    # with torch.no_grad():
    for bottleneck_name in bottlenecks:
        if len(upconvs) == 0 or len(downconvs) == 0:
            dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"
            student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"
            upconv_m, downconv_m = map_activation_spaces(
                dino_acts_path, student_acts_path, pairs_lis, 150, opt
            )
            upconvs[bottleneck_name] = upconv_m
            downconvs[bottleneck_name] = downconv_m
        else:
            upconv_m = upconvs[bottleneck_name]
            downconv_m = downconvs[bottleneck_name]
        proto_dic = {}
        prev_proto_dic = prev_protos[bottleneck_name]
        with torch.no_grad():
            for c in tqdm(class_labels):
                acts = []
                inds = [i for i in range(len(data)) if data[i][1] == c]

                def save_activation_hook(mod, inp, out):
                    global bn_activation
                    bn_activation = out

                X_c_dataloader = DataLoader(
                    data, batch_size=bss, sampler=SubsetRandomSampler(inds)
                )
                handle = named_layers[bottleneck_name].register_forward_hook(
                    save_activation_hook
                )
                for idx, inp in enumerate(X_c_dataloader):
                    imgs, _ = inp
                    with torch.no_grad():
                        _ = model(imgs.cuda())
                    acts.append(upconv_m(bn_activation))
                # print(acts.shape, type(acts))

                acts = torch.concat(acts, axis=0)
                # return
                ss = acts.shape
                print(ss)
                flat_dim = 1
                ss_len = len(ss)
                for i in range(1, len(ss)):
                    flat_dim *= ss[i]
                # giving half weightage to old proto mean and half to new proto
                if use_knn_proto and opt.knn_k != 1:
                    acts = acts.detach().cpu().numpy().reshape((-1, flat_dim))
                    kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                    centers = kmeans.cluster_centers_
                    for f in range(opt.knn_k):
                        if c in proto_dic:
                            if len(ss) == 4:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1], ss[2], ss[3]) + (
                                    1 - opt.delta
                                ) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = (
                                    opt.delta
                                ) * centers[f].reshape(ss[1]) + (1 - opt.delta) * (
                                    prev_proto_dic[c]["cluster_" + str(f)]
                                )
                        else:
                            proto_dic[c] = {}
                            if ss_len == 4:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1], ss[2], ss[3]
                                )
                            else:
                                proto_dic[c]["cluster_" + str(f)] = centers[f].reshape(
                                    ss[1]
                                )

                else:
                    proto_dic[c] = (opt.delta) * torch.mean(acts, axis=0).unsqueeze(
                        0
                    ).detach().cpu().numpy() + (1 - opt.delta) * (prev_proto_dic[c])
                    # proto_dic[c] = torch.mean(acts,axis=0).unsqueeze(0).detach().cpu().numpy()
            protos[bottleneck_name] = proto_dic
    # for param in model.parameters():
    #     param.requires_grad = True
    return protos, upconvs, downconvs


def map_activation_spaces(img_fea_path, stu_pred_path, pairs, num_imgs, opt):
    """
    Step 2: training mapping module to map activation spaces of teacher and student
    """
    upconv_m, downconv_m = get_mapping_module(opt.s_name, opt.data_name)
    mse_loss = nn.MSELoss()
    upconv_m.train()
    downconv_m.train()
    upconv_m.cuda()
    downconv_m.cuda()
    model_params = list(upconv_m.parameters()) + list(downconv_m.parameters())
    optimizer = torch.optim.Adam([p for p in model_params if p.requires_grad], lr=1e-4)
    loss_lis = []
    concepts = set()
    for p in pairs:
        concepts.add(p)
        for l1, l2 in pairs[p]:
            if l1 != "" and l2 != "":
                concepts.add(l1)
                concepts.add(l2)

    for ep in tqdm(range(5)):
        for con in tqdm(concepts):
            img_fea = torch.load(img_fea_path + f"_{con}.pt").cuda()[:num_imgs]
            stu_pred = torch.load(stu_pred_path + f"_{con}.pt").cuda()[:num_imgs]
            fea_s = img_fea.shape
            # img_fea.requires_grad = True

            for i in range(min(len(stu_pred), len(img_fea))):
                # loss = torch.tensor(0.).cuda()
                # loss.requires_grad = True
                img_ = img_fea[i]
                # img_.requires_grad = True
                x = downconv_m(img_)
                x_ = upconv_m(x)
                stu_pred_ = stu_pred[i].squeeze(0)
                # print(img_.shape, x_.shape, x.shape, stu_pred_.shape)
                assert img_.shape == x_.shape
                assert x.shape == stu_pred_.shape
                loss = mse_loss(img_, x_) + mse_loss(x, stu_pred_)
                # loss.requires_grad = True
                if i != len(img_fea) - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                assert torch.isnan(torch.tensor(loss.item())) == False
                loss_lis.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
        print("ep", ep, " loss", np.mean(loss_lis))
        path = f"./map_modules/map_{opt.s_name}_{opt.dataset}_{opt.attack_method}"
        torch.save(upconv_m.state_dict(), path + "upconv.pt")
        torch.save(downconv_m.state_dict(), path + "downconv.pt")
    return upconv_m, downconv_m


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image


def get_dataset_normalization(dataset_name):
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = tfs.Normalize(
            [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        )
    elif dataset_name == "cifar100":
        """get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151"""
        dataset_normalization = tfs.Normalize(
            [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        )
    elif dataset_name == "mnist":
        dataset_normalization = tfs.Normalize([0.5], [0.5])
    elif dataset_name == "tiny":
        dataset_normalization = tfs.Normalize(
            [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
        )
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = tfs.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == "imagenet":
        dataset_normalization = tfs.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization


def get_activations(model, dataloader, named_layers, bottleneck_name, device):
    model.eval()
    print("IN get activations: ", len(dataloader), bottleneck_name)
    acts = []
    inds = []
    act_shape = ()
    for batch_idx, data in tqdm(enumerate(dataloader)):
        if len(data) == 5:
            img, label, original_index, poison_or_not, original_target = data
            label = original_target
        elif len(data) == 2:
            img, label = data
        elif len(data) == 1:
            img = data
        img = img.to(device)

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = named_layers[bottleneck_name].register_forward_hook(
            save_activation_hook
        )
        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(out)
            act = bn_activation.detach()
            act_shape = act.shape
            acts.append(act)
            inds.append(batch_idx)
            handle.remove()
    acts = torch.concat(acts, axis=0)
    print(act_shape, np.all(torch.isnan(acts).numpy() == False), np.all(torch.isinf(acts).numpy() == False))
    return acts, inds


def get_dino_activations(dataloader, device, pca_components=64):
    model = get_model(
        model_name="dino",
        model_path="./N3F/feature_extractor/ckpts/dino_vitbase8_pretrain.pth",
        device=device,
    )
    # model = get_model_v2(model_name = "dino", model_path = "./N3F/feature_extractor/ckpts/dinov2_vits14_pretrain.pth", device=device)

    print("Extracting features...")
    # all_filenames = []
    all_features = []

    for data in tqdm(dataloader):
        if len(data) == 5:
            batch, label, original_index, poison_or_not, original_target = data
        elif len(data) == 2:
            batch, label = data
        elif len(data) == 1:
            batch = data
            label = torch.tensor(5)
        batch_feats = model.extract_features(batch, transform=False, upsample=False)
        print("Dino features extraction: ", batch.shape, batch_feats.shape)
        # all_filenames.extend(label)
        feat = batch_feats.detach().cpu()
        all_features.append(feat)

    all_features = torch.cat(all_features, 0)
    # print(feat.shape)
    pca = PCA(n_components=pca_components)
    N, C, H, W = all_features.shape
    # print(N, C, H, W)
    all_features = all_features.permute(0, 2, 3, 1)
    shape_store = all_features.shape
    all_features = all_features.view(-1, C).numpy()
    print("Features shape: ", all_features.shape)
    X = pca.fit_transform(all_features)
    print("Features shape (PCA): ", X.shape)
    X = torch.Tensor(X).view(N, H, W, pca_components).permute(0, 3, 1, 2)
    print(X.shape, np.all(torch.isnan(X).numpy() == False), np.all(torch.isinf(X).numpy() == False))
    return X


def save_acts(
    model,
    opt,
    train_data,
    train_dino_data,
    train_pois_data=None,
    train_dino_pois_data=None,
    only_student=False,
):
    ## Poisoned images vs clean images concept set
    device = opt.device
    bottlenecks = opt.bottlenecks
    num_classes = opt.num_class
    cset_len = min(150, int(len(train_data) / opt.num_csets))
    indices = [set() for i in range(num_classes)]
    for ind in range(len(train_data)):  # distributing all clean data classwise
        indices[train_data[ind][1]].add(ind)

    # Choose data to poison from available clean data
    balanced_pois_inds = []
    # if train_pois_data != None:
    #     balanced_pois_inds = []
    #     cset_len = len(train_data)*0.1 # poisoning only 0.5% of the whole dataset for pois cset
    #     for ind in range(num_classes):
    #         sel_list = set()
    #         sel_list.update(np.random.choice(list(indices[ind]), size = max(int(cset_len//num_classes),5), replace = False).tolist())
    #         indices[ind] = indices[ind] - sel_list
    #         balanced_pois_inds = balanced_pois_inds + list(sel_list)

    clean_inds_lis = []
    num_csets = opt.num_csets
    for i in range(num_csets):
        balanced_act_inds = []
        for ind in range(num_classes):
            sel_list = set()
            sel_list.update(
                np.random.choice(
                    list(indices[ind]),
                    size=max(int(cset_len // num_classes), 1),
                    replace=False,
                ).tolist()
            )
            indices[ind] = indices[ind] - sel_list
            balanced_act_inds = balanced_act_inds + list(sel_list)
        clean_inds_lis.append(balanced_act_inds)

    balanced_clean_inds = balanced_act_inds

    # balanced_act_inds = balanced_clean_inds + balanced_pois_inds
    # print(len(balanced_act_inds))

    clean_set = [Subset(train_data, clean_inds_lis[i]) for i in range(num_csets)]
    print("In save acts: ", [len(c) for c in clean_set])
    pois_set = None
    if len(balanced_pois_inds) != 0:
        pois_set = Subset(train_pois_data, balanced_pois_inds)
    # pois_set = torch.clamp(pois_set + pattern, 0, 1)
    cset = [clean_set, pois_set]

    cset_path = "./csets/" + opt.attack_method + "_" + opt.dataset
    with open(cset_path + ".pkl", "wb") as f:
        pkl.dump(cset, f)

    # with open("balanced_indices.pkl", "wb") as f:
    #     pkl.dump(balanced_act_inds, f)

    named_layers = dict(model.named_modules())
    # student_clean_acts = []
    # dino_clean_acts = []
    pairs_lis = {}
    if pois_set:
        pairs_lis["pois"] = []
    for bottleneck_name in bottlenecks:
        clean_set, pois_set = cset
        for i in range(num_csets):
            clean_dataloader = DataLoader(clean_set[i], batch_size=1)
            clean_dino_dataloader = DataLoader(
                train_dino_data,
                sampler=SubsetRandomSampler(clean_inds_lis[i]),
                batch_size=1,
            )
            acts_clean, _ = get_activations(
                model, clean_dataloader, named_layers, bottleneck_name, device
            )
            print("ACTS SHAPE: ", acts_clean.shape)
            torch.save(
                acts_clean,
                f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_clean{i}.pt",
            )
            if not only_student:
                dino_acts_clean = get_dino_activations(clean_dino_dataloader, device)
                print("DINO ACTS SHAPE: ", dino_acts_clean.shape)
                pairs_lis[f"clean{i}"] = [("", "")]
                torch.save(
                    dino_acts_clean,
                    f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_clean{i}.pt",
                )

        if pois_set:
            pois_dataloader = DataLoader(pois_set, batch_size=1)
            pois_dino_dataloader = DataLoader(
                train_dino_pois_data,
                sampler=SubsetRandomSampler(balanced_pois_inds),
                batch_size=1,
            )
            acts_pois, _ = get_activations(
                model, pois_dataloader, named_layers, bottleneck_name, device
            )
            torch.save(
                acts_pois,
                f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_pois.pt",
            )
            if not only_student:
                dino_acts_pois = get_dino_activations(pois_dino_dataloader, device)
                torch.save(
                    dino_acts_pois,
                    f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_pois.pt",
                )

                print(
                    "Pois vs clean data done, student_shape and dino_shape:",
                    acts_clean.shape,
                    dino_acts_clean.shape,
                )
    return num_csets, pairs_lis


def generate_noise(opt, seed):
    np.random.seed(seed)
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    # Fill the grid with random colors
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            base[i, j] = np.random.randint(0, 255, 3)

    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    img_size = mask.size // 3
    noise_percentage = opt.clean_noise_percentage
    noise_size = int(noise_percentage * img_size)
    random_indices = np.random.choice(img_size, noise_size)

    for ind in random_indices:
        mask[int(ind / 32)][int(ind % 32)] = (1, 1, 1)
    trig = base * mask
    return trig


def save_synthetic_acts(model, opt, train_data, train_dino_data, load_existing=False):
    device = opt.device
    bottlenecks = opt.bottlenecks
    num_classes = opt.num_class
    num_csets = opt.num_csets
    # Get trigger concept set
    trig_lis = []
    seeds = [1234, 42, 0, 1]
    normalize = Normalizer(opt.dataset)
    normalize_dino = Normalizer("dino")
    if load_existing:
        for f in glob.iglob(
            f"./trig_sets/synthetic_{opt.s_name}_{opt.attack_method}_{opt.dataset}*.png"
        ):
            trig = Image.open(f)
            trig_lis.append(tfs.functional.to_tensor(trig))
        inds = np.random.choice(len(trig_lis), opt.cset_size)
        trig_lis = [trig_lis[i] for i in inds]
    else:
        if opt.dataset == "imagenet":
            shape = (3, 224, 224)
        elif opt.dataset == "tinyImagenet":
            shape = (3, 64, 64)
        else:
            shape = (3, 32, 32)

        print("Processing label: {}".format(opt.target_label))
        backdoor = PixelBackdoor(
            model,
            shape=shape,
            batch_size=opt.batch_size,
            normalize=normalize,
            steps=100,
            augment=False,
        )

        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False)

        for g in range(5, 100, 5):
            for s in seeds:
                pattern = backdoor.generate(
                    train_loader,
                    opt.target_label,
                    attack_size=opt.attack_size,
                    gamma=g,
                    seed=s,
                )
                print(g, len(pattern))
                tf_to_img = tfs.ToPILImage()
                trig_img = tf_to_img(pattern)
                trig_img.save(
                    "./trig_sets/synthetic_"
                    + opt.s_name
                    + "_"
                    + opt.attack_method
                    + "_"
                    + opt.dataset
                    + "_g_"
                    + str(g)
                    + "_seed_"
                    + str(s)
                    + ".png"
                )
                trig_lis.append(pattern)
        inds = np.random.choice(len(trig_lis), opt.cset_size)
        trig_lis = [trig_lis[i] for i in inds]

    cset_len = max(num_classes, len(trig_lis))
    # print("********** ---- ********* cset size = ", cset_len)

    dino_trig = CustomDataset(data=[normalize_dino(x) for x in trig_lis])
    student_trig = CustomDataset(data=[normalize(x) for x in trig_lis])

    pois_dataloader = DataLoader(student_trig, batch_size=1)
    pois_dino_dataloader = DataLoader(dino_trig, batch_size=1)

    transforms = tfs.Compose(
        [
            tfs.ColorJitter(
                brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)
            ),
            tfs.RandomHorizontalFlip(p=0.85),
            tfs.RandomVerticalFlip(p=0.85),
        ]
    )
    clean_set = []
    clean_dino_set = []
    for i in range(num_csets):
        # generate random noise images
        clean_lis = []
        for s in seeds:
            clean_lis.append(tfs.functional.to_tensor(generate_noise(opt, s)))

        l = len(clean_lis)
        # print("Initial length of clean_list:", l)
        k = cset_len // l
        for i in range(l):
            for j in range(k):
                img = transforms(clean_lis[i])
                if torch.equal(img, clean_lis[i]):
                    j -= 1
                    continue
                clean_lis.append(img)

        dino_clean = CustomDataset(data=[normalize_dino(x) for x in clean_lis])
        student_clean = CustomDataset(data=[normalize(x) for x in clean_lis])
        clean_dataloader = DataLoader(student_clean, batch_size=1)
        clean_dino_dataloader = DataLoader(dino_clean, batch_size=1)
        clean_set.append(clean_dataloader)
        clean_dino_set.append(clean_dino_dataloader)

    named_layers = dict(model.named_modules())
    # student_clean_acts = []
    # dino_clean_acts = []
    for bottleneck_name in bottlenecks:
        for i in range(num_csets):
            clean_dataloader = clean_set[i]
            clean_dino_dataloader = clean_dino_set[i]
            acts_clean, _ = get_activations(
                model, clean_dataloader, named_layers, bottleneck_name, device
            )
            dino_acts_clean = get_dino_activations(clean_dino_dataloader, device)

            torch.save(
                acts_clean,
                f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig_clean{i}.pt",
            )
            torch.save(
                dino_acts_clean,
                f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig_clean{i}.pt",
            )

        acts_pois, _ = get_activations(
            model, pois_dataloader, named_layers, bottleneck_name, device
        )
        dino_acts_pois = get_dino_activations(pois_dino_dataloader, device)

        torch.save(
            acts_pois,
            f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig.pt",
        )
        torch.save(
            dino_acts_pois,
            f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig.pt",
        )


"""
    indices = [set() for i in range(num_classes)]
    for ind in range(len(train_data)): # distributing all clean data classwise
        indices[train_data[ind][1]].add(ind)
    
    clean_inds_lis = []
    balanced_act_inds = []
    for i in range(num_csets):
        balanced_act_inds = []
        for ind in range(num_classes):
            sel_list = set()
            sel_list.update(np.random.choice(list(indices[ind]), size = int(cset_len//num_classes), replace = False).tolist())
            indices[ind] = indices[ind] - sel_list
            balanced_act_inds = balanced_act_inds + list(sel_list)
        clean_inds_lis.append(balanced_act_inds)
    
    balanced_clean_inds = balanced_act_inds
    clean_set = [Subset(train_data, clean_inds_lis[i]) for i in range(num_csets)]
    pois_set = student_trig

    cset = [clean_set, pois_set]
    cset_path = "./csets/" + opt.attack_method+ "_" + opt.dataset + "_trigger"
    with open(cset_path + ".pkl", "wb") as f:
        pkl.dump(cset, f)
    named_layers = dict(model.named_modules())
    # student_clean_acts = []
    # dino_clean_acts = []
    for bottleneck_name in bottlenecks:
        clean_set, pois_set = cset
        for i in range(num_csets):
            clean_dataloader = DataLoader(clean_set[i], batch_size = 1)
            clean_dino_dataloader = DataLoader(train_dino_data, sampler = SubsetRandomSampler(clean_inds_lis[i]), batch_size = 1)
            acts_clean, _ = get_activations(model, clean_dataloader, named_layers, bottleneck_name, device)
            dino_acts_clean = get_dino_activations(clean_dino_dataloader, device)

            torch.save(acts_clean, f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig_clean{i}.pt")
            torch.save(dino_acts_clean, f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig_clean{i}.pt")
        print(len(pois_dataloader), "*********** ------------ ******")
        acts_pois, _ = get_activations(model, pois_dataloader, named_layers, bottleneck_name, device)
        dino_acts_pois = get_dino_activations(pois_dino_dataloader, device)

        torch.save(acts_pois, f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig.pt")
        torch.save(dino_acts_pois, f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}_trig.pt")


        print("Pois vs clean data done, student_shape and dino_shape:", acts_clean.shape, dino_acts_clean.shape)
"""
