import random
import copy

from data_loader import get_backdoor_loader
from data_loader import get_train_loader, get_test_loader, DatasetBD
from inversion_torch import PixelBackdoor
from utils.util import *
from utils_train import *
from CD_utils import *
from models.selector import *
from config import (
    get_arguments,
    get_arguments_1,
    get_arguments_2,
    get_arguments_3,
    get_arguments_2_preact,
    get_arguments_3_preact,
    get_arguments_4_preact,
    get_arguments_signal_preact,
    get_arguments_wanet_preact,
)
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torchvision.datasets import ImageFolder
import tqdm
import matplotlib.pyplot as plt
from utils import Normalizer, Denormalizer
import argparse
from argparse import ArgumentParser
import wandb
from ROF import ROF
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_act_classwise(model, data, opt, dino=False):
    acts_classwise = {}
    proto_dic = {}
    class_labels = [i for i in range(opt.num_class)]
    named_layers = dict(model.named_modules())
    for bottleneck_name in opt.bottlenecks:
        for c in tqdm.tqdm(class_labels):
            acts = []
            inds = [i for i in range(len(data)) if data[i][1] == c]

            def save_activation_hook(mod, inp, out):
                global bn_activation
                bn_activation = out

            X_c_dataloader = DataLoader(
                data, batch_size=35, sampler=SubsetRandomSampler(inds)
            )
            if dino == False:
                handle = named_layers[bottleneck_name].register_forward_hook(
                    save_activation_hook
                )
                for idx, inp in enumerate(X_c_dataloader):
                    imgs, _ = inp
                    with torch.no_grad():
                        _ = model(imgs.cuda())
                    acts.append(bn_activation)
                acts = torch.concat(acts, axis=0)
            else:
                acts = get_dino_activations(X_c_dataloader, opt.device)
                torch.concat([acts])
            acts_classwise[c] = acts.detach().cpu().numpy().reshape(acts.shape[0], -1)
            ss = acts.shape
            ss_len = len(ss)
            # print(acts_classwise[c].shape)

            flat_dim_ss = 1
            for i in range(1, len(ss)):
                flat_dim_ss *= ss[i]
            # giving half weightage to old proto mean and half to new proto
            if opt.knn_k != 1:
                acts = acts.detach().cpu().numpy().reshape((-1, flat_dim_ss))
                kmeans = KMeans(n_clusters=opt.knn_k, random_state=0).fit(acts)
                centers = kmeans.cluster_centers_
                for f in range(opt.knn_k):
                    if c in proto_dic:
                        if ss_len == 4:
                            proto_dic[c] += centers[f].reshape(ss[1], ss[2], ss[3])
                        else:
                            proto_dic[c] += centers[f].reshape(ss[1])
                    else:
                        proto_dic[c] = {}
                        if ss_len == 4:
                            proto_dic[c] = centers[f].reshape(ss[1], ss[2], ss[3])
                        else:
                            proto_dic[c] = centers[f].reshape(ss[1])
                proto_dic[c] /= opt.knn_k
            else:
                proto_dic[c] = (
                    torch.mean(acts, axis=0).unsqueeze(0).detach().cpu().numpy()
                )
            proto_dic[c] = proto_dic[c].reshape(1, -1)
            acts_classwise[c] = np.vstack([acts_classwise[c], proto_dic[c]])
    return acts_classwise, proto_dic


def tsne_vis(opt, p, e, acts_classwise, acts_classwise_pois, epoch, seed=42):
    pca = PCA(n_components=8)
    for c in range(opt.num_class):
        acts_classwise[c] = pca.fit_transform(acts_classwise[c])
        acts_classwise_pois[c] = pca.fit_transform(acts_classwise_pois[c])
    colors = [
        "#1f77b4",
        "#1f77b4",
        "#ff7f0e",
        "#ff7f0e",
        "#2ca02c",
        "#2ca02c",
        "#d62728",
        "#d62728",
        "#9467bd",
        "#9467bd",
        "#8c564b",
        "#8c564b",
        "#e377c2",
        "#e377c2",
        "#7f7f7f",
        "#7f7f7f",
        "#bcbd22",
        "#bcbd22",
        "#17becf",
        "#17becf",
    ]

    tsne = TSNE(n_components=2, random_state=seed, perplexity=p, early_exaggeration=e)
    acts_classwise_tsne = {}
    acts_classwise_pois_tsne = {}
    for c in acts_classwise:
        acts_classwise_tsne[c] = tsne.fit_transform(acts_classwise[c])
        acts_classwise_pois_tsne[c] = tsne.fit_transform(acts_classwise_pois[c])
        # proto_dic[c] = tsne.fit_transform(proto_dic[c])
        # proto_dic_pois[c] = tsne.fit_transform(proto_dic_pois[c])
        # print(acts_classwise_tsne[c].shape)
        plt.figure(figsize=(10, 8))
    for i in range(opt.num_class - 1, -1, -1):
        if i != opt.target_label:
            plt.scatter(
                acts_classwise_tsne[i][:-1, 0],
                acts_classwise_tsne[i][:-1, 1],
                label=str(i),
                s=10,
                c=colors[i],
            )

        else:
            plt.scatter(
                acts_classwise_tsne[i][:, 0],
                acts_classwise_tsne[i][:, 1],
                label=str(i) + " (target)",
                c="grey",
                s=12,
            )

    # for i in range(opt.num_class-1,-1,-1):
    #     if i != opt.target_label:
    #         plt.scatter(acts_classwise_pois_tsne[c][-1:,0], acts_classwise_pois_tsne[c][-1:,1], label=str(i), edgecolors="white", s = 35, c = colors[i], lw=6)
    #     else:
    #         plt.scatter(acts_classwise_pois_tsne[c][-1:,0], acts_classwise_pois_tsne[c][-1:,1], s = 35, label=str(i) + " (target)", edgecolors="white", c="grey", lw=6)
    acts_pois_tsne = np.vstack(
        [acts_classwise_pois_tsne[c] for c in acts_classwise_pois]
    )
    acts_pois_mean_tsne = np.mean(acts_pois_tsne, axis=0)

    plt.scatter(
        acts_pois_tsne[:, 0],
        acts_pois_tsne[:, 1],
        label="poisoned",
        c="black",
        s=1,
        marker="^",
    )
    # plt.scatter(acts_pois_mean_tsne[0], acts_pois_mean_tsne[1], label="poisoned", c="black", s= 35, edgecolors="white", lw=4, marker = "^")

    # plt.legend()
    plt.title(f"{opt.attack_method}")
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f"tsne_figs/tSNE_{opt.attack_method}_{p}_{e}_ep{epoch}.png")
    # plt.show()


def get_norm(args):
    global normalize
    normalize = Normalizer(args.dataset)


def delta_sched(opt, epoch):
    if (epoch % (opt.update_gap)) < 1:
        opt.delta = 1
    elif (epoch % (opt.update_gap)) < 2:
        opt.delta = 0.75
    else:
        opt.delta = 0.5


def smooth_delta_sched(opt, epoch):
    if epoch <= 20:
        opt.deta = 1 - epoch / 50
    else:
        opt.delta = 0.5


def attack_with_trigger(args, model, train_loader, target_label, pattern):

    global normalize
    denormalize = Denormalizer(args.dataset)
    correct = 0
    total = 0
    pattern = pattern.to(device)
    model.eval()
    asr = 0
    with torch.no_grad():
        for images, _ in tqdm.tqdm(train_loader):

            images = images.to(device)
            trojan_images = torch.clamp(images + pattern, 0, 1)
            trojan_images = normalize(trojan_images)
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(
                device
            )

            _, y_pred = y_pred.max(1)
            correct += y_pred.eq(y_target).sum().item()
            total += images.size(0)
        asr = correct / total
        print(correct / total)
    return asr


def inversion(args, model, target_label, train_loader, gamma=10, seed=1234):

    global normalize

    if args.dataset == "imagenet":
        shape = (3, 224, 224)
    elif args.dataset == "tinyImagenet":
        shape = (3, 64, 64)
    elif args.dataset == "ROF":
        shape = (3, 64, 64)
    else:
        shape = (3, 32, 32)
    print("Processing label: {}".format(target_label))
    backdoor = PixelBackdoor(
        model,
        shape=shape,
        batch_size=args.batch_size,
        normalize=normalize,
        steps=100,
        augment=False,
    )

    pattern = backdoor.generate(
        train_loader, target_label, attack_size=args.attack_size, gamma=gamma, seed=seed
    )

    attack_with_trigger(args, model, train_loader, target_label, pattern)

    return pattern


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):

    test_process = []

    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets["model"]

    criterionCls = criterions["criterionCls"]

    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        target = target * 0 + opt.target_label

        with torch.no_grad():
            output_s = snet(img)
            # print(torch.argmax(output_s), target)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    if opt.use_wandb:
        wandb.log({"val_acc": acc_clean[0], "test_acc": acc_bd[0]})

    print("[clean]Prec@1: {:.2f}".format(acc_clean[0]))
    print("[bad]Prec@1: {:.2f}".format(acc_bd[0]))

    return acc_clean[0], acc_bd[0]


def train_step_cd(
    opt,
    train_data,
    train_pois_data,
    train_loader,
    nets,
    optimizer,
    criterions,
    pairs_lis,
    protos,
    protos_pois,
    wtcavs,
    cav_dic,
    cav_dic_class,
):
    model = nets["model"]
    if opt.eval_mode:
        model.eval()
    else:
        model.train()
    mse_loss = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    affect = False
    for cur_iter, data in enumerate(train_loader):
        # if opt.update_gap_iter!= -1 and cur_iter%opt.update_gap_iter == 0 and opt.update_cav:
        #     if opt.weight_proto:
        #         protos = save_classwise_protos_delta(model, train_data, opt, protos)
        #         if opt.cav_type == "synth":
        #             protos_pois = save_classwise_protos_delta(model, train_pois_data, opt, protos_pois)
        #     else:
        #         delta = opt.delta
        #         opt.delta = 1
        #         protos = save_classwise_protos_delta(model, train_data, opt, protos)
        #         if opt.cav_type == "synth":
        #             protos_pois = save_classwise_protos_delta(model, train_pois_data, opt, protos_pois)
        #         opt.delta = delta
        #     if opt.weight_cav:
        #         cav_dic_class, cav_dic = get_cav_delta(opt, protos, pairs_lis, cav_dic_class, cav_dic, protos_pois)
        #     else:
        #         delta = opt.delta
        #         opt.delta = 1
        #         cav_dic_class, cav_dic = get_cav_delta(opt, protos, pairs_lis, cav_dic_class, cav_dic, protos_pois)
        #         opt.delta = delta
        imgs, gt = data
        imgs, gt = imgs.to(opt.device), gt.to(opt.device)
        out = model(imgs)
        criterion = criterions["gt_loss"]
        gt_loss = criterion(out, gt)
        if opt.use_gt_loss:
            gt_loss.backward(retain_graph=True)
        cav_loss_lis = []
        # cav_dic = {}
        loss_dic = {}
        for bneck in opt.bottlenecks:
            direc_mean = cav_dic[bneck]

            named_layers = dict(model.named_modules())
            bn_activation = None
            grad = None
            tcav_score = {}
            results = []
            ep_loss = 0
            dot_lis = []
            cos_lis = []
            mse_dvec_lis = []
            proto_dic = protos[bneck]

            for b in range(imgs.shape[0]):
                e = imgs[b]

                def save_activation_hook(mod, inp, out):
                    global bn_act
                    bn_act = out

                handle = named_layers[bneck].register_forward_hook(save_activation_hook)

                out = model(e.unsqueeze(0).to(device))

                act = bn_act

                grad_ = None
                # mapping stuff
                # teacher mapped to student third last layer
                if opt.use_proto:
                    # if use_knn_proto:
                    for f in range(opt.knn_k):
                        if f == 0:
                            loss = mse_loss(
                                act,
                                torch.from_numpy(
                                    proto_dic[gt[b].item()]["cluster_" + str(f)]
                                )
                                .cuda()
                                .unsqueeze(0),
                            )
                        else:
                            loss += mse_loss(
                                act,
                                torch.from_numpy(
                                    proto_dic[gt[b].item()]["cluster_" + str(f)]
                                )
                                .cuda()
                                .unsqueeze(0),
                            )
                    # loss = autograd.Variable(loss, requires_grad = True)
                    grad_ = torch.autograd.grad(
                        loss, act, retain_graph=True, create_graph=True
                    )
                else:
                    grad_ = torch.autograd.grad(
                        out[0][gt[b]], act, retain_graph=True, create_graph=True
                    )
                dot = None
                vec = None
                if opt.agg_cav:
                    dot = torch.dot(
                        grad_[0].to(device).float().squeeze(0).flatten(),
                        direc_mean.to(device).float().flatten(),
                    ) / torch.linalg.norm(direc_mean)
                else:
                    if gt[b].item() != opt.target_label:
                        vec = cav_dic_class[bneck][gt[b].item()].clone()
                    else:
                        vec = grad_[0]
                    dot = torch.dot(
                        grad_[0].to(device).float().squeeze(0).flatten(),
                        vec.to(device).float().flatten(),
                    ) / torch.linalg.norm(vec)
                dot_lis.append(dot)

                unit_grad = grad_[0].to(device).float().squeeze(0).flatten()
                unit_grad = unit_grad / torch.linalg.norm(unit_grad)

                if opt.agg_cav:
                    unit_direc = direc_mean.to(
                        device
                    ).float().flatten() / torch.linalg.norm(direc_mean)
                else:
                    unit_direc = vec.to(device).float().flatten() / torch.linalg.norm(
                        vec
                    )

                mse_dvec_lis.append(mse_loss(unit_grad, unit_direc))

                if "cos" in opt.loss_type:  ### Concept loss
                    if opt.agg_cav:
                        cos_ = cos(
                            grad_[0].to(device).float().squeeze(0).flatten(),
                            direc_mean.to(device).flatten().float(),
                        )
                    else:
                        cos_ = cos(
                            grad_[0].to(device).float().squeeze(0).flatten(),
                            vec.to(device).flatten().float(),
                        )
                    cos_lis.append(cos_)

                handle.remove()

            dot_lis = torch.stack(dot_lis)
            mse_dvec_lis = torch.stack(mse_dvec_lis)
            if "cos" in opt.loss_type:
                cos_lis = torch.stack(cos_lis)

            # or cs(grad and direc_mean) max?
            score = len(torch.where(dot_lis < 0)[0]) / len(dot_lis)
            if opt.use_wandb:
                wandb.log({"tcav score": score})

            if opt.loss_type == "cos":
                if affect == False:
                    loss_ = -torch.sum(cos_lis)

            if affect == True:
                # if loss_type =='cos_mse':
                loss_ = mse_loss(cos_lis, torch.ones(len(cos_lis)).to(device))
                # loss_ = torch.sum(torch.abs(cos_lis)) #L1  direct hence cos 0 = 1 | affect | vectors aligned

            if opt.loss_type == "L1_cos":
                if affect == False:
                    loss_ = torch.sum(torch.abs(cos_lis))  # L1

            if opt.loss_type == "L2_cos":
                if affect == False:
                    loss_ = torch.sum(torch.square(cos_lis))  # L2

            if opt.loss_type == "mse_vecs":
                if affect == False:
                    loss_ = torch.sum(mse_dvec_lis)

            if opt.loss_type == "mse":
                dot_lis_normalised = (dot_lis) / torch.max(dot_lis)
                if affect != None:
                    if affect == False:
                        # print("hereeee",max(dot_lis), mean(dot_lis), min(dot_lis))
                        dot_lis = torch.reshape(dot_lis, (-1,))
                        target = torch.ones(len(dot_lis)).to(device)
                        tcav_loss = None
                        tcav_loss = criterion(
                            dot_lis.unsqueeze(0).to(device),
                            target.unsqueeze(0).to(device),
                        )
                        loss_ = -tcav_loss

            if affect == False:
                loss_ = wtcavs[bneck] * loss_
            loss_dic[bneck] = loss_

        loss_final = 0
        for bneck in opt.bottlenecks:
            loss_final += loss_dic[bneck]

        cav_loss_lis.append(loss_final.item())

        if opt.use_wandb:
            wandb.log(
                {
                    "train loss": loss_final,
                    "gt_loss": gt_loss,
                    "cav_loss": sum(cav_loss_lis),
                }
            )

        if opt.use_cav_loss:  # and ((cur_iter//opt.loss_interval)%2 == 0):
            if bneck == opt.bottlenecks[-1]:
                loss_final.backward(retain_graph=True)
                # loss_.backward()
            else:
                loss_.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()


def get_proto(c, proto_dic):
    cluster_keys = list(proto_dic[c].keys())
    class_proto = torch.zeros_like(torch.tensor(proto_dic[c][cluster_keys[0]]))
    for k in cluster_keys:
        class_proto += proto_dic[c][k]
    class_proto /= len(cluster_keys)
    # class_proto.requires_grad = True
    return class_proto


def get_proto(c, proto_dic):
    cluster_keys = list(proto_dic[c].keys())
    class_proto = torch.zeros_like(torch.tensor(proto_dic[c][cluster_keys[0]]))
    for k in cluster_keys:
        class_proto += proto_dic[c][k]
    class_proto /= len(cluster_keys)
    # class_proto.requires_grad = True
    return class_proto


def get_cav(opt, protos, pairs_lis, protos_pois=None, upconvs=None, downconvs=None):
    cav_dic = dict()
    cav_dic_agg = {}
    if upconvs == None or downconvs == None:
        upconvs = {}
        downconvs = {}
    for bneck in opt.bottlenecks:
        print("Bottleneck: ", bneck)
        class_protos = dict()
        proto_dic = protos[bneck]

        if protos_pois != None:
            class_protos_pois = dict()
            proto_dic_pois = protos_pois[bneck]
            # print("PROTO SHAPE WITH NORMAL ACTS", proto_dic_pois[0]['cluster_0'].shape)
        if len(upconvs) == 0 or len(downconvs) == 0:
            dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
            student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
            upconv_m, downconv_m = map_activation_spaces(
                dino_acts_path, student_acts_path, pairs_lis, 150, opt
            )
            upconvs[bneck] = upconv_m
            downconvs[bneck] = downconv_m
        else:
            upconv_m = upconvs[bneck]
            downconv_m = downconvs[bneck]
        # print("PROTO SHAPE WITH MM ON NORMAL ACTS", upconv_m(torch.tensor(proto_dic[0]['cluster_0']).cuda()).shape)
        # return None, None
        for c in range(opt.num_class):
            # print(opt.distill)
            if (
                opt.distill == True
                and opt.dino_acts == False
                and opt.dino_acts_mm == False
            ):  # Because we don't need upnconv in both cases
                class_protos[c] = upconv_m(get_proto(c, proto_dic).cuda())
                if protos_pois:
                    class_protos_pois[c] = upconv_m(get_proto(c, proto_dic_pois).cuda())
            else:
                class_protos[c] = get_proto(c, proto_dic).cuda()
                if protos_pois:
                    class_protos_pois[c] = get_proto(c, proto_dic_pois).cuda()

        target_proto_dino = class_protos[opt.target_label]
        cav_dino = {}
        cav = {}
        cav_dino_agg = torch.zeros_like(target_proto_dino)
        for c in range(opt.num_class):
            if c != opt.target_label:
                if protos_pois:
                    cav_dino[c] = class_protos[c] - class_protos_pois[c]
                else:
                    cav_dino[c] = class_protos[c] - target_proto_dino
                # print("SHAPE OF COMPUTED CAV: ",cav_dino[c].shape)
                if opt.distill:
                    cav[c] = downconv_m(cav_dino[c])
                else:
                    cav[c] = cav_dino[c]
                cav_dino_agg += cav_dino[c]
        cav_dino_agg /= opt.num_class - 1
        cav_agg = None
        if opt.distill:
            cav_agg = downconv_m(cav_dino_agg.cuda())
        else:
            cav_agg = cav_dino_agg.cuda()
        cav_dic[bneck] = cav
        cav_dic_agg[bneck] = cav_agg
    return cav_dic, cav_dic_agg


def get_cav_delta(
    opt,
    protos,
    pairs_lis,
    prev_cav_dic,
    prev_cav_dic_agg,
    protos_pois=None,
    upconvs=None,
    downconvs=None,
):
    cav_dic = dict()
    cav_dic_agg = {}
    if upconvs == None or downconvs == None:
        upconvs = {}
        downconvs = {}
    for bneck in opt.bottlenecks:
        print("Bottleneck: ", bneck)
        class_protos = dict()
        prev_cav = prev_cav_dic[bneck]
        proto_dic = protos[bneck]
        if protos_pois:
            class_protos_pois = dict()
            proto_dic_pois = protos_pois[bneck]
        if len(upconvs) == 0 or len(downconvs) == 0:
            dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
            student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
            upconv_m, downconv_m = map_activation_spaces(
                dino_acts_path, student_acts_path, pairs_lis, 150, opt
            )
            upconvs[bneck] = upconv_m
            downconvs[bneck] = downconv_m
        else:
            upconv_m = upconvs[bneck]
            downconv_m = downconvs[bneck]
        for c in range(opt.num_class):
            if opt.distill and opt.dino_acts == False and opt.dino_acts_mm == False:
                class_protos[c] = upconv_m(get_proto(c, proto_dic).cuda())
                if protos_pois:
                    class_protos_pois[c] = upconv_m(get_proto(c, proto_dic_pois).cuda())
            else:
                class_protos[c] = get_proto(c, proto_dic).cuda()
                if protos_pois:
                    class_protos_pois[c] = get_proto(c, proto_dic_pois).cuda()
        target_proto_dino = class_protos[opt.target_label]
        cav_dino = {}
        cav = {}
        cav_dino_agg = torch.zeros_like(target_proto_dino)
        for c in range(opt.num_class):
            if c != opt.target_label:
                if protos_pois:
                    cav_dino[c] = class_protos[c] - class_protos_pois[c]
                else:
                    cav_dino[c] = class_protos[c] - target_proto_dino
                if opt.distill:
                    cav[c] = (opt.delta) * downconv_m(cav_dino[c]) + (
                        1 - opt.delta
                    ) * prev_cav[c]
                else:
                    cav[c] = (opt.delta) * cav_dino[c] + (1 - opt.delta) * prev_cav[c]
                cav_dino_agg += cav_dino[c]
        cav_dino_agg /= opt.num_class - 1
        cav_agg = None
        if opt.distill:
            cav_agg = downconv_m(cav_dino_agg.cuda())
        else:
            cav_agg = cav_dino_agg.cuda()
        cav_dic[bneck] = cav
        cav_dic_agg[bneck] = (opt.delta) * cav_agg + (1 - opt.delta) * prev_cav_dic_agg[
            bneck
        ]
    return cav_dic, cav_dic_agg


def cd(opt):
    print(opt.num_class)
    model = select_model(
        dataset=opt.dataset,
        model_name=opt.s_name,
        pretrained=True,
        pretrained_models_path=opt.model,
        n_classes=opt.num_class,
        ml_mmdr=opt.ml_mmdr,
    ).to(opt.device)

    get_norm(args=opt)

    print("----------- DATA Initialization --------------")

    test_clean_loader, test_bad_loader = None, None

    if (
        opt.attack_method == "semantic"
        or opt.attack_method == "semantic_mask"
        or opt.attack_method == "semantic_tattoo"
    ):
        test_transforms = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Resize((64, 64)),
                tfs.Normalize(
                    torch.tensor([0.6668, 0.5134, 0.4482]),
                    torch.tensor([0.0691, 0.0599, 0.0585]),
                ),
                tfs.RandomHorizontalFlip(),
            ]
        )
        semantic = None
        if opt.trigger_type == "semanticMaskTrigger":
            semantic = ROF(
                base_pth="./data/ROF/dataset_final/clean",
                base_pth_pois="./data/ROF/dataset_final/masked",
                pratio=opt.inject_portion,
            )
        elif opt.trigger_type == "semanticTattooTrigger":
            semantic = ROF(
                base_pth="./data/ROF/dataset_final/clean",
                base_pth_pois="./data/ROF/dataset_final/tattoo_single",
                pratio=opt.inject_portion,
            )
        else:
            semantic = ROF(
                base_pth="./data/ROF/data",
                base_pth_pois="./data/ROF/data_sunglasses",
                pratio=opt.inject_portion,
            )
        test_clean_data = semantic.get_test_data(transform=test_transforms)
        test_clean_loader = DataLoader(test_clean_data, batch_size=16)

        test_bad_data = semantic.get_bd_test_data(transform=test_transforms)
        test_bad_loader = DataLoader(test_bad_data, batch_size=16)
    elif opt.attack_method == "wanet":
        test_transforms = tfs.Compose(
            [tfs.ToTensor(), tfs.Resize(size=(32, 32), max_size=None, antialias=None)]
        )
        test_transforms_64 = tfs.Compose(
            [tfs.ToTensor(), tfs.Resize(size=(64, 64), max_size=None, antialias=None)]
        )
        test_bad_data = None
        if opt.dataset == "CIFAR10":
            test_bad_data = ImageFolder(
                "/home2/ava9/BackdoorBench/record/cifar10_vgg19_bn_wanet_0_1/bd_test_dataset/",
                transform=test_transforms,
            )
        elif opt.dataset == "CIFAR100":
            test_bad_data = ImageFolder(
                "../BackdoorBench/record/cifar100_preactresnet18_wanet_0_1/bd_test_dataset/",
                transform=test_transforms,
            )
        elif opt.dataset == "gtsrb":
            test_bad_data = ImageFolder(
                "../BackdoorBench/record/gtsrb_preactresnet18_wanet_0_01/bd_test_dataset/",
                transform=test_transforms,
            )
            print("LEN: ", len(test_bad_data))
        elif opt.dataset == "tinyImagenet":
            test_bad_data = ImageFolder(
                "../BackdoorBench/record/tiny_preactresnet18_wanet_0_05/",
                transform=test_transforms_64,
            )
        test_bad_loader = DataLoader(test_bad_data, batch_size=opt.batch_size)
        test_clean_loader, _ = get_test_loader(opt)
    elif opt.attack_method == "inputaware":
        print("Input Aware test loading start\n\n\n\n")
        test_transforms = tfs.Compose(
            [tfs.ToTensor(), tfs.Resize(size=(32, 32), max_size=None, antialias=None)]
        )
        test_transforms_64 = tfs.Compose(
            [tfs.ToTensor(), tfs.Resize(size=(64, 64), max_size=None, antialias=None)]
        )
        if opt.dataset == "CIFAR10":
            test_bad_data = ImageFolder(
                "/home2/ava9/BackdoorBench/record/cifar10_preactresnet18_inputaware_0_01/bd_test_dataset/",
                transform=test_transforms,
            )
        test_bad_loader = DataLoader(test_bad_data, batch_size=opt.batch_size)
        test_clean_loader, _ = get_test_loader(opt)
    else:
        test_clean_loader, test_bad_loader = get_test_loader(opt)
    train_data = get_train_loader(opt, without_loader=True)
    train_dino_data = get_train_loader(opt, without_loader=True, dino=True)
    train_loader = get_train_loader(opt)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    nets = {"model": model, "victimized_model": copy.deepcopy(model)}

    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    criterions = {"criterionCls": criterionCls, "gt_loss": criterionCls}

    # test(opt, test_clean_loader, test_bad_loader, nets,criterions, 0)
    # exit()

    r = opt.ratio
    opt.ratio = 0.2
    # train_data_tsne = get_train_loader(opt, without_loader = True)
    # train_gt_pois_data_tsne = get_backdoor_loader(opt, shuffle=False, without_loader= True, use_available= True)
    opt.ratio = r

    train_loader = get_train_loader(opt)
    print(len(train_loader))
    g = 10
    s = 1234
    trig_path = f"./trig_sets/synthetic_{opt.s_name}_{opt.attack_method}_{opt.dataset}_g_{g}_seed_{s}_t_{opt.target_label}.png"
    pattern = None
    try:
        trig_img = Image.open(trig_path)
        img_to_tensor = tfs.PILToTensor()
        pattern = img_to_tensor(trig_img)
    except:
        pattern = inversion(opt, model, opt.target_label, train_loader, gamma=g, seed=s)
        tensor_to_img = tfs.ToPILImage()
        trig_img = tensor_to_img(pattern)
        trig_img.save(trig_path)
        print("Done with exception")

    train_pois_data = None
    train_dino_pois_data = None
    train_gt_pois_data = None
    train_dino_gt_pois_data = None
    if opt.cav_type == "synth":
        train_pois_data = get_backdoor_loader(
            opt, shuffle=False, without_loader=True, use_available=True, pattern=pattern
        )
        train_dino_pois_data = get_backdoor_loader(
            opt,
            shuffle=False,
            without_loader=True,
            use_available=True,
            pattern=pattern,
            dino=True,
        )
    if opt.cav_type == "gt":
        train_gt_pois_data = get_backdoor_loader(
            opt, shuffle=False, without_loader=True, use_available=True
        )
        train_dino_gt_pois_data = get_backdoor_loader(
            opt, shuffle=False, without_loader=True, use_available=True, dino=True
        )

    pairs_lis = {}
    bottleneck_name = opt.bottlenecks[0]
    dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"
    student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bottleneck_name[-2:]}"

    # if os.path.exists(dino_acts_path + "_clean0.pt") and os.path.exists(
    #     student_acts_path + "_clean0.pt"
    # ):
    #     pass
    # else:
    if opt.cav_type == "proto":
        try:
            num_csets, pairs_lis = save_acts(model, opt, train_data, train_dino_data)
        except:
            raise
    elif opt.cav_type == "synth":
        num_csets, pairs_lis = save_acts(
            model,
            opt,
            train_data,
            train_dino_data,
            train_pois_data,
            train_dino_pois_data,
        )
    elif opt.cav_type == "gt":
        num_csets, pairs_lis = save_acts(
            model,
            opt,
            train_data,
            train_dino_data,
            train_gt_pois_data,
            train_dino_gt_pois_data,
        )

    protos = save_classwise_protos(
        model, train_data, opt
    )  ################################## *******************************
    protos_pois = None
    if opt.cav_type == "synth" and opt.dino_acts == False and opt.dino_acts_mm == False:
        protos_pois = save_classwise_protos(model, train_pois_data, opt)
    if opt.cav_type == "gt" and opt.dino_acts == False and opt.dino_acts_mm == False:
        protos_gt_pois = save_classwise_protos(model, train_gt_pois_data, opt)
    protos_dino = None
    protos_pois_dino = None
    protos_gt_pois_dino = None
    if opt.dino_acts:
        protos_dino = save_classwise_protos_dino(model, train_dino_data, opt)
        if opt.cav_type == "synth":
            protos_pois_dino = save_classwise_protos_dino(
                model, train_dino_pois_data, opt
            )
        if opt.cav_type == "gt":
            protos_gt_pois_dino = save_classwise_protos_dino(
                model, train_dino_gt_pois_data, opt
            )
    elif opt.dino_acts_mm:
        protos_dino, upconvs, downconvs = save_classwise_protos_dino_mm(
            model, train_data, opt
        )
        if opt.cav_type == "synth":
            protos_pois_dino, _, _ = save_classwise_protos_dino_mm(
                model, train_pois_data, opt, upconvs, downconvs
            )
        if opt.cav_type == "gt":
            protos_gt_pois_dino, _, _ = save_classwise_protos_dino_mm(
                model, train_gt_pois_data, opt, upconvs, downconvs
            )
    # print(len(protos.keys()), protos.keys())
    # pairs_lis = dict()a
    # pairs_lis["clean0"] = [("","")]

    cav_dic = None
    cav_dic_agg = None
    if opt.dino_acts:
        cav_dic, cav_dic_agg = get_cav(opt, protos_dino, pairs_lis, protos_pois_dino)
    elif opt.dino_acts_mm:
        cav_dic, cav_dic_agg = get_cav(
            opt, protos_dino, pairs_lis, protos_pois_dino, upconvs, downconvs
        )
    else:
        cav_dic, cav_dic_agg = get_cav(opt, protos, pairs_lis, protos_pois)

    wtcav = opt.wtcav
    wtcavs = {}
    for i in range(len(opt.bottlenecks)):
        wtcavs[opt.bottlenecks[i]] = wtcav / (len(opt.bottlenecks) - i)
    if opt.use_wandb:
        wandb.init(
            project="Concept Based Detriggering",
            config={
                "s_name": opt.s_name,
                "attack_method": opt.attack_method,
                "dataset": opt.dataset,
                "bottleneck_name": opt.bottlenecks,
                "knn_k": opt.knn_k,
                "wtcav": wtcav,
                "cset_type": opt.cset_type,
                "epochs": opt.epochs,
                "loss_interval": opt.loss_interval,
                "eval_mode": opt.eval_mode,
                "loss_type": opt.loss_type,
                "agg_cav": opt.agg_cav,
                "update_cav": opt.update_cav,
                "weight_cav": opt.weight_cav,
                "weight_proto": opt.weight_proto,
                "update_gap": opt.update_gap,
                "update_gap_iter": opt.update_gap_iter,
                "cav_type": opt.cav_type,
                "sched_delta": opt.sched_delta,
                "dino_acts": opt.dino_acts,
                "dino_acts_mm": opt.dino_acts_mm,
                "ml_mmdr": opt.ml_mmdr,
            },
        )
        wandb.run.name = f"Proto_and_Trig_{opt.s_name}_{opt.attack_method}_{opt.dataset}_wtcav_{wtcav}_k_{opt.knn_k}_e_{opt.epochs}"
    print(opt)
    acts_classwise_pois = None
    acts_classwise = None
    best_dem = 0
    orig_ca = 0
    orig_asr = 0
    # if opt.no_target:
    #     opt.update_gap = int(opt.epochs / opt.num_class)

    for epoch in range(opt.epochs):
        if opt.use_wandb:
            wandb.log({"epoch": epoch})
        criterions = {"criterionCls": criterionCls, "gt_loss": criterionCls}
        if epoch == 0:
            # before training test firstly
            orig_ca, orig_asr = test(
                opt, test_clean_loader, test_bad_loader, nets, criterions, epoch
            )
            exit()
        print(orig_ca, orig_asr)
        # return
        print("===Epoch: {}/{}===".format(epoch + 1, opt.epochs))
        if opt.sched_delta:
            delta_sched(opt, epoch)
            # smooth_delta_sched(opt, epoch)

        # break
        # fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset, mode="CD")
        # train_dataset = torch.utils.data.ConcatDataset([train_data, train_pois_data])
        # if opt.no_target:
        #     opt.target_label = max(
        #         epoch % int(opt.epochs / opt.num_class), opt.num_class - 1
        #     )

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        if opt.update_cav and epoch % opt.update_gap == 0 and opt.update_gap_iter == -1:
            if opt.weight_proto:
                if opt.dino_acts:
                    # model, data, opt , bss = 35, use_knn_proto = True
                    protos_dino = save_classwise_protos_dino_delta(
                        model, train_dino_data, opt, protos_dino
                    )
                elif opt.dino_acts_mm:
                    protos_dino, upconvs, downconvs = (
                        save_classwise_protos_dino_mm_delta(
                            model, train_data, opt, protos_dino
                        )
                    )
                else:
                    protos = save_classwise_protos_delta(model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    if opt.dino_acts:
                        protos_pois_dino = save_classwise_protos_dino_delta(
                            model, train_dino_pois_data, opt, protos_pois_dino
                        )
                    elif opt.dino_acts_mm:
                        protos_pois_dino, _, _ = save_classwise_protos_dino_mm_delta(
                            model,
                            train_pois_data,
                            opt,
                            protos_pois_dino,
                            upconvs,
                            downconvs,
                        )
                    else:
                        protos_pois = save_classwise_protos_delta(
                            model, train_pois_data, opt, protos_pois
                        )
                elif opt.cav_type == "gt":
                    if opt.dino_acts:
                        protos_pois_dino = save_classwise_protos_dino_delta(
                            model, train_dino_gt_pois_data, opt, protos_gt_pois_dino
                        )
                    elif opt.dino_acts_mm:
                        protos_pois_dino, _, _ = save_classwise_protos_dino_mm_delta(
                            model,
                            train_gt_pois_data,
                            opt,
                            protos_pois_dino,
                            upconvs,
                            downconvs,
                        )
                    else:
                        protos_pois = save_classwise_protos_delta(
                            model, train_gt_pois_data, opt, protos_gt_pois
                        )
            else:
                delta = opt.delta
                opt.delta = 1
                if opt.dino_acts:
                    # model, data, opt , bss = 35, use_knn_proto = True
                    protos_dino = save_classwise_protos_dino_delta(
                        model, train_dino_data, opt, protos_dino
                    )
                elif opt.dino_acts_mm:
                    protos_dino, upconvs, downconvs = (
                        save_classwise_protos_dino_mm_delta(
                            model, train_data, opt, protos_dino
                        )
                    )
                else:
                    protos = save_classwise_protos_delta(model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    if opt.dino_acts:
                        protos_pois_dino = save_classwise_protos_dino_delta(
                            model, train_dino_pois_data, opt, protos_pois_dino
                        )
                    elif opt.dino_acts_mm:
                        protos_pois_dino, _, _ = save_classwise_protos_dino_mm_delta(
                            model,
                            train_pois_data,
                            opt,
                            protos_pois_dino,
                            upconvs,
                            downconvs,
                        )
                    else:
                        protos_pois = save_classwise_protos_delta(
                            model, train_pois_data, opt, protos_pois
                        )
                elif opt.cav_type == "gt":
                    if opt.dino_acts:
                        protos_pois_dino = save_classwise_protos_dino_delta(
                            model, train_dino_gt_pois_data, opt, protos_gt_pois_dino
                        )
                    elif opt.dino_acts_mm:
                        protos_pois_dino, _, _ = save_classwise_protos_dino_mm_delta(
                            model,
                            train_gt_pois_data,
                            opt,
                            protos_pois_dino,
                            upconvs,
                            downconvs,
                        )
                    else:
                        protos_pois = save_classwise_protos_delta(
                            model, train_gt_pois_data, opt, protos_gt_pois
                        )
                opt.delta = delta
            if opt.weight_cav:
                if opt.dino_acts:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt,
                        protos_dino,
                        pairs_lis,
                        cav_dic,
                        cav_dic_agg,
                        protos_pois_dino,
                    )
                elif opt.dino_acts_mm:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt,
                        protos_dino,
                        pairs_lis,
                        cav_dic,
                        cav_dic_agg,
                        protos_pois_dino,
                        upconvs,
                        downconvs,
                    )
                else:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt, protos, pairs_lis, cav_dic, cav_dic_agg, protos_pois
                    )
            else:
                delta = opt.delta
                opt.delta = 1
                if opt.dino_acts:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt,
                        protos_dino,
                        pairs_lis,
                        cav_dic,
                        cav_dic_agg,
                        protos_pois_dino,
                    )
                elif opt.dino_acts_mm:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt,
                        protos_dino,
                        pairs_lis,
                        cav_dic,
                        cav_dic_agg,
                        protos_pois_dino,
                        upconvs,
                        downconvs,
                    )
                else:
                    cav_dic, cav_dic_agg = get_cav_delta(
                        opt, protos, pairs_lis, cav_dic, cav_dic_agg, protos_pois
                    )
                opt.delta = delta
            # linearly weigh both protos and cav & cav alone.. start with delta = 0.5
            # acts_classwise_pois, proto_dic_pois = get_act_classwise(model, train_gt_pois_data_tsne, opt)
            # acts_classwise, proto_dic = get_act_classwise(model, train_data_tsne, opt)
        train_step_cd(
            opt,
            train_data,
            train_pois_data,
            train_loader,
            nets,
            optimizer,
            criterions,
            pairs_lis,
            protos,
            protos_pois,
            wtcavs,
            cav_dic_agg,
            cav_dic,
        )
        print("testing the models......")
        ca, asr = test(
            opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1
        )
        print(orig_asr, orig_ca)
        dem = 0.5 * (
            1
            - np.max((orig_ca - ca) / orig_ca, 0)
            + np.max((orig_asr - asr) / orig_asr, 0)
        )
        if dem > best_dem:
            torch.save(
                model.state_dict(),
                f"./clean_ckpts/{opt.s_name}_{opt.attack_method}.pth.tar",
            )
        # tsne_vis(opt, opt.p, opt.e, acts_classwise, acts_classwise_pois, epoch)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # opt = get_arguments().parse_args()
    # opt = get_arguments_1().parse_args()
    opt = get_arguments_2_preact().parse_args()
    # print(opt.s_name, opt.model, opt.attack_method, ", wtcav: ", opt.wtcav, ", delta:   ", opt.delta)
    # print(opt.epochs, opt.delta)
    # exit()
    random.seed(opt.seed)  # torch transforms use this seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    # reverse_engineer(opt)
    # opt.use_wandb = False
    # opt.eval_mode = True
    # opt.ratio = 0.2

    # if "_" in opt.model:
    #     toks = opt.model.split("_")
    #     opt.target_label = int(toks[-2])
    #     opt.inject_portion = int(toks[-1].split(".")[0])/(10**len(toks[-1].split(".")[0]))
    # else:
    opt.model = f"./weight/{opt.dataset}/{opt.s_name}-{opt.attack_method}.pth.tar"

    if opt.dataset == "CIFAR10":
        opt.num_class = 10
    if opt.dataset == "CIFAR100":
        opt.num_class = 100
    if opt.dataset == "gtsrb":
        opt.num_class = 43
    if opt.dataset == "ROF":
        opt.num_class = 10
        opt.ratio = 0.5
    if opt.dataset == "tinyImagenet":
        opt.num_class = 200
        opt.trig_w = 5
        opt.trig_h = 5
    if opt.s_name == "preactresnet18":
        opt.bottlenecks = ["layer4.1.conv2"]
    if opt.s_name == "preactresnet34":
        opt.bottlenecks = ["layer4.2.conv2"]
    if opt.s_name == "preactresnet50":
        opt.bottlenecks = ["layer4.2.conv3"]
    if opt.s_name == "resnet50":
        opt.bottlenecks = ["layer4.2.conv3"]
    if opt.s_name == "vgg19_bn":
        opt.bottlenecks = ["features.49"]

    if opt.attack_method == "badnet":
        opt.trigger_type = "squareTrigger"
        if opt.dataset == "tinyImagenet":
            opt.trigger_type = "customSquareTrigger"
        opt.p = 958  # p and e are tSNE params
        opt.e = 35
    if opt.attack_method == "blended":
        opt.trigger_type = "blendTrigger"
        opt.p = 955
        opt.e = 35
    if opt.attack_method == "trojannn":
        opt.trigger_type = "trojanTrigger"
        opt.p = 900
        opt.e = 30
    if opt.attack_method == "sig":
        opt.trigger_type = "signalTrigger"
        opt.p = 955
        opt.e = 35
    if opt.attack_method == "wanet":
        opt.trigger_type = "wanetTrigger"
        opt.p = 850
        opt.e = 25
    if opt.attack_method == "inputaware":
        opt.trigger_type = "inputawareTrigger"
    if opt.attack_method == "semantic":
        opt.trigger_type = "semanticTrigger"
    if opt.attack_method == "semantic_mask":
        opt.model = f"./weight/{opt.dataset}/{opt.s_name}-{opt.attack_method}1.pth.tar"
        opt.trigger_type = "semanticMaskTrigger"
    if opt.attack_method == "semantic_tattoo":
        opt.model = f"./weight/{opt.dataset}/{opt.s_name}-{opt.attack_method}1.pth.tar"
        opt.trigger_type = "semanticTattooTrigger"

    opt.data_name = opt.dataset
    # opt.no_target = True
    print(opt)
    cd(opt)
    # print(opt)
