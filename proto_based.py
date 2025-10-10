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

        target = (target * 0) + opt.target_label

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

    return acc_clean, acc_bd


def test_combat(
    netC,
    netG,
    test_dl,
    epoch,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_clean_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    gauss_smooth = tfs.GaussianBlur(
        kernel_size=opt.kernel_size, sigma=opt.sigma)

    torch.nn.BCELoss()
    for batch_idx, (inputs, targets, _) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)

            # Evaluate Clean
            preds_clean = netC(inputs)

            total_clean_sample += len(inputs)
            total_clean_correct += torch.sum(
                torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            ntrg_ind = (targets != opt.target_label).nonzero()[:, 0]
            inputs_toChange = inputs[ntrg_ind]
            targets_toChange = targets[ntrg_ind]
            noise_bd = netG(inputs_toChange)
            noise_bd = low_freq(noise_bd, opt)
            inputs_bd = torch.clamp(
                inputs_toChange + noise_bd * opt.noise_rate, -1, 1)
            inputs_bd = gauss_smooth(inputs_bd)
            targets_bd = create_targets_bd(targets_toChange, opt)
            preds_bd = netC(inputs_bd)

            total_bd_sample += len(ntrg_ind)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1)
                                          == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_clean_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample

            info_string = "Clean Acc: {:.4f}  | Bd Acc: {:.4f} ".format(
                acc_clean, acc_bd
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    # wandb
    if opt.use_wandb:
        wandb.log({"val_acc": acc_clean, "test_acc": acc_bd})

    # # Save checkpoint
    # if acc_clean > best_clean_acc:
    #     print(" Saving...")
    #     best_clean_acc = acc_clean
    #     best_bd_acc = acc_bd
    #     state_dict = {
    #         "netC": netC.state_dict(),
    #         "schedulerC": schedulerC.state_dict(),
    #         "optimizerC": optimizerC.state_dict(),
    #         "netG": netG.state_dict(),
    #         "best_clean_acc": acc_clean,
    #         "best_bd_acc": acc_bd,
    #         "epoch_current": epoch,
    #     }
    #     torch.save(state_dict, opt.ckpt_path)
    return acc_clean, acc_bd


def train_step_pgbd(
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
        if (
            opt.update_gap_iter != -1
            and cur_iter % opt.update_gap_iter == 0
            and opt.update_cav
        ):
            if opt.weight_proto:
                protos = save_classwise_protos_delta(
                    model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    protos_pois = save_classwise_protos_delta(
                        model, train_pois_data, opt, protos_pois
                    )
            else:
                delta = opt.delta
                opt.delta = 1
                protos = save_classwise_protos_delta(
                    model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    protos_pois = save_classwise_protos_delta(
                        model, train_pois_data, opt, protos_pois
                    )
                opt.delta = delta
            if opt.weight_pav:
                cav_dic_class, cav_dic = get_cav_delta(
                    opt, protos, pairs_lis, cav_dic_class, cav_dic, protos_pois
                )
            else:
                delta = opt.delta
                opt.delta = 1
                cav_dic_class, cav_dic = get_cav_delta(
                    opt, protos, pairs_lis, cav_dic_class, cav_dic, protos_pois
                )
                opt.delta = delta
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

                handle = named_layers[bneck].register_forward_hook(
                    save_activation_hook)

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
                                    proto_dic[gt[b].item()
                                              ]["cluster_" + str(f)]
                                )
                                .cuda()
                                .unsqueeze(0),
                            )
                        else:
                            loss += mse_loss(
                                act,
                                torch.from_numpy(
                                    proto_dic[gt[b].item()
                                              ]["cluster_" + str(f)]
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

                if "cos" in opt.loss_type:  # Concept loss
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


def get_cav(opt, protos, pairs_lis, protos_pois=None):
    cav_dic = dict()
    cav_dic_agg = {}
    upconvs = {}
    downconvs = {}
    for bneck in opt.bottlenecks:
        print("Bottleneck: ", bneck)
        class_protos = dict()
        proto_dic = protos[bneck]
        if protos_pois != None:
            class_protos_pois = dict()
            proto_dic_pois = protos_pois[bneck]
        dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        upconv_m, downconv_m = map_activation_spaces(
            dino_acts_path, student_acts_path, pairs_lis, 150, opt
        )
        upconvs[bneck] = upconv_m
        downconvs[bneck] = downconv_m
        for c in range(opt.num_class):
            # print(opt.distill)
            if (
                opt.distill == True and opt.dino_acts == False
            ):  # Because we don't need MM in both cases
                class_protos[c] = upconv_m(get_proto(c, proto_dic).cuda())
                if protos_pois:
                    class_protos_pois[c] = upconv_m(
                        get_proto(c, proto_dic_pois).cuda())
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
    opt, protos, pairs_lis, prev_cav_dic, prev_cav_dic_agg, protos_pois=None
):
    cav_dic = dict()
    cav_dic_agg = {}
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
        dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        upconv_m, downconv_m = map_activation_spaces(
            dino_acts_path, student_acts_path, pairs_lis, 150, opt
        )
        upconvs[bneck] = upconv_m
        downconvs[bneck] = downconv_m
        for c in range(opt.num_class):
            if opt.distill and opt.dino_acts == False:
                class_protos[c] = upconv_m(get_proto(c, proto_dic).cuda())
                if protos_pois:
                    class_protos_pois[c] = upconv_m(
                        get_proto(c, proto_dic_pois).cuda())
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
                    cav[c] = (opt.delta) * cav_dino[c] + \
                        (1 - opt.delta) * prev_cav[c]
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


def pgbd(opt):
    print(opt.num_class)
    model = select_model(
        dataset=opt.dataset,
        model_name=opt.s_name,
        pretrained=True,
        pretrained_models_path=opt.model,
        n_classes=opt.num_class,
    ).to(opt.device)

    get_norm(args=opt)

    print("----------- DATA Initialization --------------")
    train_data = get_train_loader(opt, without_loader=True)
    train_loader = get_train_loader(opt)

    train_loader = get_train_loader(opt)

    # Initialization for synthetic trigger in the case of ST-PGBD
    pattern = None
    train_pois_data = None
    if opt.cav_type == "synth":
        g = 10
        s = 1234
        trig_path = f"./trig_sets/synthetic_{opt.s_name}_{opt.attack_method}_{opt.dataset}_g_{g}_seed_{s}_t_{opt.target_label}.png"
        try:
            trig_img = Image.open(trig_path)
            img_to_tensor = tfs.PILToTensor()
            pattern = img_to_tensor(trig_img)
        except:
            pattern = inversion(opt, model, opt.target_label,
                                train_loader, gamma=g, seed=s)
            tensor_to_img = tfs.ToPILImage()
            trig_img = tensor_to_img(pattern)
            trig_img.save(trig_path)
            print("Done with exception")

        train_pois_data = get_backdoor_loader(
            opt, shuffle=False, without_loader=True, use_available=True, pattern=pattern
        )

    protos = save_classwise_protos(
        model, train_data, opt
    )  # *******************************
    protos_pois = None
    # synth stands for V(S) usage, i.e. ST-PGBD variant in the paper
    if opt.cav_type == "synth":
        protos_pois = save_classwise_protos(model, train_pois_data, opt)

    pairs_lis = dict()
    pairs_lis["clean0"] = [("", "")]

    # Check if activation files already exist, if not save them before starting sanitization
    is_acts_saved = {bneck: False for bneck in opt.bottlenecks}
    for bneck in opt.bottlenecks:
        dino_acts_path = f"dino_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        student_acts_path = f"student_activations_{opt.s_name}_{opt.dataset}_{opt.attack_method}_{bneck[-2:]}"
        if os.path.exists(dino_acts_path) and os.path.exists(student_acts_path):
            print(
                f"Activation files for {bneck} already exist! Loading them directly...")
            is_acts_saved[bneck] = True
        else:
            train_dino_data = get_train_loader(
                opt, without_loader=True, dino=True)
            if opt.cav_type == "proto":
                num_csets, pairs_lis = save_acts(
                    model, opt, train_data, train_dino_data)
            elif opt.cav_type == "synth":
                train_dino_pois_data = get_backdoor_loader(
                    opt,
                    shuffle=False,
                    without_loader=True,
                    use_available=True,
                    pattern=pattern,
                    dino=True,
                )
                save_acts(model, opt, train_data, train_dino_data,
                          train_pois_data, train_dino_pois_data)
            break

    cav_dic, cav_dic_agg = get_cav(opt, protos, pairs_lis, protos_pois)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    nets = {"model": model, "victimized_model": copy.deepcopy(model)}
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    # TODO: Can this part be pushed into data_loader.py?
    if (
        opt.attack_method == "semantic"
        or opt.attack_method == "semantic_mask"
        or opt.attack_method == "semantic_tattoo"
    ):
        test_transforms = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Resize((64, 64)),
                tfs.Normalize(  # print(len(protos.keys()), protos.keys())
                    torch.tensor([0.6668, 0.5134, 0.4482]),
                    torch.tensor([0.0691, 0.0599, 0.0585]),
                ),
                tfs.RandomHorizontalFlip(),
            ]
        )
        semantic = ROF(pratio=opt.inject_portion)
        if opt.trigger_type == "semanticMaskTrigger":
            test_transforms = tfs.Compose(
                [
                    tfs.ToTensor(),
                    tfs.Resize((64, 64)),
                    tfs.Normalize(
                        torch.tensor([0.6137, 0.4663, 0.4065]),
                        torch.tensor([0.0724, 0.0628, 0.0611]),
                    ),
                    tfs.RandomHorizontalFlip(),
                ]
            )
            semantic = ROF(
                pratio=opt.inject_portion,
                base_pth="./data/ROF/dataset_final/clean",
                base_pth_pois="./data/ROF/dataset_final/masked",
            )
        elif opt.trigger_type == "semanticTattooTrigger":
            test_transforms = tfs.Compose(
                [
                    tfs.ToTensor(),
                    tfs.Resize((64, 64)),
                    tfs.Normalize(
                        torch.tensor([0.6137, 0.4663, 0.4065]),
                        torch.tensor([0.0724, 0.0628, 0.0611]),
                    ),
                    tfs.RandomHorizontalFlip(),
                ]
            )
            semantic = ROF(
                pratio=opt.inject_portion,
                base_pth="./data/ROF/clean_tattoo",
                base_pth_pois="./data/ROF/tattoo",
            )

        test_clean_data = semantic.get_test_data(transform=test_transforms)
        test_clean_loader = DataLoader(test_clean_data, batch_size=10)

        test_bad_data = semantic.get_bd_test_data(transform=test_transforms)
        test_bad_loader = DataLoader(test_bad_data, batch_size=10)
        print(
            "Shape of ROF test data post transform",
            test_clean_data[0][0].shape,
            test_bad_data[0][0].shape,
        )

    elif opt.attack_method == "wanet":
        test_transforms = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Resize(size=(32, 32), max_size=None, antialias=None),
            ]
        )
        test_bad_data = None
        if opt.dataset == "CIFAR10":
            test_bad_data = ImageFolder(
                "../BackdoorBench/record/cifar10_preactresnet18_wanet_0_1/bd_test_dataset/",
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
        test_bad_loader = DataLoader(test_bad_data, batch_size=opt.batch_size)
    elif opt.attack_method == "inputaware":
        test_transforms = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Resize(size=(32, 32), max_size=None, antialias=None),
                tfs.Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.247, 0.243, 0.261]),
            ]
        )
        test_bad_data = None
        if opt.dataset == "CIFAR10":
            test_bad_data = ImageFolder(
                "../BackdoorBench/record/cifar10_preactresnet18_inputaware_0_01/bd_test_dataset/",
                transform=test_transforms,
            )
        test_bad_loader = DataLoader(test_bad_data, batch_size=opt.batch_size)

    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    wtcav = opt.wtcav
    wtcavs = {}
    for i in range(len(opt.bottlenecks)):
        wtcavs[opt.bottlenecks[i]] = wtcav / (len(opt.bottlenecks) - i)
    if opt.use_wandb:
        wandb.init(
            project="PGBD",
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
                "weight_pav": opt.weight_pav,
                "weight_proto": opt.weight_proto,
                "update_gap": opt.update_gap,
                "update_gap_iter": opt.update_gap_iter,
                "cav_type": opt.cav_type,
                "sched_delta": opt.sched_delta,
            },
        )
        wandb.run.name = f"Proto_and_Trig_{opt.s_name}_{opt.attack_method}_{opt.dataset}_wtcav_{wtcav}_k_{opt.knn_k}_e_{opt.epochs}"
    print(opt)
    for epoch in range(opt.epochs):
        if opt.use_wandb:
            wandb.log({"epoch": epoch})
        criterions = {"criterionCls": criterionCls, "gt_loss": criterionCls}
        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader,
                 nets, criterions, epoch)
        print("===Epoch: {}/{}===".format(epoch + 1, opt.epochs))
        if opt.sched_delta:
            delta_sched(opt, epoch)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        if opt.update_cav and epoch % opt.update_gap == 0 and opt.update_gap_iter == -1:
            # Do weighted update step for prototypes and pavs
            if opt.weight_proto:
                protos = save_classwise_protos_delta(
                    model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    protos_pois = save_classwise_protos_delta(
                        model, train_pois_data, opt, protos_pois
                    )
            else:
                delta = opt.delta
                opt.delta = 1
                protos = save_classwise_protos_delta(
                    model, train_data, opt, protos)
                if opt.cav_type == "synth":
                    protos_pois = save_classwise_protos_delta(
                        model, train_pois_data, opt, protos_pois
                    )
                opt.delta = delta
            if opt.weight_pav:
                cav_dic, cav_dic_agg = get_cav_delta(
                    opt, protos, pairs_lis, cav_dic, cav_dic_agg, protos_pois
                )
            else:
                delta = opt.delta
                opt.delta = 1
                cav_dic, cav_dic_agg = get_cav_delta(
                    opt, protos, pairs_lis, cav_dic, cav_dic_agg, protos_pois
                )
                opt.delta = delta
            # linearly weigh both protos and pav & pav alone.. start with delta = 0.5
        train_step_pgbd(
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
        if opt.attack_method == "combat":
            netG = select_model(
                dataset=opt.dataset, model_name="combat_gen", pretrained=False
            ).to(opt.device)
            state_dict = torch.load(
                "/home2/ava9/COMBAT/checkpoints/train_generator_n008_pc05_clean/cifar10/cifar10_train_generator_n008_pc05_clean.pth.tar"
            )
            netG.load_state_dict(state_dict)
            netG.eval()
        else:
            test(opt, test_clean_loader, test_bad_loader,
                 nets, criterions, epoch + 1)


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

    if opt.s_name == "preactresnet18":
        opt.bottlenecks = ["layer4.1.conv2"]
    if opt.s_name == "resnet50":
        opt.bottlenecks = ["layer4.2.conv3"]
    if opt.s_name == "vgg19_bn":
        opt.bottlenecks = ["features.49"]

    if opt.attack_method == "badnet":
        opt.trigger_type = "squareTrigger"
    if opt.attack_method == "blended":
        opt.trigger_type = "blendTrigger"
    if opt.attack_method == "trojannn":
        opt.trigger_type = "trojanTrigger"
    if opt.attack_method == "sig":
        opt.trigger_type = "signalTrigger"
    if opt.attack_method == "badnet":
        opt.trigger_type = "squareTrigger"
    if opt.attack_method == "wanet":
        opt.trigger_type = "wanetTrigger"
    if opt.attack_method == "inputaware":
        opt.trigger_type = "inputawareTrigger"
    if opt.attack_method == "semantic":
        opt.trigger_type = "semanticTrigger"
    if opt.attack_method == "semantic_mask":
        opt.trigger_type = "semanticMaskTrigger"
        opt.model = (
            f"./weight/{opt.dataset}/{opt.s_name}-{opt.attack_method}_gen.pth.tar"
        )
    if opt.attack_method == "combat":
        opt.trigger_type = "combatTrigger"
        opt.dct_ratio = 0.65
        opt.input_height = 32

    if opt.attack_method == "semantic_tattoo":
        opt.trigger_type = "semanticTattooTrigger"
        opt.model = f"./weight/{opt.dataset}/{opt.s_name}-{opt.attack_method}.pth.tar"
    opt.data_name = opt.dataset
    opt.use_kmedoids = False
    print(opt)
    pgbd(opt)
    # print(opt)
