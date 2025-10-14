import sys
sys.path.append("/home2/ava9/PGBD/N3F/feature_extractor/lib/")

import torch
import numpy as np
from torchvision import transforms as tfs
from .dinov2.models.vision_transformer import DinoVisionTransformer
from PIL import Image



MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model_v2(model_name, model_path, device):
    if model_name == "dino":
        model = DINOv2(patch_size=14, device=device)
        # downloaded from links provided in https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # model.load_checkpoint(model_path)
    elif model_name == "dino16":
        model = DINOv2(patch_size=16, device=device)
        model.load_checkpoint(model_path)
    else:
        raise NotImplementedError
    return model


class DINOv2:
    def __init__(self, patch_size, device="cuda:0"):
        self.device = device
        # self.model = DinoVisionTransformer(patch_size=patch_size, qkv_bias=True).to(
        #     self.device
        # )
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()

    def load_checkpoint(self, ckpt_file, checkpoint_key="model"):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded with msg: {}".format(msg))

    def extract_features(
        self, img: Image or torch.Tensor, transform=True, upsample=True
    ):
        if transform:
            img = self.transform(img, 224).unsqueeze(0)  # Nx3xHxW
            print("performing transform")
        with torch.no_grad():
            out = self.model.get_intermediate_layers(img.to(self.device), n=1)[0]
            # print(img.shape, out.shape)
            # out = out[:, 1:, :]  # we discard the [CLS] token
            # print(self.model.patch_embed.patch_size)    
            h, w = int(img.shape[2] / self.model.patch_embed.patch_size[0]), int(
                img.shape[3] / self.model.patch_embed.patch_size[0]
            )
            
            dim = out.shape[-1]
            out = out.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
            if upsample:
                out = torch.nn.functional.interpolate(out, (270, 480)).squeeze(0)
        return out

    @staticmethod
    def transform(img, image_size):
        transforms = tfs.Compose(
            [tfs.Resize(image_size), tfs.ToTensor(), tfs.Normalize(MEAN, STD)]
        )
        img = transforms(img)
        return img
