# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, ImageEncoderViT0
import torch.nn.functional as F

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registryv0 = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 512
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT0(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    # sam.eval()
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        #     state_dict = torch.load(f)
        # sam.load_state_dict(state_dict)
        with open(checkpoint, "rb") as f:
            ckpt=torch.load(f,map_location='cpu')
        # sam.load_state_dict(state_dict)
        from collections import OrderedDict
        dict=OrderedDict()
        for k,v in ckpt.items():
            if 'pos_embed' in k :
                dict[k] = reshapePos(v, image_size)
                continue
            if 'rel_pos' in k:
                dict[k] = reshapeRel(k, v, image_size)
                continue
            if "image_encoder" in k and "neck" in k:
                #Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k]=v

        sam.load_state_dict(dict, strict=False)
    return sam

def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('5' in k or '11' in  k or '17' in k or '23' in k) or '15' in k:
        return rel_pos_params   
    
    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]