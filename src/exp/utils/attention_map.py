## Original code: https://github.com/wooyeolBaek/attention-map/blob/main/utils.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from einops import rearrange

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)

def cross_attn_init():
    ########## attn_call is faster ##########
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call

def attn_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ####################################################################################################
        # (20,4096,77) or (40,1024,77)
        if hasattr(self, "store_attn_map"):
            print(attention_probs.shape)
            # self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height) # (10,9216,77) -> (10,77,96,96)
            # self.timestep = int(timestep.item())
        ####################################################################################################
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def lora_attn_call(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()
    ####################################################################################################
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor)
    ####################################################################################################

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)

def hook_fn(name, attn_maps, detach=True):
    def forward_hook(module, input, output):
        pass
        # if hasattr(module.processor, "attn_map"):
        #     timestep = module.processor.timestep
        #     attn_maps[timestep] = attn_maps.get(timestep, dict())
        #     attn_maps[timestep][name] = module.processor.attn_map.detach().cpu() if detach else module.processor.attn_map
        #     del module.processor.attn_map
    return forward_hook

def register_cross_attention_hook(unet):
    """
    store attention map in dictionary
    Args:
        unet (nn.Module): denoising unet
    Returns:
        unet (nn.Module): denoising unet
        attn_maps (Dict): dictionary that stores attention maps
    """
    cross_attn_init()

    attn_maps = {}
    for name, module in unet.named_modules():
        # if not name.split('.')[-1].startswith('attn2'):
        #     continue

        # if name.find('audio') > 0:
        #     continue

        ## cross-attention (spatial, audio)
        if not name.split('.')[-1] == 'attn2':
            continue

        ## temporal cross-attention (motion_module)
        # if not name.split('.')[-1] == 'attention_blocks':
        #     continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name, attn_maps))
    
    return unet, attn_maps

def interpolate_attn_map():
    value = torch.mean(value, axis=0).squeeze(0)
    seq_len, h, w = value.shape
    max_height = max(h, max_height)
    max_width = max(w, max_width)
    value = F.interpolate(
        value.to(dtype=torch.float32).unsqueeze(0),
        size=(max_height, max_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0) # (77,64,64)

def resize_and_save(
        attn_maps, 
        tokenizer, 
        prompt, 
        timestep=None,
        path=None, 
        max_height=256, 
        max_width=256, 
        save_path='attn_maps'
    ):
    resized_map = None

    if path is None:
        for path_ in list(attn_maps[timestep].keys()):
            
            value = attn_maps[timestep][path_]
            value = torch.mean(value,axis=0).squeeze(0)
            seq_len, h, w = value.shape
            max_height = max(h, max_height)
            max_width = max(w, max_width)
            value = F.interpolate(
                value.to(dtype=torch.float32).unsqueeze(0),
                size=(max_height, max_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0) # (77,64,64)
            resized_map = resized_map + value if resized_map is not None else value
    else:
        value = attn_maps[timestep][path]
        value = torch.mean(value,axis=0).squeeze(0)
        seq_len, h, w = value.shape
        max_height = max(h, max_height)
        max_width = max(w, max_width)
        value = F.interpolate(
            value.to(dtype=torch.float32).unsqueeze(0),
            size=(max_height, max_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # (77,64,64)
        resized_map = value

    # init dirs
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + f'/{timestep}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if path is not None:
        save_path = save_path + f'/{path}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    for i, token_attn_map in enumerate(resized_map):

        # min-max normalization(for visualization purpose)
        token_attn_map = token_attn_map.numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)

        # save the image
        image = Image.fromarray(normalized_token_attn_map)
        image.save(os.path.join(save_path, f"{i:04d}"))

    # # match with tokens
    # tokens = prompt2tokens(tokenizer, prompt)
    # bos_token = tokenizer.bos_token
    # eos_token = tokenizer.eos_token
    # pad_token = tokenizer.pad_token

    # # init dirs
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # save_path = save_path + f'/{timestep}'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # if path is not None:
    #     save_path = save_path + f'/{path}'
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    
    # for i, (token, token_attn_map) in enumerate(zip(tokens, resized_map)):
    #     if token == bos_token:
    #         continue
    #     if token == eos_token:
    #         break
    #     token = token.replace('</w>','')
    #     token = f'{i}_<{token}>.jpg'

    #     # min-max normalization(for visualization purpose)
    #     token_attn_map = token_attn_map.numpy()
    #     normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
    #     normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)

    #     # save the image
    #     image = Image.fromarray(normalized_token_attn_map)
    #     image.save(os.path.join(save_path, token))