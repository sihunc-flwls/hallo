import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_stage1_emo import Net
from hallo.models.face_locator import FaceLocator
from hallo.models.mutual_self_attention import ReferenceAttentionControl
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler

class CFG():
    def __init__(self):
        self.base_model_path= "./pretrained_models/stable-diffusion-v1-5/"
        self.vae_model_path= "./pretrained_models/sd-vae-ft-mse"
        self.face_analysis_model_path= "./pretrained_models/face_analysis"
        self.face_locator_pretrained=False
        
if __name__ == "__main__":
    cfg = CFG()
    cfg.face_locator_pretrained

    weight_dtype=torch.float32

    # create model
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
        use_landmark=False
    ).to(device="cuda", dtype=weight_dtype)

    face_locator = FaceLocator(
        conditioning_embedding_channels=320,
        conditioning_channels=1,
        act='relu',
    ).to(device="cuda", dtype=weight_dtype)

    # Freeze
    vae.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)
    face_locator.requires_grad_(False)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    noisy_latents = torch.randn([1, 4, 1, 64, 64]) * 0.18215
    timesteps = torch.randint(0, 1000, (1,)).long()
    ref_image_latents = torch.randn([1 ,4, 64, 64]) * 0.18215
    hidden_emb = torch.zeros(1, 512)
    face_mask = torch.zeros(1, 1, 1, 512, 512)
    face_mask[:, :, 100:300, 100:300] = 1
    uncond_fwd = True

    print('-'*20)
    print(noisy_latents.shape)
    print(timesteps.shape)
    print(hidden_emb.shape)
    print(ref_image_latents.shape)
    print(face_mask.shape)
    print('-'*20)

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        reference_control_writer,
        reference_control_reader
    ).to(dtype=weight_dtype)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()

    import pdb;pdb.set_trace()
    output = net(noisy_latents.cuda(),timesteps.cuda(),ref_image_latents.cuda(),hidden_emb.cuda(),face_mask.cuda(),uncond_fwd)