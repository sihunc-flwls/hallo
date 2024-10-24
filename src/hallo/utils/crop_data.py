# python align_data.py --input_dir ./src/real_test --save_dir ./out/real_test
# python align_data.py --input_dir ../video_to_frame/out/ys --save_dir ../video_to_frame/out/yss

import scipy
import scipy.ndimage
import numpy as np
import glob, os
import mediapy as mp
from PIL import Image
import copy
from hrnet_lms.landmark_criteria import LandmarkCriterion
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
import math

import kornia.geometry.transform as kgt

import numpy as np
from torchvision.utils import save_image, draw_keypoints

T = transforms.Compose([
            transforms.Resize([256,256]), # assert resolution == 256
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
T512 = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

def to_numpy(x, is_mask=False, is_flow=False):
    if isinstance(x, list):
        x = torch.cat(x, dim=3)
    if is_mask:
        return x.clamp_(0,1).permute(0,2,3,1).detach().cpu().numpy()
    return ((x+1)/2).permute(0,2,3,1).clamp_(0,1).detach().cpu().numpy()

def get_alignment_positions(lms, eyes_distance_only: bool = True):

    # Parse landmarks.
    lm_chin          = lms[0  : 17]  # left-right
    lm_eyebrow_left  = lms[17 : 22]  # left-right
    lm_eyebrow_right = lms[22 : 27]  # left-right
    lm_nose          = lms[27 : 31]  # top-down
    lm_nostrils      = lms[31 : 36]  # top-down
    lm_eye_left      = lms[36 : 42]  # left-clockwise
    lm_eye_right     = lms[42 : 48]  # left-clockwise
    lm_mouth_outer   = lms[48 : 60]  # left-clockwise
    lm_mouth_inner   = lms[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    if eyes_distance_only:
        x *= np.hypot(*eye_to_eye) * 2.0
    else:
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1

    return c, x, y

def get_alignment_transformation(c: np.ndarray, x: np.ndarray, y: np.ndarray, scale: float):
    x_ = copy.deepcopy(x) * scale
    y_ = copy.deepcopy(y) * scale
    quad = np.stack([c - x_ - y_, c - x_ + y_, c + x_ + y_, c + x_ - y_])
    qsize = np.hypot(*x_) * 2
    return quad, qsize

def crop_face_by_transform(img: str, quad: np.ndarray, qsize: int, output_size: int = 1024,
                           transform_size: int = 1024, enable_padding: bool = True):
    # read image
    # from PIL import Image
    img = Image.fromarray(img)
    # img = Image.open(filepath)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img

def align_face(img, lms, scale=0.55, y_pos=115):
    '''
    Params
        img: np.array() # (H, W, C)
        lms: np.array() # (68, 2)
        scale : float   # scale up-> face smaller
        y_pos : int     # y position [115: rom02 sihun, 55: random metahuman]
    '''
    c, x, y = get_alignment_positions(lms)
    c[1] = c[1]+y_pos # rom02 sihun y position
    # c[1] = c[1]+55 # random metahuman y position
    quad, qsize = get_alignment_transformation(c, x, y, scale)
    img = crop_face_by_transform(img, quad, qsize)
    return img
'''
c : [556.19995 356.72498]
x : [416.5580636    9.97294593]
y : [ -9.97294593 416.5580636 ]
qaud : [[ 83.93185703  15.50016796]
 [ 67.97514355 681.99306973]
 [734.46804532 697.94978321]
 [750.4247588   31.45688144]]
 qsize 666.6838867187502
'''

def gen_mask(lms, shape):
    import skimage
    import copy
    """
    return mask
    """
    lms_ = lms
    face_h = int(abs(lms_[20,1] - lms_[8,1]))
    ldmks = copy.deepcopy(lms_) 
    for i in range(17,27):
        ldmks[i,1] -= int(face_h * 0.15)
    outline = ldmks[[*range(17), *range(26,16,-1)]]

    ### get mask outline
    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
    mask = np.zeros(shape).astype(float)
    X = [i if i < shape[0] else shape[0]-1 for i in X]
    X = [i if i > 0 else 0 for i in X]
    Y = [i if i < shape[0] else shape[0]-1 for i in Y]
    Y = [i if i > 0 else 0 for i in Y]
    mask[Y,X] = 1

    # soft_mask, _ = soft_erose(mask)

    ### return
    return mask # shape: (h,w) --> dim is 2

def tensor2sq(x, shift=0):
    height, width = x.shape[-2], x.shape[-1]
    diff = height - width

    if diff < 0:
        # top = -int(diff*0.5)
        # bot = -(diff - top)
        top  = int(shift * width)
        bot  = -diff - top
        x    = F.pad(input=x, pad=(0, 0, top, bot), mode='constant', value=-1.0)
        return x
    else:
        # left = int(diff*0.5)
        # right = diff - left
        left = int(shift * height)
        right= diff - left
        x    = F.pad(input=x, pad=(left, right, 0, 0), mode='constant', value=-1.0)
        return x 

def normalize_idm(ldm):
    # ldm = ldm - ldm.mean()
    W_min_l, W_max_l = ldm_min_max(ldm, scale=1.0, Axis="W", get_int=False)
    H_min_l, H_max_l = ldm_min_max(ldm, scale=1.0, Axis="H", get_int=False)

    heigth = H_max_l - H_min_l
    width  = W_max_l - W_min_l
    diff   = heigth  - width

    if diff < 0:
        ldm = ldm - ldm.min(0)
        ldm = ldm / width

        shift = 0.5 - ldm[:, 1].mean()
        ldm[:, 1] = ldm[:, 1] + shift
    else:
        ldm = ldm - ldm.min(0)
        ldm = ldm / heigth

        shift = 0.5 - ldm[:, 0].mean()
        ldm[:, 0] = ldm[:, 0] + shift

    ldm = ldm * 2.0 -1.0
    return ldm, shift

def norm_range(img):
    if img.min() < 0:  # [-1 ~ 1] >>> [ 0 ~ 1]
        return (img * 0.5) + 0.5
    if img.min() >= 0: # [ 0 ~ 1] >>> [-1 ~ 1]
        return (img * 2.0) - 1.0

def ldm_min_max(part_ldm, scale=1.0, Axis="", get_int=True):
    A = 0 if Axis == "W" else 1
    if get_int:
        return int(part_ldm[..., A].min()*scale), int(part_ldm[..., A].max()*scale)
    else:
        return part_ldm[..., A].min()*scale, part_ldm[..., A].max()*scale

class SoftErosion(object):
    def __init__(self, kernel_size=11, threshold=0.9, iterations=1):
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = np.meshgrid(np.arange(0., kernel_size), np.arange(0., kernel_size))
        dist = np.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        self.weight = kernel

    def __call__(self, x):
        x = x
        print(x.shape)
        for i in range(self.iterations - 1):
            tmp = scipy.signal.convolve2d(x, self.weight, mode='same')
            x = np.min(np.stack((x, tmp),axis=2), axis=2)
        x = scipy.signal.convolve2d(x, self.weight, mode='same')

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        return x, mask

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)
        # self.min_cutoff = np.ones_like(x0) * min_cutoff
        # self.beta       = np.ones_like(x0) * beta
        # self.d_cutoff   = np.ones_like(x0) * d_cutoff
        # # Previous values.
        # self.x_prev  = x0
        # self.dx_prev = np.zeros_like(dx0)
        # self.t_prev  = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class LowPassFilter:
    def __init__(self, cutoff_freq, ts):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        self.pre_out = 0.
        self.tau = self.calc_filter_coef() 

    def calc_filter_coef(self):
        w_cut = 2*np.pi*self.cutoff_freq
        return 1/w_cut

    def __call__(self, data):
        out = (self.tau * self.pre_out + self.ts * data) / (self.tau + self.ts)
        self.pre_out = out
        return out

"""
TODO:
    Need to define position
"""
DST_Landmark = {
    # "chin":          preds_pos[0 :17], # (left-right)
    # "eyebrow_left":  preds_pos[17:22], # (left-right)
    # "eyebrow_right": preds_pos[22:27], # (left-right)
    # "nose":          preds_pos[27:31], # (top-down)
    # "nostrils":      preds_pos[31:36], # (top-down)
    "eye_left": torch.tensor([[ # (left-clockwise)
            [-1.0, -0.2],
            [-0.4, -0.9],
            [ 0.5, -0.9],
            [ 1.0,  0.3],
            [ 0.5,  0.9],
            [-0.4,  0.9],
        ]]), 
    # "eye_right":     preds_pos[42:48], # (left-clockwise)
    # "mouth_outer":   preds_pos[48:60], # (left-clockwise)
    # "mouth_inner":   preds_pos[60:68], # (left-clockwise)
}

def main(opts):
    # from kornia.filters import gaussian_blur2d
    from kornia.filters.kernels import get_gaussian_kernel1d
    import torch.nn.functional as F
    import torch.nn as nn
    from tqdm import tqdm

    # load face / landmark detection modul
    # import face_alignment
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
    lms_criterion = LandmarkCriterion("cuda", False)

    # load files
    image_list = glob.glob(os.path.join(opts.input_dir, "*.png"))
    image_list += glob.glob(os.path.join(opts.input_dir, "*.jpg"))
    image_list = sorted(image_list)

    len_image_list = len(image_list)
    print("image len: ",len_image_list)


    os.makedirs(opts.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.save_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(opts.save_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(opts.save_dir, "comp"), exist_ok=True)

    # soft_erose = SoftErosion()

    # lms_pos_all = np.zeros([len_image_list,68,2])

    # has_lms = False

    # ## pred face-landmarks
    # if os.path.isfile(opts.save_dir+"/"+"lms_pos_all.npy"):
    #     lms_pos_all = np.load(opts.save_dir+"/"+"lms_pos_all.npy")
    #     print("loaded: "+opts.save_dir+"/"+"lms_pos_all.npy")
    # else:
    #     pbar = tqdm(enumerate(image_list),total=len_image_list)
    #     for index, f in pbar:
    #         name, ext = os.path.basename(f).split(".")

    #         pbar.set_description(">>>> face-landmarks {}".format(name))
    #         # print(">>>>", name)

    #         img = mp.read_image(f)
    #         img = img[:,:,:3]
    #         h,w,_ = img.shape

    #         pil_img = Image.open(f).convert('RGB')
    #         torch_img = T(pil_img).unsqueeze(0).cuda()

    #         # get lms pos
    #         lms_pos = fa.get_landmarks(img)[0]
    #         # import pdb;pdb.set_trace()
    #         lms_pos_all[index]=lms_pos
    #     np.save(opts.save_dir+"/"+"lms_pos_all", lms_pos_all)

    # lms_pos_all = torch.from_numpy(lms_pos_all).type(torch.float32) # 2000, 68, 2
    # lms_pos_all = lms_pos_all.permute(2,1,0) # 2, 68, 2000
    # lms_pos_all = lms_pos_all.unsqueeze(0)  # 1, 2, 68, 2000

    # ## Apply Gaussian filter
    # # sigma = 1.5
    # sigma = 11.0
    # kernel_width = 11
    # kernel = torch.ones([1, 1, 1, kernel_width]) * get_gaussian_kernel1d(kernel_width, sigma) #.unsqueeze(1)

    # padding_width = int((kernel_width) / 2)
    # padding = nn.ReflectionPad2d((padding_width, padding_width, 0, 0))
    # lms_pos_all_padd = padding(lms_pos_all)

    # for ldmk in range(68):
    #     for coord in range(2):
    #         lms_pos_all[:,coord,ldmk] = F.conv2d(lms_pos_all_padd[:,coord,ldmk][None][None], kernel, padding=0)[0][0]
    # lms_pos_all = lms_pos_all[0].permute(2,1,0).numpy() # 2000, 68, 2


    pbar = tqdm(enumerate(image_list), total=len_image_list)
    for index, f in pbar:
        name, ext = os.path.basename(f).split(".")
        pbar.set_description(">>>> processing {}".format(name))
        # lms_pos = blur_lms_pos_all[index]
        # lms_pos = lms_pos_all[index]

        img = mp.read_image(f)
        img = img[:,:,:3]
        h,w,_ = img.shape

        pil_img = Image.open(f).convert('RGB')
        torch_img = T(pil_img).unsqueeze(0).cuda()
        torch_img_512 = T512(pil_img).unsqueeze(0).cuda() # for cropping

        # get preds_pos (hrnet)
        ### hrnet mask ###
        preds_heat = lms_criterion.get_heatmap(torch_img) # ([1, 68, 64, 64])
        size_heat = preds_heat.size(-1) # 64
        preds_pos = lms_criterion.get_pos(preds_heat) # [bs, lms, xy] ([1, 68, 2]) ~64

        # scale_pos = h / size_heat
        # preds_pos = preds_pos * scale_pos
        scale_pos = 1 / size_heat
        preds_pos = (preds_pos * scale_pos) * 2.0 - 1.0 # normalize
        preds_pos = preds_pos[0].detach().cpu().numpy() # (68, 2)

        # Parse landmarks.
        landmarks = {
            "chin":          preds_pos[0 :17], # (left-right)
            "eyebrow_left":  preds_pos[17:22], # (left-right)
            "eyebrow_right": preds_pos[22:27], # (left-right)
            "nose":          preds_pos[27:31], # (top-down)
            "nostrils":      preds_pos[31:36], # (top-down)
            "eye_left":      preds_pos[36:42], # (left-clockwise)
            "eye_right":     preds_pos[42:48], # (left-clockwise)
            "mouth_outer":   preds_pos[48:60], # (left-clockwise)
            "mouth_inner":   preds_pos[60:68], # (left-clockwise)
        }
        for ldm, prt_ldm in landmarks.items():
            if ldm != "eye_left":
                continue
            norm_prt_ldm, shift = normalize_idm(prt_ldm)

            ## local landmark

            size = 512
            W_min_l, W_max_l = ldm_min_max((prt_ldm * 0.5 + 0.5) * (size-1), scale=1.0, Axis="W")
            H_min_l, H_max_l = ldm_min_max((prt_ldm * 0.5 + 0.5) * (size-1), scale=1.0, Axis="H")
            image_bfr = torch_img_512[:, :, H_min_l:H_max_l, W_min_l:W_max_l]
            image_bfr = tensor2sq(image_bfr, shift)
            new_size =  image_bfr.shape[-1]

            src_p = torch.tensor(norm_prt_ldm)[None].cuda()

            ## warped landmark
            # dst_p = src_p.clone()
            dst_p = DST_Landmark["eye_left"].cuda()
            # dst_p[0][0,1] = dst_p[0][0,1] / dst_p[0][0,1] * 0.3
            # dst_p[0][1:3,1] = dst_p[0][1:3,1] / dst_p[0][1:3,1] * -0.9
            # dst_p[0][3,1]   = dst_p[0][3,1] / dst_p[0][3,1] * 0.5
            # dst_p[0][4:6,1] = dst_p[0][4:6,1] / dst_p[0][4:6,1] * 0.8

            c = torch.zeros(1,1,2).cuda()
            src_p = torch.cat((src_p, c), dim=1)
            dst_p = torch.cat((dst_p, dst_p.mean(1)[None]), dim=1)

            print(src_p)
            print(dst_p)


            ### using kornia TPS
            kernel_weights, affine_weights = kgt.get_tps_transform(dst_p, src_p)
            image_aft = kgt.warp_image_tps(image_bfr, src_p, kernel_weights, affine_weights, align_corners=True)

            ### sanity check
            tg_src_p2img_p = norm_range(src_p) * (new_size-1)
            tg_dst_p2img_p = norm_range(dst_p) * (new_size-1)

            image_bfr = (norm_range(image_bfr) * 255).type(torch.uint8)
            image_aft = (norm_range(image_aft) * 255).type(torch.uint8)
            image_bfr = draw_keypoints(image_bfr[0], tg_src_p2img_p, colors="blue", radius=1)
            image_aft = draw_keypoints(image_aft[0], tg_dst_p2img_p, colors="red", radius=1)

            final_image = torch.cat((image_bfr, image_aft),dim=-1) / 255.0
            save_image(final_image, f"{opts.save_dir}/img/{name}.png", normalize=True)
            # import pdb;pdb.set_trace()
            break
        break

if __name__ =="__main__":
    ### options
    import argparse

    parser = argparse.ArgumentParser(description="image alignment")
    parser.add_argument("--input_dir", type=str, default="./test")
    parser.add_argument("--save_dir",  type=str, default="./output")
    opts = parser.parse_args()

    main(opts)
