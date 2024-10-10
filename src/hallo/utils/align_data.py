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
import cv2

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

'''
CHECK scale, y position
'''
# scale = 0.55
# y_pos = 115
scale = 0.55
y_pos = 115
# scale up-> face smaller
def align_face(img, lms, scale=scale):
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

T = transforms.Compose([
            transforms.Resize([256,256]), # assert resolution == 256
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

import math


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


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

# from align_data import smoothing_factor, exponential_smoothing, OneEuroFilter, LowPassFilter; import numpy as np

def main(opts):
    # from kornia.filters import gaussian_blur2d
    from kornia.filters.kernels import get_gaussian_kernel1d
    import torch.nn.functional as F

    import torch.nn as nn
    from tqdm import tqdm

    # load face / landmark detection modul
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
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

    soft_erose = SoftErosion()

    lms_pos_all = np.zeros([len_image_list,68,2])
    # lms_pos_all = np.random.rand(len_image_list,68,2).

    has_lms = False

    if os.path.isfile(opts.save_dir+"lms_pos_all.npy"):
        lms_pos_all = np.load(opts.save_dir+"lms_pos_all.npy")
        print("loaded: "+opts.save_dir+"lms_pos_all.npy")

    else:
        pbar = tqdm(enumerate(image_list),total=len_image_list)
        for index, f in pbar:
            name, ext = os.path.basename(f).split(".")

            pbar.set_description(">>>> {}".format(name))
            # print(">>>>", name)

            img = mp.read_image(f)
            img = img[:,:,:3]
            h,w,_ = img.shape

            pil_img = Image.open(f).convert('RGB')
            torch_img = T(pil_img).unsqueeze(0).cuda()

            # get lms pos
            lms_pos = fa.get_landmarks(img)[0]
            # import pdb;pdb.set_trace()
            lms_pos_all[index]=lms_pos

        np.save(opts.save_dir+"lms_pos_all", lms_pos_all)

    lms_pos_all = torch.from_numpy(lms_pos_all).type(torch.float32) # 2000, 68, 2
    lms_pos_all = lms_pos_all.permute(2,1,0) # 2, 68, 2000
    lms_pos_all = lms_pos_all.unsqueeze(0)  # 1, 2, 68, 2000

    # kernel
    # sigma = 1.5
    sigma = 11

    kernel_width = 11
    kernel = torch.ones([1, 1, 1, kernel_width]) * get_gaussian_kernel1d(kernel_width, sigma) #.unsqueeze(1)

    padding_width = int((kernel_width) / 2)
    padding = nn.ReflectionPad2d((padding_width, padding_width, 0, 0))
    # import pdb;pdb.set_trace()
    lms_pos_all_padd = padding(lms_pos_all)

    for ldmk in range(68):
        for coord in range(2):
            # import pdb;pdb.set_trace()
            lms_pos_all[:,coord,ldmk] = F.conv2d(lms_pos_all_padd[:,coord,ldmk][None][None], kernel, padding=0)[0][0]
    lms_pos_all = lms_pos_all[0].permute(2,1,0).numpy() # 2000, 68, 2


    pbar = tqdm(enumerate(image_list), total=len_image_list)
    for index, f in pbar:
        name, ext = os.path.basename(f).split(".")
        pbar.set_description(">>>>{}".format(name))
        # lms_pos = blur_lms_pos_all[index]
        lms_pos = lms_pos_all[index]

        img = mp.read_image(f)
        img = img[:,:,:3]
        h,w,_ = img.shape

        pil_img = Image.open(f).convert('RGB')
        torch_img = T(pil_img).unsqueeze(0).cuda()

        # get preds_pos (hrnet)
        ### hrnet mask ###
        preds_heat = lms_criterion.get_heatmap(torch_img) # ([1, 68, 64, 64])
        size_heat = preds_heat.size(-1) # 64
        preds_pos = lms_criterion.get_pos(preds_heat) # [bs, lms, xy] ([1, 68, 2]) ~64
        # print(preds_pos)

        scale_pos = h / size_heat
        preds_pos = preds_pos * scale_pos
        preds_pos = preds_pos[0].detach().cpu().numpy()

        # get mask
        full_mask = gen_mask(preds_pos, shape=(h,w)).astype(float)
        full_mask = np.expand_dims(full_mask, axis=2)
        full_mask = np.tile(full_mask, [1, 1, 3])
        full_mask = (full_mask*255).astype(np.uint8)

        # align
        align_mask = align_face(full_mask, lms_pos[:,:2])
        align_img = align_face(img, lms_pos[:,:2])

        align_img = align_img.resize((256,256))
        align_mask = align_mask.resize((256,256))

        # save
        # mp.write_image(f"{opts.save_dir}/img/{name}.{ext}", align_img)
        mp.write_image(f"{opts.save_dir}/img/{name}.png", align_img)
        mp.write_image(f"{opts.save_dir}/mask/{name}.png", align_mask)

        # sanity
        comp = (np.array(align_img).astype(float) * \
            np.array(align_mask).astype(float)/255.0).astype(np.uint8)
        mp.write_image(f"{opts.save_dir}/comp/{name}.png", comp)


if __name__ =="__main__":
    ### options
    import argparse

    parser = argparse.ArgumentParser(description="image alignment")
    parser.add_argument("--input_dir", type=str, default="./src/real")
    parser.add_argument("--save_dir",  type=str, default="./out/real_0.55_115")
    opts = parser.parse_args()

    main(opts)
