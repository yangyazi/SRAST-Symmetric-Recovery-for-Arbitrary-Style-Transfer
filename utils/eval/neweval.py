# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:35:21 2023

@author: LY
"""
import argparse
import os
import re
from pathlib import Path

import scipy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(r"E:\00000000000000000\newnew\IEContraAST003")
import piqa
from piqa import LPIPS
# from piqa.lpips import LPIPS
# from piqa.psnr import psnr as PSNR
# from piqa.ssim import ssim as SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
print(device)
print(os.environ['CUDA_VISIBLE_DEVICES'])

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the IEContraAST-Last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# def test_transform(size, crop):
#     transform_list = []
#     if size != 0:
#         transform_list.append(transforms.Resize(size))
#     if crop:
#         transform_list.append(transforms.CenterCrop(size))
#     transform_list.append(transforms.ToTensor())
#     transform = transforms.Compose(transform_list)
#     return transform
def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512,512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# def test_transform1(size, crop):
#     transform = transforms.Compose([
#         transforms.Resize((height, width)),
#         transforms.ToTensor()
#     ])
#     return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--ref', type=str,
                    help='File path to the ref image')
#######
# parser.add_argument('--content_dir', type=str,default = r'/mnt/nfs/data/home/1120220284/myexper/testinput/content',
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str,default = r'/mnt/nfs/data/home/1120220284/myexper/testinput/style',
#                     help='Directory path to a batch of style images')
# parser.add_argument('--ref_dir', type=str,default = r'/mnt/nfs/data/home/1120220284/myexper/testoutput/fulloutput/fulloutput1',
#                     help='Directory path to a batch of ref images')
#
parser.add_argument('--content_dir', type=str,default = r'D:\Users\luyang\Desktop\最终选定图片\content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,default = r'D:\Users\luyang\Desktop\最终选定图片\style',
                    help='Directory path to a batch of style images')
parser.add_argument('--ref_dir', type=str,default = r'D:\Users\luyang\Desktop\最终选定图片\final_output\final_output\my',
                    help='Directory path to a batch of ref images')

# parser.add_argument('--content_dir', type=str,default = r'E:\00000000000000000\testinput\content2',
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str,default = r'E:\00000000000000000\testinput\style2',
#                     help='Directory path to a batch of style images')
# parser.add_argument('--ref_dir', type=str,default = r'E:\00000000000000000\testoutput\test',
#                     help='Directory path to a batch of ref images')
#######
parser.add_argument('--vgg', type=str, default = '../../model/vgg_normalised.pth')

# Additional options

parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--ref_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--outputtxtname', default = 'woloss.txt',
                    help='The extension name of the output image')
# Advanced options

args = parser.parse_args()

vgg.load_state_dict(torch.load(args.vgg))
vgg.eval()

norm = nn.Sequential(*list(vgg.children())[:1])
def create_encoder(vgg):
    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)

    encoders = [enc_1, enc_2, enc_3, enc_4, enc_5]

    def encode_with_intermediate(input):
        results = []
        for enc in encoders:
            input = enc(input)
            results.append(input)
        return results

    return encode_with_intermediate

# 使用方式
encode_with_intermediate = create_encoder(vgg)


#############################
# content_tf = test_transform(args.content_size, args.crop)
# style_tf = test_transform(args.style_size, args.crop)
# ref_tf = test_transform(args.ref_size, args.crop)
content_tf = test_transform()
style_tf = test_transform()
ref_tf = test_transform()

assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]


assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]


assert (args.ref or args.ref_dir)
if args.ref:
    ref_paths = [Path(args.ref)]
else:
    ref_dir = Path(args.ref_dir)
    ref_paths = [f for f in ref_dir.glob('*')]



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

mse_loss = nn.MSELoss()

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def calc_content_loss(input, target, norm = False):
    if(norm == False):
      return mse_loss(input, target)
    else:
      return mse_loss(mean_variance_norm(input), mean_variance_norm(target))

def calc_style_loss(input, target):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)

def find_matching_ref_paths1(content_path, ref_paths):
    name_to_match = content_path.stem[1:]
    # matching_paths = []
    for ref_path in ref_paths:
        if name_to_match in ref_path.stem:
            return(ref_path)


def find_matching_ref_paths(content_path, ref_paths, mode='style'):
    # if mode == 'style':
    #     name_to_match = re.search(r's(\d+)', content_path.stem).group(1)
    #     print("s")
    #     print(name_to_match)
    # elif mode == 'content':
    #     name_to_match = re.search(r'c(\d+)', content_path.stem).group(1)
    #     print("c")
    #     print(name_to_match)

    matching_paths = []
    for ref_path in ref_paths:
        if mode == 'style':
            ref_name = re.search(r's(\d+)', ref_path.stem)
            name_to_match = re.search(r's(\d+)', content_path.stem).group(1)

        elif mode == 'content':
            ref_name = re.search(r'c(\d+)', ref_path.stem)
            name_to_match = re.search(r'c(\d+)', content_path.stem).group(1)

        if ref_name and ref_name.group(1) == name_to_match:
            matching_paths.append(ref_path)

    return matching_paths



single_style_transfer_sloss = []
single_style_transfer_closs = []

single_style_transfer_psnr = []
single_style_transfer_ssim = []
single_style_transfer_lpips = []


N = 0
sumpsnr, sumssim, sumlpips = 0., 0., 0.



for style_path in style_paths:


    print("style_path: " + str(style_path) + "\n")
    style = style_tf(Image.open(str(style_path)))
    # print("style shape: {}".format(style.shape))  # 使用 format() 方法将元组转换为字符串

    style = style.to(device).unsqueeze(0)
    style_feats = encode_with_intermediate(style)

    ref_paths1 = find_matching_ref_paths(style_path, ref_paths)
    sloss = []
    n = 0
    for ref_path in ref_paths1:
        print("ref_path: " + str(ref_path) + "\n")
        ref = ref_tf(Image.open(str(ref_path)))
        # print("ref shape: {}".format(ref.shape))  # 使用 format() 方法将元组转换为字符串
        ref = ref.to(device).unsqueeze(0)
        with torch.no_grad():
            style_feats = encode_with_intermediate(style)
            ref_feats = encode_with_intermediate(ref)
            loss_s = calc_style_loss(style_feats[0], ref_feats[0])
            for i in range(1, 5):
                loss_s += calc_style_loss(style_feats[i], ref_feats[i])
            sloss.append(loss_s)
        n += 1
    single_style_transfer_sloss.append(sum(sloss)/n)
    N += 1
    print("单图片风格损失：")
    for i, s_loss in enumerate(sloss):
        print(f"Iteration {i}: {s_loss}")
    print(f"单图片总风格损失：{sum(sloss)}")
    print(f"单图片总风格损失均值：{sum(sloss)/n}")

total_style_transfer_sloss = sum(single_style_transfer_sloss)


print("\n" + "所有损失汇总" + "\n")
print("风格损失：")
for i, s_loss in enumerate(single_style_transfer_sloss):
    print(f"Iteration {i}: {s_loss}")
print(f"总风格损失：{total_style_transfer_sloss}")

if not os.path.exists(args.outputtxtname):
    with open(args.outputtxtname, 'w') as file:
        pass  # 创建一个空文件

with open(args.outputtxtname, 'a') as file:
    file.write(os.path.basename(args.ref_dir))
    file.write("\n")
    for i in range(N):
        file.write(f"风格图片: {style_paths[i].stem}, ")
        file.write(f"第{i+1}张: {single_style_transfer_sloss[i]}, ")
        file.write("\n")
    file.write(f"总风格损失：{total_style_transfer_sloss}, {N}张风格图片\n")
    file.write("\n")

M = 0
for content_path in content_paths:
    print("content_path: " + str(content_path) + "\n")
    # 获取图像的宽度和高度
    # width, height = Image.open(str(content_path)).size
    content = content_tf(Image.open(str(content_path)))
    # print("content shape: {}".format(content.shape))  # 使用 format() 方法将元组转换为字符串
    content = content.to(device).unsqueeze(0)


    closs = []
    ssim = []
    lpips = []
    psnr = []
    m = 0


    ref_path1 = find_matching_ref_paths(content_path, ref_paths, mode='content')
    for ref_path in ref_path1:
        m += 1
        print("ref_path: " + str(ref_path) + "\n")
        # ref_tf = transforms.Compose([
        #     transforms.Resize((height, width)),
        #     transforms.ToTensor()
        # ])


        ref = ref_tf(Image.open(str(ref_path)))
        # print("ref shape: {}".format(ref.shape))  # 使用 format() 方法将元组转换为字符串
        ref = ref.to(device).unsqueeze(0)

        # print(ref.shape, ref_path.stem)
        # print(content.shape, content_path.stem)
        # print(style.shape, style_path.stem)


        # 计算gram损失，风格损失，内容损失
        with torch.no_grad():
            content_feats = encode_with_intermediate(content)
            ref_feats = encode_with_intermediate(ref)
            loss_c = calc_content_loss(content_feats[3], ref_feats[3], norm=True) + calc_content_loss(content_feats[4], ref_feats[4],norm=True)

            closs.append(loss_c)

            # loss_s = calc_style_loss(style_feats[0], ref_feats[0])
            # for i in range(1, 5):
            #     loss_s += calc_style_loss(style_feats[i], ref_feats[i])
            # sloss.append(loss_s)


        # 计算 ssim psnr lipis

        # a = content
        # b = ref
        x = content
        y = ref

        _psnr = piqa.PSNR()
        l_psnr = _psnr(x, y)

        ssim1 = piqa.SSIM().cuda()
        l_ssim = 1 - ssim1(x, y)

        lpips_model = LPIPS(network='vgg')
        lpips_model.to(device)
        lpips1 = lpips_model(x, y)

        # l_lpips1 = lpips_model(x, y)


        psnr.append(l_psnr.item())
        ssim.append(l_ssim.item())
        lpips.append(lpips1)

    M += 1
    single_style_transfer_psnr.append(sum(psnr)/m)
    single_style_transfer_ssim.append(sum(ssim)/m)
    single_style_transfer_lpips.append(sum(lpips)/m)

    print('单图片psnr', sum(psnr), content_path.stem)
    print('单图片ssim', sum(ssim), content_path.stem)
    print('单图片lpips', sum(lpips), content_path.stem)
    print('单图片psnr均值', sum(psnr)/m, content_path.stem, m, '张图片')
    print('单图片ssim均值', sum(ssim)/m, content_path.stem)
    print('单图片lpips均值', sum(lpips)/m, content_path.stem)

    print("psnr")
    for i, psnr in enumerate(psnr):
        print(f"Iteration {i}: {psnr}")
    print("ssim")
    for i, ssim in enumerate(ssim):
        print(f"Iteration {i}: {ssim}")
    print("lpips")
    for i, lpips in enumerate(lpips):
        print(f"Iteration {i}: {lpips}")


    single_style_transfer_closs.append(sum(closs)/m)
    print("单图片内容损失：")
    for i, c_loss in enumerate(closs):
        print(f"Iteration {i}: {c_loss}")
    print(f"单图片总内容损失：{sum(closs)}")
    print(f"单图片总内容损失均值：{sum(closs) / m}")


# total_style_transfer_sloss = sum(single_style_transfer_sloss)
total_style_transfer_closs = sum(single_style_transfer_closs)

print("风格损失：")
for i, s_loss in enumerate(single_style_transfer_sloss):
    print(f"Iteration {i}: {s_loss}")

print("内容损失：")
for i, c_loss in enumerate(single_style_transfer_closs):
    print(f"Iteration {i}: {c_loss}")

print(f"总风格损失：{total_style_transfer_sloss}")
print(f"总内容损失：{total_style_transfer_closs}")

with open(args.outputtxtname, 'a') as file:
    file.write(f"内容图片: {content_path}, ")
    for i, s_loss in enumerate(single_style_transfer_closs):
        file.write(f"Iteration {i}: {s_loss}, ")
    file.write(f"总内容损失：{total_style_transfer_closs}")



print("psnr ssim lpips" + "\n")
total_style_transfer_psnr = sum(single_style_transfer_psnr)
total_style_transfer_ssim = sum(single_style_transfer_ssim)
total_style_transfer_lpips = sum(single_style_transfer_lpips)
mean_style_transfer_psnr = sum(single_style_transfer_psnr) / M
mean_style_transfer_ssim = sum(single_style_transfer_ssim) / M
mean_style_transfer_lpips = sum(single_style_transfer_lpips) / M
print("psnr损失：")
for i, s_loss in enumerate(single_style_transfer_psnr):
    print(f"Iteration {i}: {s_loss}")
print("ssim损失：")
for i, c_loss in enumerate(single_style_transfer_ssim):
    print(f"Iteration {i}: {c_loss}")
print("lpips损失：")
for i, c_loss in enumerate(single_style_transfer_lpips):
    print(f"Iteration {i}: {c_loss}")

print(f"总psnr损失：{total_style_transfer_psnr}", M, '张内容图片')
print(f"总ssim损失：{total_style_transfer_ssim}")
print(f"总lpips损失：{total_style_transfer_lpips}")

print(f"平均psnr损失：{mean_style_transfer_psnr}", M, '张内容图片')
print(f"平均ssim损失：{mean_style_transfer_ssim}")
print(f"平均lpips损失：{mean_style_transfer_lpips}")


with open(args.outputtxtname, 'a') as file:
    for i in range(M):
        file.write(f"内容图片: {content_paths[i].stem}\n ")
        file.write(f"第{i+1}张closs: {single_style_transfer_closs[i]}, psnr: {single_style_transfer_psnr[i]}, ssim: {single_style_transfer_ssim[i]}, lpips: {single_style_transfer_lpips[i]}")
        file.write("\n")
    file.write(f"总内容损失：{total_style_transfer_closs}, {M}张内容图片\n")
    file.write(f"总psnr损失：{total_style_transfer_psnr}\n")
    file.write(f"总ssim损失：{total_style_transfer_ssim}\n")
    file.write(f"总lpips损失：{total_style_transfer_lpips}\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")


