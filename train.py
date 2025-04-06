# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:01:58 2020

@author: ZJU
"""

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
# parser.add_argument('--content_dir', type=str, default=r'E:\00propropropro\dataset\train2014CoCo\train2014',
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str, default=r'E:\00propropropro\dataset\wikiart_1\train_1',
#                     help='Directory path to a batch of style images')
parser.add_argument('--content_dir', type=str, default=r'/mnt/nfs/data/home/1120220284/datasets/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default=r'/mnt/nfs/data/home/1120220284/datasets/train',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./full_experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
# parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--style_weight', type=float, default=1.0)


parser.add_argument('--content_transitive_loss_weight', type=float, default=0.5)
parser.add_argument('--style_transitive_loss_weight', type=float, default=0.5)
parser.add_argument('--style_diff_loss_weight', type=float, default=1.0)
parser.add_argument('--content_diff_loss_weight', type=float, default=1.0)
parser.add_argument('--content_restoration_loss_weight', type=float, default=0.5)
parser.add_argument('--style_restoration_loss_weight', type=float, default=0.5)

# parser.add_argument('--contrastive_weight_c', type=float, default=0.3)
# parser.add_argument('--contrastive_weight_s', type=float, default=0.3)
parser.add_argument('--gan_weight', type=float, default=5.0)
# parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda')
print(device)
print(os.environ['CUDA_VISIBLE_DEVICES'])

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print("可用的GPU数量：", device_count)
    else:
        print("没有可用的GPU设备")
else:
    print("没有安装CUDA或者没有可用的GPU设备")

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

valid = 1
fake = 0
D = net.MultiDiscriminator()
D.to(device)

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    # content_dataset, batch_size=int(args.batch_size / 2),
    content_dataset, batch_size=int(args.batch_size),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    # style_dataset, batch_size=int(args.batch_size / 2),
    style_dataset, batch_size=int(args.batch_size),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transformer.parameters()},
                              {'params': network.featureFusion.parameters()}], lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))
    # Load featureFusion parameters
    network.featureFusion.load_state_dict(
        torch.load('{:s}/featureFusion_iter_{:d}.pth'.format(args.save_dir, args.start_iter)))

# 为tqdm创建一个实例
progress_bar = tqdm(range(args.start_iter, args.max_iter))
for i in progress_bar:
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)



    img, rc1, rs1, output2, rc2, rs2, l_identity1, l_identity2, content_transitive_loss, style_transitive_loss, style_diff_loss, content_diff_loss, content_restoration_loss, style_restoration_loss = network(content_images, style_images, args.batch_size)

    # train discriminator
    loss_gan_d = D.compute_loss(style_images, valid) + D.compute_loss(img.detach(), fake) + D.compute_loss(content_images, valid) + D.compute_loss(output2.detach(), fake)
    optimizer_D.zero_grad()
    loss_gan_d.backward()
    optimizer_D.step()


    content_transitive_loss = args.content_transitive_loss_weight * content_transitive_loss
    style_transitive_loss = args.style_transitive_loss_weight * style_transitive_loss
    style_diff_loss = args.style_diff_loss_weight * style_diff_loss
    content_diff_loss = args.content_diff_loss_weight * content_diff_loss
    content_restoration_loss = args.content_restoration_loss_weight * content_restoration_loss
    style_restoration_loss = args.style_restoration_loss_weight * style_restoration_loss


    loss_gan_g = args.gan_weight * D.compute_loss(img, valid) + args.gan_weight * D.compute_loss(output2, valid)
    loss = content_transitive_loss + style_transitive_loss + style_diff_loss + content_diff_loss + content_restoration_loss + style_restoration_loss + l_identity1 * 50 + l_identity2 * 1 + loss_gan_g
    # loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1 + loss_contrastive_c + loss_contrastive_s + loss_gan_g

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # writer.add_scalar('loss_content', loss_c.item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('content_transitive_loss', content_transitive_loss.item(), i + 1)
    writer.add_scalar('style_transitive_loss', style_transitive_loss.item(), i + 1)
    writer.add_scalar('style_diff_loss', style_diff_loss.item(), i + 1)
    writer.add_scalar('content_diff_loss', content_diff_loss.item(), i + 1)
    writer.add_scalar('content_restoration_loss', content_restoration_loss.item(), i + 1)
    writer.add_scalar('style_restoration_loss', style_restoration_loss.item(), i + 1)


    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    # writer.add_scalar('loss_contrastive_c', loss_contrastive_c.item(), i + 1)  # attention
    # writer.add_scalar('loss_contrastive_s', loss_contrastive_s.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)  # attention

    writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i == 0) or ((i + 1) % 500 == 0):
        output = torch.cat([style_images, content_images, img], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(output, str(output_name), nrow=args.batch_size)

    # 使用set_postfix方法在进度条后添加损失值
    progress_bar.set_postfix({
        # 'Content Loss': '{:.4f}'.format(loss_c),
        # 'Style Loss': '{:.4f}'.format(loss_s),
        'content_transitive_loss Loss': '{:.4f}'.format(content_transitive_loss),
        'style_transitive_loss Loss': '{:.4f}'.format(style_transitive_loss),
        'style_diff_loss Loss': '{:.4f}'.format(style_diff_loss),
        'content_diff_loss Loss': '{:.4f}'.format(content_diff_loss),
        'content_restoration_loss Loss': '{:.4f}'.format(content_restoration_loss),
        'style_restoration_loss Loss': '{:.4f}'.format(style_restoration_loss),
        'Identity1 Loss': '{:.4f}'.format(l_identity1),
        'Identity2 Loss': '{:.4f}'.format(l_identity2),
        'GAN G Loss': '{:.4f}'.format(loss_gan_g),
        'GAN D Loss': '{:.4f}'.format(loss_gan_d)
    })

    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        # Save state of featureFusion module
        state_dict = network.featureFusion.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, '{:s}/featureFusion_iter_{:d}.pth'.format(args.save_dir, i + 1))

        # Save state of transformer module which includes AdaptiveMultiAdaAttN_v2 and AdaptiveMultiAttn_Transformer_v2
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
writer.close()