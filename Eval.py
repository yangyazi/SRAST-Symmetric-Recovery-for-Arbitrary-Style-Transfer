import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net



def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512,512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')



parser.add_argument('--content_dir', type=str,default = r'E:\00000000000000000\testinput\content1\1',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,default = r'E:\00000000000000000\testinput\style1',
                    help='Directory path to a batch of style images')
#######

parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = r'experiments1/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default = r'experiments1/transformer_iter_160000.pth')
parser.add_argument('--featureFusion', type=str, default = r'experiments1/featureFusion_iter_160000.pth')

# Additional options

parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = r'E:\00000000000000000\testoutput\test',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
print(device)
print(os.environ['CUDA_VISIBLE_DEVICES'])

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
query_channels = 512  # +256+128+64
key_channels = 512 + 256 + 128 + 64
transform = net.AdaptiveMultiAttn_Transformer_v2(in_planes=512, out_planes=512, query_planes=query_channels, key_planes=key_channels)
vgg = net.vgg
featureFusion = net.FeatureFusion(512)

decoder.eval()
transform.eval()
vgg.eval()
featureFusion.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))
featureFusion.load_state_dict(torch.load(args.featureFusion))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1


def encode_with_intermediate(input, encoder):
    results = [input]
    for i in range(5):
        func = encoder['enc_{:d}'.format(i + 1)]   #这里用字典替代原来的self.功能
        results.append(func(results[-1]))
    return results[1:]

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)
featureFusion.to(device)


content_tf = test_transform()
style_tf = test_transform()
output_dir = Path(args.output)
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]



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

def adaptive_get_keys(feats, start_layer_idx, last_layer_idx):
    results = []
    _, _, h, w = feats[last_layer_idx].shape
    for i in range(start_layer_idx, last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[last_layer_idx]))
    return torch.cat(results, dim=1)

def AdaIN(content, style):

    style_strength=1.0
    eps=1e-5
    b, c, h, w = content.size()

    content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
    style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)

    normalized_content = (content.view(b, c, -1) - content_mean) / (content_std + eps)

    stylized_content = (normalized_content * style_std) + style_mean

    output = (1 - style_strength) * content + style_strength * stylized_content.view(b, c, h, w)
    return output


for content_path in content_paths:
    print(content_path)
    for style_path in style_paths:
        print(style_path)
        content = content_tf(Image.open(str(content_path)))
        style = style_tf(Image.open(str(style_path)))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():

            print(content_path.stem + ' iteration ' + style_path.stem)
            encoder = {'enc_1': enc_1, 'enc_2': enc_2, 'enc_3': enc_3, 'enc_4': enc_4, 'enc_5': enc_5}
            style_feats = encode_with_intermediate(style, encoder)
            content_feats = encode_with_intermediate(content, encoder)

            content = decoder(featureFusion(transform(
                content_feats[3], style_feats[3], content_feats[4], style_feats[4],
                adaptive_get_keys(content_feats, 3, 3),
                adaptive_get_keys(style_feats, 0, 3),
                adaptive_get_keys(content_feats, 4, 4),
                adaptive_get_keys(style_feats, 0, 4))[0], AdaIN(content_feats[3], style_feats[3])))

            content.clamp(0, 255)
            content = content.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            print(output_name)
            save_image(content, str(output_name))
