import argparse
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def transform_image(size):
    transform_list = [
        transforms.Resize(size=(size, size)),
        # transforms.CenterCrop(size),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, help='File path to the content image')
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images',
                    default=r'D:\Users\luyang\Desktop\最终选定图片\0000000\content')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style images separated by commas')
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images',
                    default=r'D:\Users\luyang\Desktop\最终选定图片\0000000\style')
parser.add_argument('--output_c', type=str, default=r'D:\Users\luyang\Desktop\最终选定图片\0000000\content_format', help='Directory to save the output image(s)')
parser.add_argument('--output_s', type=str, default=r'D:\Users\luyang\Desktop\最终选定图片\0000000\style_format', help='Directory to save the output image(s)')
parser.add_argument('--size', type=int, default=512, help='Size of the output images')
args = parser.parse_args()

output_size = args.size
output_path_c = args.output_c
output_path_s = args.output_s

if not os.path.exists(output_path_c):
    os.mkdir(output_path_c)
if not os.path.exists(output_path_s):
    os.mkdir(output_path_s)

image_transform = transform_image(output_size)

if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

if args.style:
    style_paths = args.style.split(',')
    style_paths = [Path(p) for p in style_paths]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

for content_path in content_paths:
    content = image_transform(Image.open(content_path).convert("RGB"))
    content_name = '{:s}/{:s}.jpg'.format(output_path_c, content_path.stem)
    save_image(content, content_name)

for style_path in style_paths:
    style = image_transform(Image.open(style_path).convert("RGB"))
    style_name = '{:s}/{:s}.jpg'.format(output_path_s, style_path.stem)
    save_image(style, style_name)
