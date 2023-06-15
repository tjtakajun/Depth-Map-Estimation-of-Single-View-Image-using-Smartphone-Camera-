import argparse
from skimage import color
from skimage.io import imread, imsave
from skimage.color import rgb2gray

from inpainter import Inpainter
from Median import Median


def main():
    args = parse_args()

    print(args.model_image)
    model_image = imread(args.model_image)
    model_depth = rgb2gray(imread(args.model_depth))
    image = imread(args.image)
    patch_size = args.patch_size

    output_image = Inpainter(
        model_image,
        model_depth,
        image,
        patch_size,
        plot_progress=args.plot_progress
    ).inpaint()

    output_image = Median(output_image).median()

    imsave(args.output, output_image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch_size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='result/output9 random third.png'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        'model_image',
        help='the model image'
    )
    parser.add_argument(
        'model_depth',
        help='the model depth of model image'
    )
    parser.add_argument(
        'image',
        help='the image you want to make depth image with'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
