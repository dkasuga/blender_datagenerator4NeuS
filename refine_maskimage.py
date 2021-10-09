# You can't make mask image directly on Blender, so
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mask_dir")
    parser.add_argument("output_mask_dir")

    args = parser.parse_args()

    input_mask_dir = args.input_mask_dir
    output_mask_dir = args.output_mask_dir

    if not os.path.exists(output_mask_dir):
        os.mkdir(output_mask_dir)

    mask_path_list = sorted(glob.glob(os.path.join(input_mask_dir, "*.png")))

    for idx, mask_path in enumerate(mask_path_list):
        mask_rgba = mpimg.imread(mask_path)
        alpha = mask_rgba[..., 3]
        mask = np.where(alpha > 0.5, 255.0, 0.0)
        mask_rgb = np.stack([mask, mask, mask], axis=2)
        pilImg = Image.fromarray(np.uint8(mask_rgb))
        output_mask_path = os.path.join(output_mask_dir, "{}.png".format(str(idx).zfill(3)))
        pilImg.save(output_mask_path)


if __name__ == "__main__":
    main()
