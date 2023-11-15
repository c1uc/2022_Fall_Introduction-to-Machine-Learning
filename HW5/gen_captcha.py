import os
import csv
import itertools
from captcha.image import ImageCaptcha
import shutil
from dataset import TaskDataset
import numpy as np

ALL_CHARS = TaskDataset.CHARS
GEN_PATH = './gen'
char_len = [0, 1, 2, 4]
gen_counts = [0, 10000, 80000, 100000]
img_size = [(0, 0), (72, 72), (72, 72), (96, 72)]
rng = np.random.default_rng()
res = []


def _gen_captcha(img_dir, num_per_image, num_images, width, height):
    if os.path.exists(f"{GEN_PATH}/{img_dir}"):
        shutil.rmtree(f"{GEN_PATH}/{img_dir}")
    if not os.path.exists(f"{GEN_PATH}/{img_dir}"):
        os.mkdir(f"{GEN_PATH}/{img_dir}")

    image = ImageCaptcha(width=width, height=height, font_sizes=(42, 50, 56))
    perms = list(itertools.product(ALL_CHARS, repeat=num_per_image))
    count = 0

    sam = rng.choice(perms, size=num_images, replace=True)
    for _ in sam:
        label = ''.join(_)
        fn = os.path.join(f"{GEN_PATH}/{img_dir}", f'{label}_{count}.png')
        image.write(label, fn)
        res.append([f"{img_dir}/{label}_{count}.png", label])
        count += 1


if __name__ == '__main__':
    for i in [1, 2, 3]:
        print(f"generating task{i} dataset with {gen_counts[i]} images...")
        _gen_captcha(f"task{i}", char_len[i], gen_counts[i], img_size[i][0], img_size[i][1])
    with open(f"{GEN_PATH}/annotations.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(res)
    print("done.")
