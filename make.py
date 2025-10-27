# !/usr/bin/env python3

import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_cell(args):
    row, col, im_array, split_x, split_y, variance, circle_variance, cell_w, cell_h = args
    y_start = row * split_y
    x_start = col * split_x
    region = im_array[y_start:y_start + split_y, x_start:x_start + split_x]
    
    avg = region.mean(axis=(0, 1))
    avg = np.clip(np.random.normal(avg, np.sqrt(variance)), 0, 255).astype(np.uint8)
    
    result = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255
    y, x = np.ogrid[:cell_h, :cell_w]
    center_y, center_x = cell_h / 2, cell_w / 2
    radius = min(cell_w, cell_h) / 2
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    noise = np.random.normal(0, np.sqrt(circle_variance), (cell_h, cell_w, 3))
    result[mask] = np.clip(avg + noise[mask], 0, 255).astype(np.uint8)
    
    return (row, col, result)

if __name__ == "__main__":
    # Usage: python make.py <image> <split> <variance> <circle_variance> [output_width]
    im = Image.open(sys.argv[1])
    split = int(sys.argv[2])
    variance = float(sys.argv[3])
    circle_variance = float(sys.argv[4])
    
    split_x = im.size[0] // split
    split_y = im.size[1] // split
    
    if len(sys.argv) > 5:
        output_width = int(sys.argv[5])
        cell_w = output_width // split
        cell_h = int(cell_w * split_y / split_x)
    else:
        cell_w = split_x
        cell_h = split_y

    im_array = np.array(im)
    args_list = [(row, col, im_array, split_x, split_y, variance, circle_variance, cell_w, cell_h)
                 for row in range(split) for col in range(split)]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_cell, args_list), total=len(args_list)))

    result_array = np.ones((split * cell_h, split * cell_w, 3), dtype=np.uint8) * 255
    for row, col, region in results:
        y = row * cell_h
        x = col * cell_w
        result_array[y:y + cell_h, x:x + cell_w] = region

    im = Image.fromarray(result_array)
    im.save("dot_" + sys.argv[1])