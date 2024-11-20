# Scipt inspired by the repo SAM-OCTA

import numpy as np
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
from collections import *
import random
from itertools import *
from functools import *
from tqdm import tqdm

random_seed = 0

if random_seed:  
    random.seed(random_seed)
    np.random.seed(random_seed)

# Converting 2D coordinate points to Gaussian heat maps
def points_to_gaussian_heatmap(centers, height, width, scale): 
    gaussians = []
    for y, x in centers:
        s = np.eye(2) * scale
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)
    x, y = np.arange(0, width), np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    zz = sum(g.pdf(xxyy) for g in gaussians)
    img = zz.reshape((height, width))

    return img / np.max(img)

def get_labelmap(label):
    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, connected_num = ndimage.label(label, structure=structure)
    # Pixel->connected component, 0 is the background
    pixel2connetedId = {(x, y): val for (x, y), val in np.ndenumerate(labelmaps)}
    return labelmaps, connected_num, pixel2connetedId

def get_negative_region(labelmap, neg_range=8):
    kernel = np.ones((neg_range, neg_range), np.uint8)
    negative_region = cv2.dilate(labelmap, kernel, iterations=1) - labelmap
    return negative_region

def label_to_point_prompt_global(label, positive_num=2, negative_num=-1):
    labelmaps, connected_num, _ = get_labelmap(label)
    positive_points, negative_points = [], []
    connected_points_pos, connected_points_neg = defaultdict(list), defaultdict(list)
    negative_region = get_negative_region(labelmaps.astype(np.uint8))

    for (x, y), val in np.ndenumerate(labelmaps): connected_points_pos[val].append((y, x))
    for (x, y), val in np.ndenumerate(negative_region): connected_points_neg[val].append((y, x))
    
    # time consuming loop
    for connected_id in range(1, connected_num+1):
        if positive_num <= len(connected_points_pos[connected_id]):
            positive_points += random.sample(connected_points_pos[connected_id], max(0, positive_num))
        if 0 < negative_num <= len(connected_points_neg[connected_id]): 
            negative_points += random.sample(connected_points_neg[connected_id], max(0, negative_num))

    if negative_num == -1:
        total_num = 30 * positive_num
        negative_num = total_num - connected_num * positive_num
        negative_region = get_negative_region(label)
        negative_points = [(y, x) for (x, y), val in np.ndenumerate(negative_region) if val]
        negative_points = random.sample(negative_points, max(0, negative_num))

    return np.array([label], dtype=float), positive_points, negative_points