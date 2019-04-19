import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N, 5, 32, 32)
    :return: numpy array (N, 5, 2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def merge_and_scale_heatmap(heatmap, scale):
    merged = np.mean(heatmap, axis=0)
    h, w = merged.shape
    scaled = cv2.resize(merged, dsize=(h * scale, w * scale), interpolation=cv2.INTER_LINEAR)
    return scaled

def plot_heatmap(heatmap, img, mean, scale=4, alpha=0.5):
    '''
    merge heatmaps of different points into one heatmap
    :param heatmap: numpy array (5, 32, 32)
    :param img: image array (3, 128, 128)
    :param mean: mean rgb of dataset
    :param scale: scale factor
    :param alpha: float alpha
    '''
    scaled = merge_and_scale_heatmap(heatmap, scale)
    img_s = np.transpose(np.clip(img + mean, 0, 1), (1, 2, 0))
    fig_withhm = plt.figure()
    plt.imshow(img_s)
    plt.imshow(scaled, cmap='hot', alpha=alpha)
    plt.axis('off')
    return fig_withhm

def plot_heatmap_compare(heatmaps, heatmap_gt, img, mean, scale=4, alpha=0.5):
    '''
    merge heatmaps of different points into one heatmap
    :param heatmaps: list of numpy array [(5, 32, 32)]
    :param heatmap_gt: numpy array (5, 32, 32) ground truth
    :param img: image array (3, 128, 128)
    :param mean: mean rgb of dataset (3, 1, 1)
    :param scale: scale factor
    :param alpha: float alpha
    '''
    scaled = [merge_and_scale_heatmap(x, scale) for x in heatmaps]
    scaled_gt = merge_and_scale_heatmap(heatmap_gt, scale)
    img_s = np.transpose(np.clip(img + mean, 0, 1), (1, 2, 0))
    img_s = np.concatenate([img_s for _ in range(len(heatmaps) + 1)], axis=1)
    scaled.insert(0, scaled_gt)
    
    scaled_s = np.concatenate(scaled, axis=1)
    fig_withhm = plt.figure(figsize=(2*len(scaled), 2))
    plt.imshow(img_s)
    plt.imshow(scaled_s, cmap='hot', alpha=alpha)
    plt.axis('off')
    return fig_withhm