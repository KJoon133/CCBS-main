import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def visualize_heatmap(heatmap_arrray, fig_title):
    h_shape = heatmap_arrray.shape
    heatmap_arrray = heatmap_arrray.reshape(h_shape[0], h_shape[2], h_shape[3])
    
    fig, axs = plt.subplots(4, 5, figsize=(21, 26))
    for i, (ax, heatmap) in enumerate(zip(axs.flat, heatmap_arrray)):
        # if np.count_nonzero(heatmap) > 0:
        #     print(heatmap)
        ax.pcolor(heatmap[::], cmap='gray')
        ax.set_title("{}th agent heatmap".format(i))
    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout()

    plt.show()

def compare_heatmap(heatmap_pred, target, fig_title):
    nonzero_cnt = 0

    for target_ in target:
        if np.count_nonzero(target_) > 0:
            nonzero_cnt += 1

    fig, axs = plt.subplots(4, nonzero_cnt%4, figsize=(21, 26))
    for i, (ax, target_, pred_) in enumerate(zip(axs.flat, target, heatmap_pred)):
            fig, (sub_ax1, sub_ax2) = ax.subplots(1,2)
            sub_ax1.pcolor(target_[::], cmap='gray')
            sub_ax1.set_title("{}th agent target".format(i))
            sub_ax2.pcolor(pred_[::], cmap='gray')
            sub_ax2.set_title("{}th agent pred".format(i))

    plt.show()

def main():
    file_path = "/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_68_heatmap/fundus/*"
    heatmap_path_list = glob.glob(file_path)

    for path in heatmap_path_list:
        heatmap_array = np.load(path)
        compare_heatmap(None, heatmap_array)
        break
        # visualize_heatmap(heatmap_array)

if __name__=="__main__":
    main()