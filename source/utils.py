import os, glob, shutil, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def save_dict2pkl(path, dictionary):

    with open(path,'wb') as fw:
        pickle.dump(dictionary, fw)

def load_dict2pkl(path):

    with open(path, 'rb') as fr:
        dictionary = pickle.load(fr)

    return dictionary

def min_max_norm(x):

    return (x - x.min() + (1e-30)) / (x.max() - x.min() + (1e-30))

def plot_comparison(list_img, list_name, cmap='gray', savepath=""):

    num_img = len(list_img)

    plt.figure(figsize=(num_img*2, 2), dpi=100)

    for idx_img, _ in enumerate(list_img):

        plt.subplot(1, num_img, idx_img+1)
        plt.title(list_name[idx_img])
        plt.imshow(list_img[idx_img], cmap=cmap)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.close()
