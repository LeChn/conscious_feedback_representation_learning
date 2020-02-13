# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from sklearn import datasets
import pdb
from sklearn.manifold import TSNE

features_path = "./all_features_un.npy"
labels_path = "./all_labels_un.npy"
low = 0
high = 100

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def get_rsna_data():
    all_features = []
    all_labels = []
    features = np.load(features_path, allow_pickle=True)[low:high]
    labels = np.load(labels_path, allow_pickle=True)[low:high]
    print(labels.shape)
    for i in range(features.shape[0]):
        print(labels[i].shape)
        all_features.append(features[i])
        all_labels.append(labels[i])
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_features = np.reshape(all_features, (-1, 128))
    all_labels = np.reshape(all_labels, (-1, ))
    all_selected_features = np.zeros((1, 128))
    all_selected_labels = np.zeros((1, ))
    for num in range(all_labels.shape[0]):
        if all_labels[num] != None:
            a = np.ones((1, ))
            a[0] = all_labels[num]
            all_selected_labels = np.concatenate((all_selected_labels, a))
            all_selected_features = np.concatenate((all_selected_features, all_features[num][np.newaxis, :]))
    return all_selected_features[1:, :], all_selected_labels[1:], all_selected_features.shape[0], all_selected_features.shape[1]


def get_rsna_model():
    all_features = []
    all_labels = []
    features = np.load(features_path, allow_pickle=True)[low:high]
    labels = np.load(labels_path, allow_pickle=True)[low:high]
    print(labels.shape)
    for batch in features:
        all_features += batch.tolist()
    for batch in labels:
        all_labels += batch.tolist()
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_features = np.reshape(all_features, (-1, 128))
    all_labels = np.reshape(all_labels, (-1, 6))
    return all_features, np.argmax(all_labels,axis= 1), all_labels.shape[0], all_labels.shape[1]

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data[:,0], columns = ['x']) 
    df['y'] = data[:,1]
    df['label'] = label
    num_Classes = len(Counter(label).keys())
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(int(label[i])),
    #              color=plt.cm.Set1(label[i]),
    #              fontdict={'weight': 'bold', 'size': 9})

    fig = plt.figure(figsize=(16,10))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    print("The number of classes is", num_Classes)
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        style="label",
        # cmap=sns.cubehelix_palette(light=1, as_cmap=True),
        palette=sns.color_palette("hls", num_Classes),
        data=df,
        legend="full",
        alpha=0.8
    )
    return fig


def main():
    data, label, n_samples, n_features = get_rsna_model()
    print(data.shape, label.shape)
    #print(label[:30])
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    t0 = time()
    # result = tsne.fit_transform(data)
    # np.save("x.npy", result)
    result = np.load("x.npy")
    # tsne = TSNE(n_components=1, perplexity=5, random_state=0, n_iter=2000)
    # label = tsne.fit_transform(label)
    # print(result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the RSNA data (time %.2fs)'
                         % (time() - t0))
    plt.savefig("result.pdf")
    plt.show(fig)


if __name__ == '__main__':
    #a, b, c, d = get_rsna_data()
    #print(a.shape, b.shape)
    main()