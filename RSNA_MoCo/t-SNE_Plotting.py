# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

features_path = "./all_features.npy"
new_labels_path = "./new_labels_un_new.npy"
low = 80
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





def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_rsna_data()
    print(data.shape, label.shape)
    #print(label[:30])
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    t0 = time()
    result = tsne.fit_transform(data)
    print(result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the RSNA data (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    #a, b, c, d = get_rsna_data()
    #print(a.shape, b.shape)
    main()