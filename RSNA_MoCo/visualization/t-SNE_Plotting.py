# coding='utf-8'
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
high = 700
class_Name = ['epidural', 'intraparenchymal',
              'intraventricular', 'subarachnoid', 'subdural', 'any']


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
            all_selected_features = np.concatenate(
                (all_selected_features, all_features[num][np.newaxis, :]))
    return all_selected_features[1:, :], all_selected_labels[1:], all_selected_features.shape[0], all_selected_features.shape[1]


def get_rsna_model():
    all_features = []
    all_labels = []
    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    print(labels.shape)
    for batch in features:
        all_features += batch.tolist()
    for batch in labels:
        all_labels += batch.tolist()
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_features = np.reshape(all_features, (-1, 128))
    all_labels = np.reshape(all_labels, (-1, 6))
    return all_features, np.argmax(all_labels, axis=1), all_labels.shape[0], all_labels.shape[1]


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data[:, 0], columns=['Dimension 1'])
    df['Dimension 2'] = data[:, 1]
    counts = Counter(label)
    df['Label'] = [class_Name[i] +
                   " (" + str(counts[i]) + ")" for i in label.tolist()]
    num_Classes = len(counts.keys())
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(int(label[i])),
    #              color=plt.cm.Set1(label[i]),
    #              fontdict={'weight': 'bold', 'size': 9})

    fig = plt.figure(figsize=(16, 10))
    plt.title(title)
    print("The number of classes is", num_Classes)
    print(Counter(label))
    g = sns.scatterplot(
        x='Dimension 1', y='Dimension 2',
        hue='Label',
        style="Label",
        palette=sns.color_palette("hls", num_Classes),
        data=df,
        legend="full",
        alpha=0.8
    )

    g.legend(loc='upper right')
    return fig


def removeLargestClass(data, label):
    rem = [i != 0 for i in label]
    data = data[rem]
    label = label[rem]
    return data, label


def main():
    data, label, n_samples, n_features = get_rsna_model()
    print(data.shape, label.shape)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    t0 = time()
    result = tsne.fit_transform(data)
    np.save("x.npy", result)
    result = np.load("x.npy")
    fig = plot_embedding(result, label,
                         't-SNE embedding of the last layer of encoder of MoCo with 100% labeled data fine-tuning')
    # t-SNE embedding of the last layer of encoder of MoCo with 5% labeled data fine-tuning
    plt.savefig("MoCo_supervised_100percentage_data.pdf")

    data, label = removeLargestClass(data, label)
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the last layer of encoder of MoCo with 100% labeled data fine-tuning, underrepresented')
    plt.savefig("MoCo_supervised_100percentage_data_underrepresented.pdf")

    plt.show(fig)


if __name__ == '__main__':
    main()
