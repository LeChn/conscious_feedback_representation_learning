import numpy as np


features_path = "F:/MICCAI 2020/MoCo/tSNE/all_features_un.npy"
labels_path = "F:/MICCAI 2020/MoCo/tSNE/all_targets_un.npy"
new_labels_path = "./new_labels_un_new.npy"

features = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
print(features[0].shape)
print(labels[0].shape)
print(labels[0])
print(labels.shape)
def rewrite_labels(labels):
    new_labels = []
    for i in range(labels.shape[0]):
        if labels[i][-1] == 1.:
            labels_single = labels[i][:-1]
            print(labels_single)
            k = 0
            for i in range(labels_single.shape[0]):
                if labels_single[i] == 1.:
                    k += 1

                    #label = i
                    #print(label)
            if k > 1:
                label = None
            else:
                for i in range(labels_single.shape[0]):
                    if labels_single[i] == 1.:
                        label = i
        else:
            label = 0
        new_labels.append(label)
    return np.array(new_labels)
new_labels = rewrite_labels(labels[0])
print(new_labels.shape)
all_labels = []
for j in range(labels.shape[0]):
    print("start i", j)
    labels_j = rewrite_labels(labels[j])
    all_labels.append(labels_j)
    print(labels_j.shape)
    print("done")
np.save(new_labels_path, all_labels)
new_l = np.load(new_labels_path, allow_pickle=True)
print(new_l.shape)
