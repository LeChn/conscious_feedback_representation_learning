import pandas as pd
import os


'''dir_csv = '/media/ubuntu/data/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'
train = pd.read_csv(dir_csv)

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
#train.set_index(['Image'], inplace=True)
print(train.head())
train.to_csv('train.csv', index=False)'''
csv_train = "/media/ubuntu/data/train.csv"
label_csv = pd.read_csv(csv_train)
label_csv.set_index(['Image'], inplace=True)
print(label_csv.index)
