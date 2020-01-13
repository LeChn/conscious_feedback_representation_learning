import os
import random
import shutil
from glob import glob
train_path = "/media/ubuntu/data/RSNApng"
files = os.listdir(train_path)
random.shuffle(files)
split = int(len(files)*0.75)
train_files = files[:split]
val_files = files[split:]

def move_files(files, move_path):
    for file in files:
        ori_path = os.path.join(train_path, file)
        move_to_path = os.path.join(move_path, file)
        shutil.move(ori_path, move_to_path)

if __name__ == '__main__':
    valp = "/media/ubuntu/data/RSNAVAL"
    trainp = "/media/ubuntu/data/RSNATR"
    move_files(train_files, trainp)
    move_files(val_files, valp)
