import os


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = data[i]
        s = s + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()


if __name__ == '__main__':
    from random import shuffle
    import random
    percent = 1
    path = "/DATA2/Data/RSNA/RSNAFTR"
    files = os.listdir(path)
    shuffle(files)
    files = files[:int(percent*len(files))]
    train_txt = "../../experiments_configure/train" + \
        str(int(percent*100)) + "F.txt"
    text_save(train_txt, files)

    path = "/DATA2/Data/RSNA/RSNAFVAL"
    files = os.listdir(path)
    shuffle(files)
    # files = files[:int(percent*len(files))]
    val_txt = "../../experiments_configure/valF.txt"
    text_save(val_txt, files)
