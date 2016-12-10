from PIL import Image, ImageChops, ImageOps
import os
import shutil
import pandas as pd
import numpy as np
import shutil
from sklearn import preprocessing


def extract(train_data):
    train = pd.read_csv(train_data, index_col=['id'])
    mapping = {k: v for k, v in enumerate(pd.get_dummies(train['species']).columns)}
    dummies = pd.get_dummies(train['species'])
    dummies.columns = mapping.keys()
    pid_label = dict(zip(train.index, np.array(dummies)))
    id_name = dict(zip(train.index, train['species']))
    data = train.ix[:, ~train.columns.isin(['species'])]
    return pid_label, id_name, mapping, data


def delete_folders(dirs=['test', 'train', 'validation'], dir_path='leaf/images/'):

    for directory in dirs:
        if os.path.exists(dir_path + directory):
            shutil.rmtree(dir_path + directory)


def pic_resize(f_in, size=(96, 96), pad=True):

    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)

    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) // 2, 0)
        offset_y = max((size[1] - image_size[1]) // 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    return thumb


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """batch iterator"""
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            new_data = np.random.permutation(data)
        else:
            new_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield epoch, batch_num, new_data[start_index:end_index]


def move_classified(test_order, pid_name, ans, mapping, dir_path='leaf/images/'):
    answers = dict()
    for k, i in enumerate(test_order):
        quest = list(ans[k]).index(max(list(ans[k])))
        name = mapping[quest]
        answers[i] = name
        print(i, name)

    for pid in list(pid_name.keys()) + list(answers.keys()):
        try:
            directory = dir_path + 'result/' + answers[pid]
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copyfile(str(dir_path + str(pid) + r'.jpg'), str(directory + '/' + str(pid) + r'.jpg'))
        except KeyError:
            directory = dir_path + 'result/' + pid_name[pid]
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copyfile(str(dir_path + str(pid) + r'.jpg'), str(directory + '/' + str(pid) + r'.jpg'))


def generate_training_set(data, pid_label, std=True):
    """ raw data transformation (Standardisation)"""
    if std:
        data = data.apply(preprocessing.scale, with_mean=False, with_std=True, axis=0)
    margins = data.ix[:, data.columns.str.startswith('margin')]
    shapes = data.ix[:, data.columns.str.startswith('shape')]
    textures = data.ix[:, data.columns.str.startswith('texture')]

    input_data = dict()
    if pid_label:
        for i in data.index:
            input_data[i] = (np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(3, 64), pid_label[i])
    else:
        for i in data.index:
            input_data[i] = np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(3, 64)
    return input_data
