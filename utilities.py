from PIL import Image, ImageChops, ImageOps
import os
import pandas as pd
import numpy as np
import shutil
from sklearn import preprocessing


def extract(file, target=None):
    data = pd.read_csv(file, index_col=['id'])
    if target in data.columns:
        label = pd.get_dummies(data[target])
        train = data.drop([target], axis=1)
    else:
        label = None
        train = None
    return train, label, data


def transform(data, label, pixels=None, standardize=True):
    """standard scaling and turning data into 3-dim array for either train or test"""

    if standardize:
        data = data.apply(preprocessing.scale, with_mean=False, with_std=True, axis=0)

    margins = data.ix[:, data.columns.str.startswith('margin')]
    shapes = data.ix[:, data.columns.str.startswith('shape')]
    textures = data.ix[:, data.columns.str.startswith('texture')]

    d = 4 if pixels is not None else 3

    if label is not None and pixels is not None:
        transformed = \
            [(np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :], np.array(pixels[i]).flatten()), axis=0).reshape(d, 64), label.ix[i, :]) for i in data.index]
    if label is not None and pixels is None:
        transformed = \
            [(np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(d, 64), label.ix[i, :]) for i in data.index]
    if label is None and pixels is not None:
        transformed = \
            [np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :], np.array(pixels[i]).flatten()), axis=0).reshape(d, 64) for i in data.index]
    if label is None and pixels is None:
        transformed = \
            [np.concatenate((margins.ix[i, :], shapes.ix[i, :], textures.ix[i, :]), axis=0).reshape(d, 64) for i in data.index]

    return np.array(transformed)


def generate_training_set(data, test_size=.05):

    index = len(data)
    random_index = np.random.permutation(index)

    train_size = int((1 - test_size) * index)

    train_index = random_index[:train_size]
    test_index = random_index[train_size:]

    combined_train = data[train_index]
    combined_valid = data[test_index]

    return combined_train, combined_valid


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
    num_batches_per_epoch = int(data_size/batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            new_data = np.random.permutation(data)
        else:
            new_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(start_index + batch_size, data_size)
            if start_index == end_index:
                break
            else:
                yield epoch, batch_num, new_data[start_index:end_index]


def move_classified(test_data, train_data, columns, index, path='leaf/images/'):
    """moving classified photos together in labeled folder to eye testing"""

    test_df = pd.DataFrame(data=test_data, columns=columns, index=index)
    test_df['species'] = test_df.idxmax(axis=1)

    combined_df = pd.concat([test_df['species'], train_data['species']], axis=0).sort_index()

    for index in combined_df.index:
        pid = int(index)
        directory = path + 'result/' + combined_df[pid]
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copyfile(path + str(pid) + '.jpg', directory + '/' + str(pid) + '.jpg')


def delete_folders(dirs=['test', 'train', 'validation','result'], path='leaf/images/'):

    for directory in dirs:
        if os.path.exists(path + directory):
            shutil.rmtree(path + directory)