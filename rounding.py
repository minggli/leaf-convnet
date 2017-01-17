import numpy as np
import pandas as pd
from utilities import delete_folders, extract, pic_resize, batch_iter, transform, move_classified, generate_training_set


data_raw = pd.read_csv('0.01371.csv', encoding='utf-8', index_col='id')

data_raw = data_raw.applymap(lambda x: np.around(x, decimals=4))


def submit(raw):

    delete_folders()

    move_classified(test_data=raw, train_data=data, columns=label.columns, index=test.index, path=IMAGE_PATH)

    df = pd.DataFrame(data=raw, columns=label.columns, index=test.index)
    df.to_csv('submission.csv', encoding='utf-8', header=True, index=True)

MODEL_PATH = 'models/'
IMAGE_PATH = 'leaf/images/'
INPUT_PATH = 'leaf/'

train, label, data = extract(INPUT_PATH + 'train.csv', target='species')
_, _, test = extract(INPUT_PATH + 'test.csv')

submit(data_raw)
