import os
import shutil
from pandas import DataFrame

__author__ = 'Ming Li'


# allocating labels into seperate folders

def copy_pics_into_folders(mapping_dict, path='leaf/images/'):

    assert isinstance(mapping_dict, DataFrame), 'require a DataFrame'

    for k, v in mapping_dict.iterrows():

        leaf_id = str(int(v.values))
        full_path = path + k + '/'
        file_name = leaf_id + '.jpg'

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        shutil.copy(path + file_name, full_path)


def delete_folders(mapping_dict, path='leaf/images/'):

    assert isinstance(mapping_dict, DataFrame), 'require a DataFrame'

    for k, v in mapping_dict.iterrows():

        full_path = path + k + '/'

        if os.path.exists(full_path):
            shutil.rmtree(full_path)

