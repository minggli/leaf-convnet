import numpy as np
import pandas as pd

data = pd.read_csv('submission.csv', encoding='utf-8', index_col='id')

data = data.applymap(lambda x: np.around(x, decimals=1))

data.to_csv('submission.csv', encoding='utf-8', index='id')

