import numpy as np
import pandas as pd

input_data = pd.read_csv('submission.csv', encoding='utf-8', index_col=['id'])


output_data = input_data.applymap(lambda x: np.around(x, 2))

output_data.to_csv('submission_rounded.csv', header=True, index=True, encoding='utf-8')
