import pandas as pd
import numpy as np

path = '/home/raymond/Downloads/data/SICK.txt'

def load_csv(path):
    df_sick = pd.read_csv(path, sep="\t", usecols=[1, 2, 4], names=['s1', 's2', 'score'],
                          dtype={'s1': object, 's2': object, 'score': object})
    df_sick = df_sick.drop([0])
    sources = df_sick.s1.values
    targets = df_sick.s2.values
    scores = np.asarray(map(float, df_sick.score.values), dtype=np.float32)
    return scores, sources, targets
