import numpy as np
import pandas as pd

data = pd.read_csv('F:/.python/NLP/Datasets/ROCO/train/data.csv')

print(data.iloc[5, :].name)