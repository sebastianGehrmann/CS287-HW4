import numpy as np
import pandas as pd
import os
import h5py

myFile = h5py.File("predLarge.hdf5", 'r')
data = myFile['preds'][:]
df = pd.DataFrame(data)
#print(df.shape)
df.columns = ["Count"]
ids = [i+1 for i in range(len(data))]
df.insert(0, 'ID',ids)
df.to_csv('predictions.csv', index=False)