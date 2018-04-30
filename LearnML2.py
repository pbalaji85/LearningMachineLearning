import pandas as pd
import numpy as np

X_train = pd.read_table('train/X_train.txt', delim_whitespace = True, header = None )
y_train = pd.read_table('train/y_train.txt', delim_whitespace = True, header = None)

## Create column names for the training dataset
names = pd.Series(data=["feat" + str(i) for i in range(1,(len(X_train.columns)+1))])

X_train.columns = names

y_train.columns = "Target"
