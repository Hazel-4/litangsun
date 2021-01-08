import numpy as np
import pandas as pd

txt = np.loadtxt('file.txt')
txtDF = pd.DataFrame(txt)
txtDF.to_csv('file.csv', index=False)