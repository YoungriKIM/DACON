import numpy as np
import pandas as pd
<<<<<<< HEAD


x = []
for i in range(1,9):
    df = pd.read_csv(f'/content/drive/MyDrive/colab_data/dacon1/dongdong/dong_{i}.csv', index_col=0, header=0)
=======
import tensorflow.keras.backend as K


x = []
for i in range(3,7):
    df = pd.read_csv(f'D:/aidata/dacon12/sub_save/predict_0{i}.csv', index_col=0, header=0)
>>>>>>> 1f8328ff80559623c65cef866ec88d6c22d4a198
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

<<<<<<< HEAD
df = pd.read_csv(f'/content/drive/MyDrive/colab_data/dacon1/dongdong/dong_{i}.csv', index_col=0, header=0)
for i in range(7776):
    for j in range(9):
        a = []
        for k in range(5):
=======
df = pd.read_csv(f'D:/aidata/dacon12/sub_save/predict_0{i}.csv', index_col=0, header=0)
for i in range(5000):
    for j in range(26):
        a = []
        for k in range(4):
>>>>>>> 1f8328ff80559623c65cef866ec88d6c22d4a198
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
<<<<<<< HEAD
y.to_csv('/content/drive/MyDrive/colab_data/dacon1/dongdong/result_3.csv') 
=======
y.to_csv('D:/aidata/dacon12/sub_save/mean_2.csv')  
>>>>>>> 1f8328ff80559623c65cef866ec88d6c22d4a198
