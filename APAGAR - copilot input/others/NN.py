import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras import Input, Model
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from sklearn.model_selection import train_test_split
import os


maindf = pd.read_csv("./costsim/src/masoud/out22.csv" , encoding='windows-1252')
columnNames = ['annual_target_quantity', 'max_thickness','raw_material_price', 'part_volume', 'part_weight','part_width','part_height', 'part_length','avrg_thickness']
maindf['QRV']=np.log10(np.log10(maindf['annual_target_quantity']))/maindf['raw_material_price']/maindf['part_volume']*3
print(len(maindf[maindf['label']=="New_Economy_Europe"]),
len(maindf[maindf['label']=="India"]),
len(maindf[maindf['label']=="Asia"]),
len(maindf[maindf['label']=="China"]),
len(maindf[maindf['label']=="Mexico"]),
len(maindf[maindf['label']=="Asia"]))


def algorithmNN(value_to_predict,label, retrainer=False):
  global maindf
  df=maindf[maindf['label']==label]
  if label!="China":
    batchSize = len(df)
  else:
    batchSize = len(df)//4

  if not os.path.isdir("./costsim/src/masoud/mymodel"+label) or retrainer==True:
    maindf = pd.read_csv("./costsim/src/masoud/out22.csv" , encoding='windows-1252')
    columnNames = ['annual_target_quantity', 'max_thickness','raw_material_price', 'part_volume', 'part_weight','part_width','part_height', 'part_length','avrg_thickness']
    maindf['QRV']=np.log10(np.log10(maindf['annual_target_quantity']))/maindf['raw_material_price']/maindf['part_volume']*3
    # print(len(maindf[maindf['label']=="New_Economy_Europe"]),
    # len(maindf[maindf['label']=="India"]),
    # len(maindf[maindf['label']=="Asia"]),
    # len(maindf[maindf['label']=="China"]),
    # len(maindf[maindf['label']=="Mexico"]),
    # len(maindf[maindf['label']=="Asia"]))
    # print("model: mymodel"+label+" is not found or retrain request is received")
    ### saving model Phase 
    ### load data
    X=df.loc[:,columnNames]
    Y=df.loc[:,'part_price_amortizated']
    transformation='log'
    if transformation=="log":
        Y=np.log10(df.loc[:,'part_price_amortizated'])
        X=np.log10(df.loc[:,columnNames])

    data_size=len(X.columns)

    # define encoder
    visibleLayer = Input(shape=(data_size,))
    e2 = Dense(data_size*4)(visibleLayer)
    e2 = BatchNormalization()(e2)
    e2 = ReLU()(e2)
    # define bottleneck
    d2 = Dense(data_size*6)(e2)

    # output layer
    output2 = Dense(1, activation='linear')(d2)
    # define autoencoder model
    nnModel = Model(inputs=visibleLayer, outputs=output2)
    # compile autoencoder model
    nnModel.compile(optimizer='adam', loss='mean_absolute_percentage_error') #mean_absolute_percentage_error

    ### Test model
    flag=0
    for jj in range(0,len(X)):
      nnModel.fit(X, Y, epochs=1, batch_size=batchSize, verbose=False, validation_data=(X,Y))
      ###   Run candidated model
      # print(X,Y, nnModel.predict(X))
      
      
      ### if you have test file
      # Testdf = pd.read_csv("Test2.csv" , encoding='windows-1252')
      # Testdf['QRV']=np.log10(np.log10(Testdf['annual_target_quantity']))/Testdf['raw_material_price']/Testdf['part_volume']*3
      # Testdf['colorWeight']=df.groupby('color')['part_price_amortizated'].mean()[Testdf['color']].values
      # XTestdf=Testdf.loc[:,columnNames]
      # # XTestdf.loc[:,columnNames[:-5]]=np.log10(Testdf.loc[:,columnNames[:-5]])
      # XTestdf.loc[:,columnNames]=np.log10(Testdf.loc[:,columnNames])
      # YTestdf=np.log10(Testdf.loc[:,'part_price_amortizated'])
      # mean_absolute_percentage_error(10**YTestdf,10**regr.predict(XTestdf, verbose=False))
      # return
      
      regr = nnModel
      numberOfTests=0;avgerror=0;errArr=[];fitscore=0
      for k in range(0,len(X)//6):
        numberOfTests+=1
        split_test_train = train_test_split(X,Y, test_size=0.1)
        X_train_df, dfTets, y_train_df, dfTets2 = split_test_train

        if transformation=="log":
            fitscore+=mean_absolute_percentage_error(10**y_train_df,10**regr.predict(X_train_df, verbose=False))
            avgerror+=mean_absolute_percentage_error(10**dfTets2,10**regr.predict(dfTets.loc[:,columnNames], verbose=False))
            errArr.append(mean_absolute_percentage_error(10**dfTets2,10**regr.predict(dfTets.loc[:,columnNames], verbose=False)))
            # print(mean_absolute_percentage_error(10**dfTets2,10**regr.predict(dfTets.loc[:,columnNames], verbose=False)))
        else:
            fitscore+=mean_absolute_percentage_error(y_train_df,regr.predict(X_train_df, verbose=False))
            avgerror+=mean_absolute_percentage_error(dfTets2,regr.predict(dfTets.loc[:,columnNames], verbose=False))
            errArr.append(mean_absolute_percentage_error(dfTets2,regr.predict(dfTets.loc[:,columnNames], verbose=False)))

      # print("Average error: ", avgerror/numberOfTests,"Average fitted: ", fitscore/numberOfTests)
      if avgerror/numberOfTests <0.23:
        # nnModel.save('./costsim/src/masoud/mymodel'+label)
        # print("new model saved")
        flag=1
        break
      
    if flag==0:
      # print("something is wrong, model is not fitted")
      return
    
  # if retrainer==True:
  #   return 0
  regr = keras.models.load_model('./costsim/src/masoud/mymodel'+label)
  # print(10**regr.predict(np.log10(value_to_predict), verbose=False))
  return 10**regr.predict(np.log10(value_to_predict), verbose=False)[0]

# values_to_predict=pd.DataFrame([{'annual_target_quantity': 1, 'max_thickness': 2, 'raw_material_price': 2, 'part_volume': 3, 'part_weight': 3, 'part_width': 3, 'part_height': 1, 'part_length': 4, 'avrg_thickness': 4}])
# print(algorithmNN(values_to_predict,"China"))
# print(algorithmNN(0,"China", True))