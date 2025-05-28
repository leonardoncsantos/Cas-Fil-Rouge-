from sklearn.metrics import mean_absolute_percentage_error
import itertools
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
    
def correlation2(dfQueryFiltered,columnNames,correlationThreshold,y_train_df):
    col = columnNames
    dfQueryFiltered = pd.concat([dfQueryFiltered, y_train_df],axis=1)
    correlation_df = dfQueryFiltered.corr()
    ### Correlation
    for row in col:
        for column in col:
            if column=='part_price_amortizated' or row=='part_price_amortizated':
                continue
            correlationRowColumn = correlation_df.loc[row,column]
            if abs(correlationRowColumn) >= correlationThreshold and abs(correlationRowColumn) < 1:
                if correlation_df.loc[column,'part_price_amortizated'] >= correlation_df.loc[row,'part_price_amortizated']:
                    del col[col.index(row)]   
                else:
                    del col[col.index(column)]
                break
    
    if not('annual_target_quantity' in col):
        col.append('annual_target_quantity')  
    return col
import statsmodels.api as sm
    
def p_value(dfQueryFiltered,col,y_train_df):
    flag=False
    # ### P-Value Testing
    if 'annual_target_quantity' in col:
        del col[col.index('annual_target_quantity')]
        flag=True
    # correlation_df=correlation_df.join(correlationWithTarget)
    y = y_train_df
    X = dfQueryFiltered
    model = sm.OLS(y,X).fit()
    for column in np.copy(col):
        p_value = model.pvalues[column]
        if p_value <0.01:            # if p-value is less than 0.05, it means the column has less effect on y so we can delete it. ( null hyposis that says it has relation with y is refused)
            del col[col.index(column)]
    if flag:
        col.append('annual_target_quantity')  
    
    return col

flag=True
def testPhase(X_train_df,y_train_df, dfTets,dfTets2,trainModel,transformFunction):
    global flag
    if flag:
        flag=False
    
    score=0
    numberXG = trainModel.predict(dfTets)
    numberP = numberXG
    realV = dfTets2.values
    if transformFunction=='log10':
        numberP = 10**(numberXG)
        realV=10**(dfTets2.values)
        score = mean_absolute_percentage_error(10**y_train_df,10**trainModel.predict(X_train_df))
    return abs(realV-numberP)/realV,(1-score)

def featureCreation(dfq,quantity):
    dfq['QRV']=np.log10(np.log10(quantity))/dfq['raw_material_price']/dfq['part_volume']*3
    dfq['annual_target_quantity'] = np.log10(quantity)
    return dfq

arrmin=[];selectedFeatures=[]
testId=-1
### GridSearchCV takes time and just choose optimized parameters according to train dataset while we have overfitting problem in XGBoost, so this is all possible transformation we can do
### result would be saved and would be used when runtime
def algorithmAI(label, values_to_predict, tunningParameters="", algorithm="xgboost"):
    """This method estimate the Cost of the part using database prepared using regression or xgboost.
    
    this file was implemented in this way, instead of using GridSearch in order to tune tranformation functions and other factors as well as hyperparameters of xgb, like using nearest neighbors to train, classification before xgb and (not) using p-value 
    and correlation for each country. So to test all, GridSearch was not used and I kept the code below, not doing the same tests in the future. you can also access to cleaner version of the code in the joke_previous.py file.
    
    ***** in general, if you call algorithmAI(lbl, 1) then the algorithm would iterate in all possiblities to tune hyperparameters (and transformation functions that you might specify). 
    it returns the best way of tunning after bruteforce.
    
    if you call algorithmAI(lbl, 1, tunningParameters=main_parameters[lbl]) it would give the test result on local dataset.
    
    algorithmAI(label, values_to_predict) : test on runtime: 
        final return is always a non-variant error amount for the algorithm on the testdata.
    """
    global arrmin,testId,selectedFeatures,flag
    minerror=100000;possibilities=[]

    if algorithm=='xgboost':  
        ### put bstr first, to increase prunning speed in the lastest for
        bstr=['gbtree'];lrnR=[0.08,0.3];lambdaR=[0.05,0.5,1,5];#lambdaR=[100,50]   
        prixTF= ['log10'];corr=['n','y'];pval=['n','y'];#bstr=['gbtree'];lrnR=[0.3];lambdaR=[100]  
        possibilities=list(itertools.product(pval,corr,prixTF,bstr,lrnR,lambdaR))
    else:
        prixTF= ['no','log10'];corr=['n','y'];pval=['y','n']
        possibilities=list(itertools.product(pval,corr,prixTF))
    df = pd.read_csv("./costsim/src/masoud/out22.csv" , encoding='windows-1252')
    df=df[df['label']==label]
    price=df['part_price_amortizated']
    
    quantity=df.loc[:,'annual_target_quantity']
    skipFlag=tunningParameters!=""
    if skipFlag:
        defaultcolumns=tunningParameters[-1]
        tunningParameters = tunningParameters[:-1]
    df= featureCreation(df,quantity)
    for param in possibilities:
        if skipFlag:    
            if param!=tunningParameters:
                continue
        columnNames = ['annual_target_quantity', 'max_thickness','raw_material_price', 'part_volume', 'part_weight','part_width','part_height', 
        'part_length','avrg_thickness'] #,'madeup2',,'madeup2','QRV','made3', ,'coefweight','QRV'
        
        if skipFlag:
            columnNames = defaultcolumns+columnNames#+list(df_encoded.columns)+['ChinaCol','AsiaCol','MexicoCol','New_Economy_EuropeCol']
            
        df2=df.copy()
        if param[2]=='log10':
            df2['part_price_amortizated'] = np.log10(price)
            
        if param[1]=='y':
            columnNames= correlation2(df2.loc[:,columnNames],columnNames, 0.8,df2.loc[:,'part_price_amortizated'])
        if param[0]=='y':
            columnNames= p_value(df2.loc[:,columnNames],columnNames,df2.loc[:,'part_price_amortizated'])
        
        if algorithm=="reg":
            trainModel=LinearRegression()
        else:
            params={"reg_lambda":param[5],"booster":param[3],"learning_rate":param[4]}#,"min_child_weight":0.1,"gamma":0.06}#,"subsample":lambdaU}
            trainModel=xgb.XGBRegressor(**params) 
            
        if type(values_to_predict)!=int:
            break
        
        avgRealError=0
        errArr=[]
        avgRealError=0
        numberOfTests=0
        averageScore=0

        for someRandomTest in range(0,len(df2)):
            split_test_train = train_test_split(df2.loc[:,columnNames],df2.loc[:,'part_price_amortizated'], test_size=0.1)
            X_train_df, dfTets, y_train_df, dfTets2 = split_test_train
            
            numberOfTests+=1
            trainModel.fit(X_train_df,y_train_df)#,sample_weight=np.delete(sampleWeights,kk))#, sample_weight=myfeatureWeights, feature_weights=[1,0.01,0.01,0.01,0.01,0.01,0.01,500,0.01])
            
            errorPercentageForThisTestSet, scoreFitted = testPhase(X_train_df,y_train_df, dfTets,dfTets2,trainModel,param[2])#,tempWeight)
            errArr.append(errorPercentageForThisTestSet)
            averageErrorPercentageForThisTestSet=np.sum(errorPercentageForThisTestSet/len(errorPercentageForThisTestSet))
            avgRealError+=averageErrorPercentageForThisTestSet      # at the end would be divided by numberOfTests      

        print("fitted: ",averageScore/numberOfTests,"error: ",avgRealError/numberOfTests, param)
        avgRealError = avgRealError/numberOfTests
        if avgRealError<minerror:
            minerror= avgRealError
            arrmin=param
            selectedFeatures=columnNames

    if type(values_to_predict)==int:
        return minerror,arrmin
    
    values_to_predict = featureCreation(values_to_predict,values_to_predict['annual_target_quantity'])
    trainModel.fit(df2.loc[:,columnNames],df2.loc[:,"part_price_amortizated"])
    predicted=trainModel.predict(values_to_predict.loc[:,columnNames])
    if tunningParameters[2]=='log10':
        predicted = 10**(predicted)
    return predicted

main_parameters={
    ### preferly we do corolation
    ### QRV is a created feature and it improves the result normally, although it should be tested on bigger datasets
    "Asia":('n', 'n', 'log10', 'gbtree', 0.08, 0.05, []), 
    "China":('n', 'y', 'log10', 'gbtree', 0.08, 1, ['QRV']),
    "New_Economy_Europe":('n', 'n', 'log10', 'gbtree', 0.08, 1,[]),
    "India":('y', 'n', 'log10', 'gbtree', 0.3, 5, []), 
    "Mexico":('n', 'y', 'log10', 'gbtree', 0.3, 1,['QRV'])
}
lbl="Mexico"

### to find the best tunning, run the command below for each label
# minerror=algorithmAI(lbl, 1)
# minerror=algorithmAI(lbl, 1, tunningParameters=main_parameters[lbl]) #main_parameters[lbl])
# print("error ratio %:",100*minerror,"parameters:",arrmin, "MAV: ",0,"features: ",selectedFeatures)

def algorithmLauncher(values_to_predict, label):
    global main_parameters
    return algorithmAI(label, values_to_predict, tunningParameters=main_parameters[lbl])