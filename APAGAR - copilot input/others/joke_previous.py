#### JOKE IS THE ALGORITHM PYTHON FILE


    ### All the libraries to import

import pyodbc 
import pandas as pd
from pandas import DataFrame
import mpmath
import numpy as np
from sympy import sympify
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.preprocessing import scale 
from sklearn.linear_model import Ridge, RidgeCV
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV

try:

    ### Try to connect to the database 

    server = '****' 
    database = '****' 
    username = '****' 
    password = '****'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()


    

    ### Select everything from the table costsim_parts of the database

    cursor.execute("SELECT * FROM costsim_parts;") 
    sql = "SELECT * FROM costsim_parts;"
    cursor.execute(sql)
    print("Sélectionner dans les lignes dans la table costsim")
    res = cursor.fetchall()
    table = []
    for row in res:
        table.append(row)
    df = pd.DataFrame.from_records(table, columns=['index', 'part_number', 'project_name', 'part_name', 'raw_material_designation', 'color', 'extra_material', 'production_place', 'percentage_of_extra_material', 'raw_material_price', 'part_volume', 'part_weight', 'part_length', 'part_width', 'part_height', 'avrg_thickness', 'max_thickness', 'annual_target_quantity', 'part_price_amortizated','label'])




    ### Apply the logarithm function to the numerical data and updating the label column

    df['logannual_quantity'] = np.log(df['annual_target_quantity'])
    df["logmaterial_price"] = np.log(df['raw_material_price'])
    df["logvolume"] = np.log(df['part_volume'])
    df["logweight"] = np.log(df['part_weight'])
    df["loglength"] = np.log(df['part_length'])
    df["logwidth"] = np.log(df['part_width'])
    df["logheight"] = np.log(df['part_height'])
    df["logthickness"] = np.log(df['avrg_thickness'])
    df["logthicknessmax"] = np.log(df['max_thickness'])
    df["logamortizated"] = np.log(df['part_price_amortizated'])

    variable = df.loc[:,["production_place"]]
    df.loc[df['production_place'].isin(['Portugal', 'Poland', 'Hungary', 'Czech Republic', 'Bulgaria', 'Turkey', 'Tunisia']), ('label')]='New_Economy_Europe'
    df.loc[df['production_place'].isin(['France', 'Germany', 'Spain', 'Italy', 'United Kingdom']), ('label')]='Mature_Economy_Europe'
    df.loc[df['production_place'].isin(['Singapore', 'Indonesia', 'Thailand', 'Malaysia', 'Vietnam', 'Philippines']), ('label')]='Asia'
    df.loc[df['production_place'].isin(['INDIA','India']), ('label')]='India'
    df.loc[df['production_place'].isin(['CHINA','China']), ('label')]='China'
    df.loc[df['production_place'].isin(['MEXICO','Mexico']), ('label')]='Mexico'
    label_to_list = df['label'].tolist()
    id_to_list = df['index'].tolist()
    value = []
        
    for idx in range(len(id_to_list)):
        value.append((label_to_list[idx],id_to_list[idx]))
    sql_update_query = """UPDATE costsim_parts SET label = ? where id = ?;"""
    cursor.executemany(sql_update_query,value)
    cnxn.commit()
    count = cursor.rowcount
    print(count, "Enregistrements mis à jour avec succès")




    ### Creation of the correlation function : 
    # Compare correlation between variables and drop those with the highest among others and the lowest to log_price_amortizated
    # Compare p-values after a linear regression and drop variables with p-value greater than 0.1
    # The function give the column to keep for the estimation cost of the part


    log_variables=['logannual_quantity', 'logmaterial_price', 'logvolume', 'logweight',
       'loglength', 'logwidth', 'logheight', 'logthickness', 'logthicknessmax',
        'logamortizated']

    def correlation(query,col):
        df_query=df.loc[df['label']==query]
        for row in col:
            for column in col:
                correlation_df = df_query.corr()
                correlation_row_column = correlation_df.loc[row,column]
                if correlation_row_column >= 0.8 and correlation_row_column < 1:
                    if correlation_df.loc[row,'logamortizated'] >= correlation_df.loc[column,'logamortizated']:
                        del col[col.index(column)]
  
                    else:
                        del col[col.index(row)]
                        break
                    
        y = df_query.loc[:,['logamortizated']]
        X = df_query.loc[:,col]
        for column in X.columns:
            model = sm.OLS(y,X).fit()
            p_value = model.pvalues[column]
            if p_value > 0.1:    
                if correlation_df.loc[column, 'logamortizated'] < 0.7:
                    X.drop(column, axis=1, inplace=True)
                    del col[col.index(column)]
                    
        return col
    



    ### Creation of the algorithm function:
    # Linear regression with L2 penalization (ridge regression) and Gradient Boost for linear and tree (XGBoost) are the three developped algorithms in this function
    # Gridsearch is executed in order to select the best parameters for each execution of the function regarding the training data
    # The alpha or lambda parameter (L1 & L2 penalization) could be search into 10**np.linspace(10,-10,1000)*0.05 list. Since there are a lot of numbers to test, it take too long to execute, thus [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7] is temporary selected.
    # After selection of parameters and comparing the test data R² score, the best algorithm is kept for the cost estimation and the result of the prediction is returned

    def algorithm(localization,dictionnary_parameters):
        df_query=df.loc[df['label']==localization.get('label')]
        split_test_train = train_test_split(df_query.loc[:,list(dictionnary_parameters.keys())], df_query.loc[:,['logamortizated']], test_size=0.3)
        X_train_df, X_test_df, y_train_df, y_test_df = split_test_train
        values_to_predict = pd.DataFrame([dictionnary_parameters])

        print('values to predict ', values_to_predict)
        print('X train is ', X_train_df)
        print('X test is ', X_test_df)
        print('Y train is ',  y_train_df)
        print('Y test is ',  y_test_df)


        ## Ridge regression algorithm:
        ridge_reg = Ridge(max_iter=10000)
        params_Ridge = {'alpha': [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7] , "fit_intercept": [True, False], "normalize": [True, False]}
        Ridge_GS = GridSearchCV(ridge_reg, cv=2, param_grid=params_Ridge, n_jobs=-1)
        Ridge_GS.fit(X_test_df,y_test_df)
        print(Ridge_GS.best_params_)
        Ridgeregression = Ridge(random_state=3, **Ridge_GS.best_params_)
        Ridgeregression.fit(X_train_df,y_train_df)
        y_ridge_pred=round(sympify(mpmath.exp(Ridgeregression.predict(np.log(values_to_predict))[0][0])),5)
        r_2_ridge_test = Ridgeregression.score(X_test_df,y_test_df)
        print("r² train : ",Ridgeregression.score(X_train_df,y_train_df),
        "r² test : ",Ridgeregression.score(X_test_df,y_test_df))
        print('y_ridge_pred is : ', y_ridge_pred)
        


        ## XGBOOST:
        params_XGB = {'booster':['gblinear','gbtree'],
         'alpha': [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7], 'lambda': [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7,1]}
        xg_reg = XGBRegressor()

        xg_grid=GridSearchCV(xg_reg, param_grid=params_XGB, cv=2, n_jobs=-1)
        xg_grid.fit(X_train_df,y_train_df)
        print(xg_grid.best_params_)
        print(xg_grid.best_score_)

        xgb= XGBRegressor(**xg_grid.best_params_)
        xgmod=xgb.fit(X_train_df,y_train_df)
        number_xg = xgmod.predict(values_to_predict)[0]
        y_xg_pred=round(sympify(mpmath.exp(number_xg)),5)
        r_2_xg_test = xgb.score(X_test_df,y_test_df)
        print('R² train = ', xgb.score(X_train_df,y_train_df))
        print('R² test = ', xgb.score(X_test_df,y_test_df))
        print('y_xg_pred is ', y_xg_pred)

        ## Compare the quality prediction of both algorithm:

        if r_2_ridge_test >= r_2_xg_test:
            result_prediction = y_ridge_pred
        else: 
            result_prediction = y_xg_pred


        return result_prediction



    cursor.close()
    cnxn.close()
    print("La connexion SQL Server est fermée")

        ### If try failed, check the error to see what is wrong

except (Exception, pyodbc.Error) as error:
    print ("Erreur lors de la connexion à SQL Server", error)



    
    
