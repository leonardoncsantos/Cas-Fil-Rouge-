import pandas
import mpmath
#from math import exp
from pandas import DataFrame
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

import pyodbc 
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port

try:
    server = '****' 
    database = '****' 
    username = '****' 
    password = '****'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cur = cnxn.cursor()

    sql = "SELECT * FROM costsim_parts"
    cur.execute(sql)
    print("Sélectionner dans les lignes dans la table costsim")
    res = cur.fetchall()

    table = []

    for row in res:
        table.append(row)
    df = DataFrame(table, columns=['index','part_number', 'project_name', 'part_name', 'raw_material_designation',
       'color', 'extra_material', 'production_place',
       'percentage_of_extra_material', 'raw_material_price', 'part_volume',
       'part_weight', 'part_length', 'part_width', 'part_height',
       'avrg_thickness', 'max_thickness',
       'annual_target_quantity', 'part_price_amortizated','label'])
    #df = df.set_index('part_number', inplace=True)
    
    
    df['logannual_quantity'] = np.log(df['annual_target_quantity'])

    df["logmaterial_price"] = np.log(df['raw_material_price'])

    df["logvolume"] = np.log(df['part_volume'])
    df["logweight"] = np.log(df['part_weight'])
    df["loglength"] = np.log(df['part_length'])
    df["logwidth"] = np.log(df['part_width'])
    df["logheight"] = np.log(df['part_height'])
    df["logthick"] = np.log(df['avrg_thickness'])
    df["logthickmax"] = np.log(df['max_thickness'])
    df["logamortizated"] = np.log(df['part_price_amortizated'])

    variable = df.loc[:,["production_place"]]
    df.loc[df['production_place'].isin(['Portugal', 'Poland', 'Hungary', 'Czech Republic', 'Bulgaria', 'Turkey', 'Tunisia']), ('label')]='New_Economy_Europe'
    df.loc[df['production_place'].isin(['France', 'Germany', 'Spain', 'Italy', 'United Kingdom']), ('label')]='Mature_Economy_Europe'
    df.loc[df['production_place'].isin(['Singapore', 'Indonesia', 'Thailand', 'Malaysia', 'Vietnam', 'Philippines']), ('label')]='Asia'
    df.loc[df['production_place'].isin(['INDIA','India']), ('label')]='India'
    df.loc[df['production_place'].isin(['CHINA','China']), ('label')]='China'
    df.loc[df['production_place'].isin(['MEXICO','Mexico']), ('region')]='Mexico'
    label_to_list = df['label'].tolist()
    id_to_list = df['index'].tolist()
    value = []
    for idx in range(len(id_to_list)):
        value.append((label_to_list[idx],id_to_list[idx]))

    sql_update_query = """UPDATE costsim_parts SET label = %s where id = %s"""
    
    cur.executemany(sql_update_query,value)
    cnxn.commit()
    count = cur.rowcount
    print(count, "Enregistrements mis à jour avec succès")


    #print(df.loc[:,['part_weight','label']])
    #def weight_category(label_i):
    #for label_i in df['labellog'].unique():
    #    minvalue=round(df['part_weight'].loc[df['label']==label_i].min(),3)
    #    maxvalue=round(df['part_weight'].loc[df['label']==label_i].max(),3)
    #    return "Category", label_i, "is between", minvalue, "gr and",maxvalue, "gr"

    #def db_category(label_i):
    
    #    return df.loc[:,col].loc[df['production_place']==label_i]
    log_variables=['logannual_quantity', 'logmaterial_price', 'logvolume', 'logweight',
       'loglength', 'logwidth', 'logheight', 'logthick', 'logthickmax',
        'logamortizated']

    def correlation(query,col):
        df_query=df.loc[df['label']==query]
        for row in col:
            for column in col:
                #col = col
                #df_query=df_query.loc[:,col]
                correlation_df = df_query.corr()
                correlation_row_column = correlation_df.loc[row,column]
                if correlation_row_column >= 0.8 and correlation_row_column < 1:
                    if correlation_df.loc[row,'logamortizated'] >= correlation_df.loc[column,'logamortizated']:
                        #print(column, row, correlation_row_column, 'for row = ', row, correlation_df.loc[row,'logamortizated'], 'compare with column : ', column, correlation_df.loc[column,'logamortizated'])
                        #print('COLUMN DELETED IS   : ', column)
                        del col[col.index(column)]
  
                    else:
                        #print(column, row, correlation_row_column, 'for row = ', row, correlation_df.loc[row,'logamortizated'], 'compare with column : ', column, correlation_df.loc[column,'logamortizated'])
                        #print ('ROW DELETED IS    :', row)
                        del col[col.index(row)]
                        break
                    
        #print(col)
        y = df_query.loc[:,['logamortizated']]
        X = df_query.loc[:,col]
        for column in X.columns:
            model = sm.OLS(y,X).fit()
            p_value = model.pvalues[column]
            if p_value > 0.1:
                #print(column, 'has a p-value of : ', model.pvalues[column], 'and a correlation of :', correlation_df.loc[column, 'logamortizated'])
                if correlation_df.loc[column, 'logamortizated'] < 0.7:
                    X.drop(column, axis=1, inplace=True)
                    del col[col.index(column)]
                    #print(col)
        print(col)
        return col
    
    def algorithm(localization,dictionnary_parameters):
        print(localization.get('label'))
        df_query=df.loc[df['label']==localization.get('label')]
        print(list(dictionnary_parameters.keys()))
        split_test_train = train_test_split(df_query.loc[:,list(dictionnary_parameters.keys())], df_query.loc[:,['logamortizated']], test_size=0.3)
        X_train_df, X_test_df, y_train_df, y_test_df = split_test_train
        values_to_predict = pandas.DataFrame([dictionnary_parameters])

        ## Ridge regression algorithm:
        ridge_reg = Ridge(max_iter=10000)
        params_Ridge = {'alpha': [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7] , "fit_intercept": [True, False], "normalize": [True, False]}
        Ridge_GS = GridSearchCV(ridge_reg, cv=2, param_grid=params_Ridge, n_jobs=-1)
        Ridge_GS.fit(X_test_df,y_test_df)
        print(Ridge_GS.best_params_)
        Ridgeregression = Ridge(random_state=3, **Ridge_GS.best_params_)
        Ridgeregression.fit(X_train_df,y_train_df)
        y_ridge_pred=round(sympify(mpmath.exp(Ridgeregression.predict(values_to_predict)[0][0])),5)
        r_2_ridge_test = Ridgeregression.score(X_test_df,y_test_df)
        print("r² train : ",Ridgeregression.score(X_train_df,y_train_df),
        "r² test : ",Ridgeregression.score(X_test_df,y_test_df))
        print('y_ridge_pred is : ', y_ridge_pred)
        print('coef are : ', Ridgeregression.coef_)
        if params_Ridge['fit_intercept'] == True:
            print('intercept is :', Ridgeregression.intercept_)


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
                

        

    cur.close()
    cnxn.close()
    print("La connexion PostgreSQL est fermée")

except (Exception, pyodbc.Error) as error:
    print ("Erreur lors de la connexion à SQL Server", error)