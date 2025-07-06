#### JOKE IS THE ALGORITHM PYTHON FILE


    ### All the libraries to import

import pyodbc 
import pandas as pd
from pandas import DataFrame
import mpmath
import numpy as np
from sympy import sympify
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.preprocessing import scale 
from sklearn.linear_model import Ridge, RidgeCV
# from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from .NN import algorithmNN 
from .xgb import algorithmLauncher
import scipy
import statsmodels
try:

### Try to connect to the database 

    server = '****' 
    database = '****' 
    username = '****' 
    password = '****'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()

    ### Select everything from the table costsim_parts of the database
    ### first please make sure that by adding data, there would be no duplicated data...if there is, they won't be in train AND test set

    cursor.execute("SELECT * FROM costsim_parts;") 
    # print(np.__version__)
    # print(scipy.__version__)
    # print(statsmodels.__version__)
    sql = "SELECT * FROM costsim_parts;"
    cursor.execute(sql)

    print("Sélectionner dans les lignes dans la table costsim")
    res = cursor.fetchall()
    table = []
    for row in res:
        table.append(row)
        
    ### at this time, connection to db is not supported by specific team in schneider, so all info just would be put in xlsx file according to time limit.

    df = pd.DataFrame.from_records(table, columns=['index', 'part_number', 'project_name', 'part_name', 'raw_material_designation', 'color', 'extra_material',
    'production_place', 'percentage_of_extra_material', 'raw_material_price', 'part_volume', 'part_weight', 'part_length', 'part_width', 'part_height', 'avrg_thickness', 'max_thickness', 'annual_target_quantity', 'part_price_amortizated','label'])

    # print(len(['index', 'part_number', 'project_name', 'part_name', 'raw_material_designation', 'color', 'extra_material',
    # 'production_place', 'percentage_of_extra_material', 'raw_material_price', 'part_volume', 'part_weight', 'part_length', 'part_width', 'part_height', 'avrg_thickness', 'max_thickness', 'annual_target_quantity', 'part_price_amortizated','label'])
    #       ,len([1213, '1', '1', '1', '1', '1', '1', '1', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,'label']))

    # cursor.execute("""INSERT INTO costsim_parts([id], [part_number], [project_name], [part_name], [raw_material_designation], [color], [extra_material], [production_place], [percentage_of_extra_material], [raw_material_price],
    # [part_volume], [part_weight], [part_length], [part_width], [part_height], [avrg_thickness], [max_thickness], [annual_target_quantity], [part_price_amortizated], [label]) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    #     1213, '1', '1', '1', '1', '1', '1', '1', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,'label').rowcount

    ### Creation of the algorithm function:
    # Linear regression with L2 penalization (ridge regression) and Gradient Boost for linear and tree (XGBoost) are the three developped algorithms in this function
    # Gridsearch is executed in order to select the best parameters for each execution of the function regarding the training data
    # The alpha or lambda parameter (L1 & L2 penalization) could be search into 10**np.linspace(10,-10,1000)*0.05 list. Since there are a lot of numbers to test, it take too long to execute, thus [0.0001,0.0005,0.0007,0.001,0.005,0.007,0.01,0.05,0.07,0.1,0.5,0.7] is temporary selected.
    # After selection of parameters and comparing the test data R² score, the best algorithm is kept for the cost estimation and the result of the prediction is returned

    def algorithm(localization,dictionnary_parameters):
        values_to_predict = pd.DataFrame([dictionnary_parameters])
        if localization=="Mexico":
            return algorithmLauncher(values_to_predict,localization)
        return algorithmNN(values_to_predict,localization, False)

    cursor.close()
    cnxn.close()
    print("La connexion SQL Server est fermée")

    ### If try failed, check the error to see what is wrong

except (Exception, pyodbc.Error) as error:
    print ("Erreur lors de la connexion à SQL Server", error)



    
    
