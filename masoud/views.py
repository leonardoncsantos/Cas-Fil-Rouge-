#### Views created here 

from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.db.models import Max, Min
import numpy as np
from costsim.models import Parts
import regex as re
import logging
from .joke import algorithm
import pandas as pd
from math import isclose
from .NN import algorithmNN


from django.template.defaulttags import register
from django import template

logger = logging.getLogger(__name__)
columnNames = ['annual_target_quantity', 'max_thickness','raw_material_price', 'part_volume', 'part_weight','part_width','part_height', 'part_length','avrg_thickness']
@register.filter
def get_item(dictionary, key):
    if dictionary.get(key)[0] != None:
            return dictionary.get(key)[0]
    else:
        return None

### Index page


def home(request):

    context = {'title': 'Cost Estimation'}

    return render(request, 'costsim/welcome.html', context)

def index(request):

    context = {'title': 'Cost Estimation'}

    return render(request, 'costsim/masoud/index.html', context)

def check_data(df):
    global columnNames
    if np.issubdtype(np.array(df.loc[:,columnNames+["part_price_amortizated"]].values).dtype, np.number) and not (df.loc[:,columnNames+["part_price_amortizated"]].isnull().values.any()):
        return True
    return False

def removeRepetitiveData(df):
    maindf = pd.read_csv("./costsim/src/masoud/out22.csv" , encoding='windows-1252')
    ind=[]
    print("heree")
    for r in range(0,len(df)):
        if df.loc[r:r+1,'part_number'].values[0] in maindf.loc[:,'part_number'].values:
            ind.append(r)
            continue
    
    print("here")
    
    df=df.drop(ind,axis=0)
    return df
    
    
def dropfile(request):
    global columnNames
    uploaded_file = request.FILES.get('getFile')
    if type(uploaded_file) is None:
        message = "data is not uploaded"
        context = {
            'error': message
        }
        print(message)
        from django.contrib import messages
        messages.warning(request, message)
        return render(request, 'costsim/masoud/index.html', context)
    
    print("uploaded_file: ", uploaded_file)
    data = pd.read_excel(uploaded_file)
    
    
    context = {'title': 'Cost Estimation',}
    PRICE_EXCEL_SPELLING = 'part_price_amortizated(CNY)'
    variables_to_print = ['annual target quantity', 'max thickness(mm)','raw material price(€)', 'part volume(cm3)', 'part weight(gr)','part width(mm)','part height(mm)', 'part length(mm)','avrg thickness']
    
    dictOfColumnsChange = {variables_to_print[i]: columnNames[i] for i in range(len(variables_to_print))}
    dictOfColumnsChange[PRICE_EXCEL_SPELLING] = 'part_price_amortizated'
    # dictOfColumnsChange[url.columns[:8]] = url.loc[:,url.columns[:8]]
    data=data.rename(columns=dictOfColumnsChange)
    try:
        data.loc[:,columnNames]=data.loc[:,columnNames]
    except:
        message = "File Format is incorrect"
        context = {
            'error': message
        }
        print(message)
        from django.contrib import messages
        messages.warning(request, message)
        return render(request, 'costsim/masoud/index.html', context)
    if not check_data(data):
        message = "Format is incorrect - file contains non numeric or Nan data"
        context = {
            'error': message
        }
        print(message)
        from django.contrib import messages
        messages.warning(request, message)
        return render(request, 'costsim/masoud/index.html', context)
    data=removeRepetitiveData(data)
    if len(data) ==0:
        message = "repetitive data"
        context = {
            'error': message
        }
        from django.contrib import messages
        messages.warning(request, message)
        return render(request, 'costsim/masoud/index.html', context)
    lbl=""
    if data['production_place'].values[0] in ['Portugal', 'Poland', 'Hungary', 'Czech Republic', 'Bulgaria', 'Turkey', 'Tunisa']:
        lbl = 'New_Economy_Europe'
        data['label']='New_Economy_Europe'
    elif data['production_place'].values[0] in ['China', 'china']:
        lbl = 'China'
        data['label']='China'
    elif data['production_place'].values[0] in ['INDIA','Inda']:
        lbl = 'India'
        data['label']='India'
    elif data['production_place'].values[0] in ['MEXICO','Mexio']:
        data['label']='Mexico'
        lbl = 'Mexico'
    elif data['production_place'].values[0] in ['Singapore', 'Indonesia', 'Thailand', 'Malaysia', 'Vietnam', 'Philippins']:
        data['label']='Asia'
        lbl = 'Asia'
    
    maindf = pd.read_csv("./costsim/src/masoud/out22.csv" , encoding='windows-1252')
    maindf= pd.concat([maindf, data],axis=0)
    try:
        maindf.to_csv('./costsim/src/masoud/out22.csv', index=False)
    except:
        message = "there is a problem in saving file"
        context = {
            'error': message
        }
        from django.contrib import messages
        messages.warning(request, message)
        return render(request, 'costsim/masoud/index.html', context)
    
    # algorithmNN(0,lbl,True)
    message = "Database updated"
    context = {
        'error': message
    }
    from django.contrib import messages
    messages.warning(request, message)
    return render(request, 'costsim/masoud/index.html', context)
### Search page:
# It's the location production place box to fulfill
# It gets the country production place and compare if it belongs to group of countries or specific country
# Gets the label (the group of countries), and applies the correlation function to get the specific variables required to execute the algorithm

def search(request):    
    query = request.GET.get('query')
    global columnNames
    
    variables_to_print = ['annual target quantity', 'max thickness(mm)','raw material price(€)', 'part volume(cm3)', 'part weight(gr)','part width(mm)','part height(mm)', 'part length(mm)','avrg thickness']
    if not query:
        message = "Please write down a production place"
        context = {
            'error': message
        }
        return render(request, 'costsim/index.html', context)
    else:
        if query in ('Portugal', 'Poland', 'Hungary', 'Czech Republic', 'Bulgaria', 'Turkey', 'Tunisia', 'New_Economy_Europe'):
            query = 'New_Economy_Europe'
        elif query in ('Singapore', 'Indonesia', 'Thailand', 'Malaysia', 'Vietnam', 'Philippines', 'Asia'):
            query = 'Asia'
        elif query in ('France', 'Germany', 'Spain', 'Italy', 'United Kingdom', 'Mature_Economy_Europe'):  # revenir dessus
            query = 'Mature_Economy_Europe'
        elif query == 'India':
            query = 'India'
        elif query == 'China':
            query = 'China'
        elif query == 'Mexico':
            query = 'Mexico'
        else:
            message = "This country doesn't exist. Please double check and enter the right name of the country."
            context = {
                'error': message
            }
            return render(request, 'costsim/masoud/index.html', context)
        allparts = Parts.objects.filter(label=query)
        if not allparts:
            # Please, add the parts produced in this country for next year cost estimation."
            message = "This country doesn't exist. Please double check and enter the right name of the country."
            context = {
                'error': message
            }
            return render(request, 'costsim/masoud/index.html', context)
        else:
            title = "Results of the request %s" % query
            context = {
                'allparts': allparts,
                'query': query,
                'title': title,
                'variables': columnNames,
                'variables_to_print': variables_to_print
            }
        print("here")
    return render(request, 'costsim/masoud/category.html', context)

    ### Isint and isfloat functions are to check if a value is convertible to an integer or a float.

def isint(value):
    try:
        int(value)
        return True 
    except ValueError:
        return False
def isfloat(value):
    try:
        float(value)
        return True 
    except ValueError:
        return False




#         ### Calculus is the page with the variables to fulfill in order to estimate the cost of the part.
#         # This function appends the right unit regarding the variable
#         # It applies the algorithm function

def calculus(request):
    given_name = request.GET.get('partname')
    global columnNames
    url = request.GET
    url = pd.DataFrame(dict(url))
    print(url)
    label = url['label'][0]
    variables_to_print = ['annual target quantity', 'max thickness(mm)','raw material price(€)', 'part volume(cm3)', 'part weight(gr)','part width(mm)','part height(mm)', 'part length(mm)','avrg thickness']
    
    url = url.loc[:,variables_to_print]
    dictOfColumnsChange = {variables_to_print[i]: columnNames[i] for i in range(len(variables_to_print))}
    url=url.rename(columns=dictOfColumnsChange)
    
    for k in columnNames:
        if url[k].values == ['']:
            message = "Please, be sure that all the box are fullfilled"
            context = {
                'message': message,
                'url': columnNames,
                'query': label
            }
            return render(request, 'costsim/masoud/estimation.html', context)
        elif not (isfloat(url[k].values) | isint(url[k].values)):
            message = "Please, write down numbers and numbers only. \n Make sure there is no comma nor space between the numbers. \n If decimal numbers, write them with a point, as follow : 3.5."
            context = {
                'message': message,
                'url': columnNames,
                'query': label
            }
            return render(request, 'costsim/masoud/estimation.html', context)
    index=0
    valid_dictionnary_parameters={}
    for k,v in url.items():
        valid_dictionnary_parameters[columnNames[index]]=float(v[0])
        index+=1
    print(valid_dictionnary_parameters)
    results = algorithm(label, valid_dictionnary_parameters)
    url = dict(request.GET)

    for k in url.keys():
        url[k]=url[k][0]



    context = {
        'information':url,
        'url': variables_to_print,
        'results': results,
        'query': label,
        'givenname': given_name
        }
    return render(request, 'costsim/masoud/estimation.html', context)