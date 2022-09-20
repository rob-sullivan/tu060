#!/usr/bin/env python
# coding: utf-8

# # H&M Recommender System

# # Generate Kaggle Predictions File

#working with files and memory management
import gc

#working with data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#working with datetime feature
from datetime import datetime

#handling missing values where not dropped
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#for evaluating our model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

#for dimension reduction
from sklearn.pipeline import Pipeline # to sequence training events
from sklearn.decomposition import TruncatedSVD

#model
from sklearn.neighbors import KNeighborsClassifier

#used to provide information to the user when running this notebook
from IPython.display import display, clear_output


# In[2]:


#H&M Collaborative KNN Model Based Recommendation System
def HmRecSys_data_prep():
    
    #GET DATA
    #output message for user
    clear_output(wait=True)
    display('Importing Data. Please wait...')
    
    #get transaction data
    transactions_train_df = pd.read_csv("data/transactions_train.csv", 
                                        dtype={"article_id": "str"}) # import the transactions dataset

    #get product meta data
    articles_df = pd.read_csv("data/articles.csv", dtype={"article_id": "str"})

    #get customer meta data
    customers_df = pd.read_csv("data/customers.csv")

    #output message for user
    clear_output(wait=True)
    display('Preparing Data. Please wait...')
    
    #PREPARE DATA
    #prepare transactions dataset
    features_df = transactions_train_df[['article_id',
                                         'customer_id', 
                                         't_dat', 
                                         'price', 
                                         'sales_channel_id']]
    
    del transactions_train_df
    gc.collect()
    
    clear_output(wait=True)
    display('imported transactions and arranged columns...')
    
    #First we will convert our date text into a panda date type.
    features_df["t_dat"] = pd.to_datetime(features_df["t_dat"])
    
    clear_output(wait=True)
    display('converted date into datetime object...')

    clear_output(wait=True)
    display('converted article_ids to strings...')
    
    #merge product meta data with transactions
    features_df = features_df.merge(articles_df, left_on='article_id', right_on='article_id')
    
    del articles_df
    gc.collect()
    
    clear_output(wait=True)
    display('merged articles with transactions...')
    
    #we drop cols we don't need from products dataset
    features_df.drop(['prod_name',
                      'product_type_name',
                      'graphical_appearance_name',
                      'colour_group_name',
                      'perceived_colour_value_name',
                      'perceived_colour_master_name',
                      'department_name',
                      'index_name',
                      'index_group_name',
                      'section_name',
                      'garment_group_name',
                      'detail_desc'], axis=1)    
    
    clear_output(wait=True)
    display('rearranged columns of features dataset for merger...')

    clear_output(wait=True)
    display('merging customers with features dataset...')
    
    #merge customer meta data with transactions
    features_df = features_df.merge(customers_df, left_on='customer_id', right_on='customer_id')
    
    del customers_df
    gc.collect()
    
    clear_output(wait=True)
    display('rearranging customers and articles with feature dataset...')
    
    #we reorganise columns
    features_df = features_df[['customer_id',#the customer
                               'FN',#customer meta data
                               'Active',
                               'club_member_status', 
                               'fashion_news_frequency', 
                               'age',
                               'product_code',#product meta data
                               'product_type_no',
                               'product_group_name',
                               'graphical_appearance_no',
                               'colour_group_code', 
                               'perceived_colour_value_id', 
                               'perceived_colour_master_id', 
                               'department_no',  
                               'index_code', 
                               'index_group_no',  
                               'section_no', 
                               'garment_group_no', 
                               't_dat',#transaction meta data
                               'price',
                               'sales_channel_id', 
                               'article_id']]#the product
    clear_output(wait=True)
    display('rearranged columns of features dataset...')
    
    #FIX MISSING DATA
    #convert from objects and floats to categories and ints
    features_df['club_member_status'] = features_df['club_member_status'].astype('category')
    features_df['fashion_news_frequency'] = features_df['fashion_news_frequency'].astype('category')
    
    features_df['FN'] = features_df['FN'].fillna(0)
    features_df['Active'] = features_df['Active'].fillna(0)

    club_member_status = features_df.iloc[:, 3:-18].values
    fashion_news_frequency = features_df.iloc[:, 4:-17].values
    age = features_df.iloc[:, 5:-16].values

    #ref: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    imputer_med = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    #we replace missing values with the most frequent
    imputer_mf.fit(club_member_status)
    club_member_status = imputer_mf.transform(club_member_status)

    imputer_mf.fit(fashion_news_frequency)
    fashion_news_frequency = imputer_mf.transform(fashion_news_frequency)

    #we replace any missing age values with the median age
    imputer_med.fit(age)
    age = imputer_med.transform(age)

    #now add corrected columns back into our main customer dataframe
    features_df.iloc[:, 3:-18] = club_member_status
    features_df.iloc[:, 4:-17] = fashion_news_frequency
    features_df.iloc[:, 5:-16] = age

    #replace minus sign in text and check result of dataset after imputing missing values
    features_df.columns = features_df.columns.str.replace('-', '')

    #lower case columns
    features_df.columns = map(str.lower, features_df.columns)

    clear_output(wait=True)
    display('filled in missing values and fixed column names...')
    
    # ENCODE DATA
    #encode our categorical variables
    le = preprocessing.LabelEncoder()
    features_df.iloc[:,3] = le.fit_transform(features_df.iloc[:,3])#club_member_status
    features_df.iloc[:,4] = le.fit_transform(features_df.iloc[:,4])#fashion_news_frequency
    features_df.iloc[:,8] = le.fit_transform(features_df.iloc[:,8])#product_group_name
    features_df.iloc[:,14] = le.fit_transform(features_df.iloc[:,14])#index_code
    
    #encode date as ordinal after we split our training and test data
    features_df['t_dat'] = features_df['t_dat'].apply(lambda x: x.toordinal())

    clear_output(wait=True)
    display('encoded features...')

    # SPLIT DATA X FEATURES Y PREDICTOR
    #These are the attributes of our customers
    X_train = features_df.iloc[:, 1: 20].values
    
    # this is the product or in our case the class
    y_train = features_df.iloc[:, 21].values

    del features_df
    gc.collect()
    
    clear_output(wait=True)
    display('split data into X features and y predictor...')
    
    # SCALE FEATURES: MinMax - range (0,1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = min_max_scaler.fit_transform(X_train)

    clear_output(wait=True)
    display('features scaling complete...')
 
    clear_output(wait=True)
    display('exporting X_train_scaled np array to text file...')

    np.savetxt("data/X_train_scaled.csv", X_train_scaled, delimiter=",")

    clear_output(wait=True)
    display('exporting y_train np array to text file...')
    np.savetxt("data/y_train.csv", y_train, delimiter=",", fmt='%s')
    
    del X_train
    del y_train
    del X_train_scaled
    gc.collect()
    
    clear_output(wait=True)
    display('Data preparation complete.')


# In[3]:


def HmRecSys_model_train(): 

    #output message for user
    clear_output(wait=True)
    display('Importing X_train np array from text file. Please wait...')
    
    X_train_scaled = np.loadtxt('data/X_train_scaled.csv', delimiter=',')
    
    #output message for user
    clear_output(wait=True)
    display('Importing y_train df from text file. Please wait...')
    
    #get product meta data
    y_train = np.loadtxt('data/y_train.csv', delimiter=',')
    
    # CREATE PIPELINE
    #create model pipeline
    clear_output(wait=True)
    display('Creating pipeline...')
    
    steps = [('svd', TruncatedSVD(n_components=15)), 
             ('knn', KNeighborsClassifier(n_neighbors=3, metric="minkowski", p=2))]
    
    model = Pipeline(steps=steps)

    #output message for user
    clear_output(wait=True)
    display('training model please wait...')
                                 
    #TRAIN MODEL
    model.fit(X_train_scaled, y_train)
    
    #output message for user
    clear_output(wait=True)
    display('model trained...')


# In[4]:


def HmRecSys_model_predict():
    
    #GET DATA
    #output message for user
    clear_output(wait=True)
    display('Importing np array from text file. Please wait...')
    
    X_train_scaled = np.loadtxt('data/X_test_scaled.csv', delimiter=',')
    

    #output message for user
    clear_output(wait=True)
    display('Importing Data. Please wait...')
    
    #get product meta data
    articles_df = pd.read_csv("data/articles.csv", dtype={"article_id": "str"})

    #get customer meta data
    customers_df = pd.read_csv("data/customers.csv")
    
    #get transaction data
    transactions_train_df = pd.read_csv("data/transactions_train.csv", 
                                        dtype={"article_id": "str"}) # import the transactions dataset

    clear_output(wait=True)
    display('Getting price mode')
    
    #get popular price to pay
    p = t_df['price'].mode()
    
    clear_output(wait=True)
    display('Getting sales mode')
    
    #get popular sales channel to buy from
    s = t_df['sales_channel_id'].mode()

    del t_df
    gc.collect() 
       
    clear_output(wait=True)
    display('Getting dates for next 7 days')
    
    #predict in next 7 days (2020-09-29)
    date = {'date': ['2020-09-29']}
    d_df = pd.DataFrame(date)
    d_df['date'] = pd.to_datetime(df['date'], format='%Y-%m%-d')
    d_df['date'] = d_df['date'].apply(lambda x: x.toordinal())
    d = d_df['date'].iloc[:].values
    
    del d_df
    gc.collect()  

    #output message for user
    clear_output(wait=True)
    display('opening CSV file to start writing...')
    
    write_file = "ros_predictions4.csv"
    with open(write_file, "wt", encoding="utf-8") as output:
        #add headers first
        output.write("customer_id,prediction" + '\n')
        
        #now we loop through each row and write predictions to csv file
        for index_i, cus in customers_df.iterrows():
            
            #we keep trying different products until we make a hit then we add it to the list
            for index_j, art in articles_df.iterrows():
                #get their meta data   
                features_df = [cus['FN'], 
                               cus['Active'], 
                               cus['club_member_status'], 
                               cus['fashion_news_frequency'],
                               cus['age'],
                               art['product_code'], 
                               art['product_type_no'],
                               art['product_group_name'],
                               art['graphical_appearance_no'],
                               art['colour_group_code'], 
                               art['perceived_colour_value_id'], 
                               art['perceived_colour_master_id'], 
                               art['department_no'],  
                               art['index_code'], 
                               art['index_group_no'],  
                               art['section_no'], 
                               art['garment_group_no'],
                               d,
                               p,
                               s]

                #normalise data to query customer
                q_cus_scaled = min_max_scaler.fit_transform(features_df)
                
                #free up memory
                del features_df
                gc.collect()  
                
                #make a prediction
                y_pred = model.predict(q_cus_scaled)
                result.append(y_pred)
 
                #create prediction csv file
                r = []
                r.append(cus.customer_id + ",")
                for n in result:
                    p = names.iloc[n]
                    r.append("0" + str(p))
                    prediction =  ' '.join(r)
                #write predictions to csv file
                output.write(prediction + '\n')
                clear_output(wait=True)
                display('Predicting: ' + str(index_i) + ", " + str(index_j))


# In[5]:


#HmRecSys_data_prep()


# In[6]:


HmRecSys_model_train()

