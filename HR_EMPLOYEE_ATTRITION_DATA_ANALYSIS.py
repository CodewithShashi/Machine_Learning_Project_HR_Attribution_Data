#!/usr/bin/env python
# coding: utf-8

# ## HR Attrition Data

# ### Prepared By
# ### - Shash Shekhar Pandey

# ### In this Project , we need to predict whether a give employee will leave the organization or not. Your target column is Attrition. We will create a model with the following steps:
# 

# ###  Import the relevant packages
# 

# ###  Download and explore the dataset

# ###  Perform EDA, Apply dataset for preprocessing

# ###  Predict the target columns

# ## What is employee attrition?

# Employee attrition is when an employee leaves the company through any method, including voluntary resignations, layoffs, failure to return from a leave of absence, or even illness or death. Whenever anyone ceases working for the company for any reason and is not replaced for a long time (if ever), that would be employee attrition.

# In[1]:


pip install category-encoders


# ## IMPORTING REQUIRED PACKAGES

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## loading dataset

# In[5]:


data = pd.read_csv(r'C:\Users\SHASHI SHEKHAR\Downloads\HR_Employee_Attrition-1\HR_Employee_Attrition-1.csv')
data.head()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.isnull().sum()


# In[9]:


data.nunique()


# In[10]:


data.dtypes.sort_index()


# ## Looking for duplicate data

# In[11]:


data.duplicated().sum()


# In[12]:


data.describe()


# ## Data Analysis

# In[13]:


attrition_count = pd.DataFrame(data['Attrition'].value_counts())
attrition_count


# In[14]:


f, ax = plt.subplots(figsize=(10,10))
ax = data['Attrition'].value_counts(). plot.pie(explode=[0,0], autopct = '%1.1f%%', shadow=True)
ax.set_title('Attrition Probability')


# In[19]:


fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(
    x='Age', 
    hue='Attrition', 
    data = data, 
    palette="colorblind", 
    ax = ax,  
    edgecolor=sns.color_palette("dark", n_colors = 1),
    )


# In[16]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'EducationField', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Education Field")

ax[1,0] = sns.countplot(x='Attrition', hue= 'Department', data=data,  ax = ax[1,0], palette='Set1' )
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Department")

ax[0,1] = sns.countplot(x='Attrition', hue= 'Education', data=data,  ax = ax[0,1], palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Education")

ax[1,1] = sns.countplot(x='Attrition', hue= 'BusinessTravel', data=data,  ax = ax[1,1], palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Bussiness Travel")


f.tight_layout()


# In[20]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'JobRole', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Job Role")

ax[0,1] = sns.countplot(x='Attrition', hue= 'OverTime', data=data,  ax = ax[0,1],palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Over Time")

ax[1,1] = sns.countplot(x='Attrition', hue= 'EnvironmentSatisfaction', data=data,  ax = ax[1,1],palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Environment Satisfaction")

ax[1,0] = sns.countplot(x='Attrition', hue='WorkLifeBalance', data=data, ax = ax[1,0], palette='Set1')
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Work Life Balance")

f.tight_layout()


# ### object data types and their unique values

# ## correlation of the columns

# In[21]:


for column in data.columns:
    if data[column].dtype == object:
        print(str(column) + ' : ' + str(data[column].unique()))
        print(data[column].value_counts())
        print("-"*90)


# In[22]:


#Remove unneeded columns

#Remove the column EmployeeNumber
data = data.drop('EmployeeNumber', axis = 1) # A number assignment 
#Remove the column StandardHours
data = data.drop('StandardHours', axis = 1) #Contains only value 80 
#Remove the column EmployeeCount
data = data.drop('EmployeeCount', axis = 1) #Contains only the value 1 
#Remove the column EmployeeCount
data = data.drop('Over18', axis = 1) #Contains only the value 'Yes'


# In[23]:


data.corr()


# In[ ]:





# In[24]:


plt.figure(figsize=(15,15))
sns.heatmap(
    data.corr(), 
    annot=True, 
    fmt='.0%',
    linewidths=1,
    cmap='inferno'
)


# In[7]:


print(x.columns)


# In[ ]:


print(y)


# ### Train test split

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=42 )


# In[ ]:


print(X_train.shape, X_test.shape)


# ### converting categorical values to numerical values

# In[ ]:


encoder = ce.OrdinalEncoder(cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)


# ### Data Preprocessing

# In[ ]:


cols = X_train.columns

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns= [cols])
X_test = pd.DataFrame(X_test, columns=[cols])


# ### Building and testing model by applyling Random Forest ML algorithm

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)


# In[ ]:


score = accuracy_score(Y_test, y_pred)
print('randomforest classifier score: ', np.abs(score)*100)


# In[ ]:


y_pred


# ## final result 
# 

# ### Accuracy score = 95.23

# In[ ]:


f

