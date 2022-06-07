#!/usr/bin/env python
# coding: utf-8

# In[541]:
import pickle


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[542]:


Data = pd.read_csv('P.csv')
Data.head()


# In[543]:


Data = Data.dropna()


# In[544]:


Data


# In[545]:


Data.drop(['reading score','parents visit'],  axis=1, inplace=True)
Data


# In[546]:


Data.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split()) > 1) else each for each in Data.columns ]
print(Data.columns)


# In[547]:


Data


# In[548]:


plt.scatter(Data.gender,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[549]:


plt.scatter(Data.parental_level,Data.score_bac,marker='+',color='red')


# In[550]:


plt.scatter(Data.restoration,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[551]:


plt.scatter(Data.test_preparation,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[552]:


plt.scatter(Data.score_math,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[553]:


X = Data.iloc[:,:-1]
Y = Data.iloc[:,-1]


# In[554]:


X


# In[555]:


Y


# In[556]:


Data.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split()) > 1) else each for each in Data.columns ]
print(Data.columns)


# In[557]:


Data.head()


# In[558]:


X['gender'].replace( 'female', 0 ,inplace=True)
X['gender'].replace( 'male', 1 ,inplace=True)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()


# In[559]:


# dfle = Data
# dfle.gender = le.fit_transform(dfle.gender)
# # dfle.race/ethnicity = le.fit_transform(dfle.race/ethnicity)
# # dfle.parental_level = le.fit_transform(parental_level)
# # dfle.lunch = le.fit_transform(lunch)
# # dfle.test_preparation = le.fit_transform(test_preparation)
# dfle


# In[ ]:





# In[560]:


X['test_preparation'].replace( 'none', 0 ,inplace=True)
X['test_preparation'].replace( 'completed', 1 ,inplace=True)


# In[561]:


X['parental_level'].replace( 'bachelor\'s degree', 1 ,inplace=True)
X['parental_level'].replace( 'some college', 2 ,inplace=True)
X['parental_level'].replace( 'master\'s degree', 3 ,inplace=True)
X['parental_level'].replace( 'associate\'s degree', 4 ,inplace=True)
X['parental_level'].replace( 'high school', 5 ,inplace=True)
X['parental_level'].replace( 'some high school', 6 ,inplace=True)


# In[562]:


# X['parents visit'].unique()


# In[563]:


# X['parents visit'].replace( 'group A', 1 ,inplace=True)
# X['parents visit'].replace( 'group B', 2 ,inplace=True)
# X['parents visit'].replace( 'group C', 3 ,inplace=True)
# X['parents visit'].replace( 'group D', 4 ,inplace=True)
# X['parents visit'].replace( 'group E', 5 ,inplace=True)


# In[564]:


X['restoration'].unique()


# In[565]:


X['restoration'].replace( 'standard', 0 ,inplace=True)
X['restoration'].replace( 'free/reduced', 1 ,inplace=True)


# In[566]:


X


# In[567]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[568]:


X_train


# In[569]:


Y_train


# In[ ]:





# In[ ]:





# In[570]:


regressor = LinearRegression()

regressor.fit(X_train,Y_train)

# model = Lasso()
# model.fit(X_train,Y_train)


# In[572]:


Y_pred=regressor.predict(X_test)


# In[573]:


Y_pred


# In[574]:


Y_test


# In[575]:


Y_test.to_numpy()


# In[ ]:





# In[ ]:





# In[577]:


print(regressor.score(X_test,Y_test))


# In[579]:


print(regressor.predict([[1,1,1,1,95]]))




# In[ ]:



pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))