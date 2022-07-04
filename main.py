#!/usr/bin/env python
# coding: utf-8

# In[42]:
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso


# In[43]:


Data = pd.read_csv('P.csv')
Data.head()


# In[44]:


Data = Data.dropna()


# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


Data


# In[46]:


Data.drop(['reading score','parents visit'],  axis=1, inplace=True)
Data


# In[47]:


Data.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split()) > 1) else each for each in Data.columns ]
print(Data.columns)


# In[48]:


Data


# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


Data


# In[50]:


plt.scatter(Data.gender,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[51]:


fig = plt.figure(figsize=(10,8))
plt.scatter(Data.parental_level,Data.score_bac,marker='+',color='red')


# In[52]:


plt.scatter(Data.restoration,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[53]:


plt.scatter(Data.test_preparation,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[54]:


plt.scatter(Data.score_math,Data.score_bac,marker='+',color='red')
plt.ylabel('Score Bac')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


X = Data.iloc[:,:-1]
Y = Data.iloc[:,-1]


# In[56]:


X


# In[57]:


Y


# In[58]:


Data.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split()) > 1) else each for each in Data.columns ]
Data.columns


# In[59]:


Data.head()


# In[60]:


X['gender'].replace( 'female', 0 ,inplace=True)
X['gender'].replace( 'male', 1 ,inplace=True)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()


# In[61]:


# dfle = Data
# dfle.gender = le.fit_transform(dfle.gender)
# # dfle.race/ethnicity = le.fit_transform(dfle.race/ethnicity)
# # dfle.parental_level = le.fit_transform(parental_level)
# # dfle.lunch = le.fit_transform(lunch)
# # dfle.test_preparation = le.fit_transform(test_preparation)
# dfle


# In[ ]:





# In[62]:


X['test_preparation'].replace( 'none', 0 ,inplace=True)
X['test_preparation'].replace( 'completed', 1 ,inplace=True)


# In[63]:


X['parental_level'].replace( 'bachelor\'s degree', 1 ,inplace=True)
X['parental_level'].replace( 'some college', 2 ,inplace=True)
X['parental_level'].replace( 'master\'s degree', 3 ,inplace=True)
X['parental_level'].replace( 'associate\'s degree', 4 ,inplace=True)
X['parental_level'].replace( 'high school', 5 ,inplace=True)
X['parental_level'].replace( 'some high school', 6 ,inplace=True)


# In[64]:


# X['parents visit'].unique()


# In[65]:


# X['parents visit'].replace( 'group A', 1 ,inplace=True)
# X['parents visit'].replace( 'group B', 2 ,inplace=True)
# X['parents visit'].replace( 'group C', 3 ,inplace=True)
# X['parents visit'].replace( 'group D', 4 ,inplace=True)
# X['parents visit'].replace( 'group E', 5 ,inplace=True)


# In[66]:


X['restoration'].unique()


# In[67]:


X['restoration'].replace( 'standard', 0 ,inplace=True)
X['restoration'].replace( 'free/reduced', 1 ,inplace=True)


# In[68]:


X


# In[69]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[70]:


X_train


# In[71]:


Y_train


# In[ ]:





# In[72]:


# model = Lasso()
# model.fit(X_train,Y_train)


# In[73]:


regressor = LinearRegression()

regressor.fit(X_train,Y_train)


# In[74]:


# Y_pred=model.predict(X_test)


# In[75]:


# Y_pred


# In[76]:


Y_pred=regressor.predict(X_test)


# In[77]:


# Y_pred


# In[78]:


Y_test


# In[79]:


Y_test.to_numpy()


# In[ ]:





# In[80]:


# model.score(X_test,Y_test)


# In[81]:


regressor.score(X_test,Y_test)


# In[82]:


regressor.predict([[0,1,1,1,60]])


pickle.dump(regressor,open('model.pkl','wb'))


print("Le fichier model.pkl est sauvegardé dans le dossier courant")




