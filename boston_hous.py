#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modules nécessaires pour le notebook
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import metrics
 


# In[4]:


# lire le fichier de données
#utiliser le param index_col: Column to use as the row labels of the DataFrame


columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Loading Boston Housing Dataset
df = pd.read_csv('housing.csv', delimiter=r"\s+", names = columns)
# Top 5 rows of the boston dataset
df.head()


# In[5]:


df.describe()


# In[ ]:





# In[32]:


import matplotlib.pyplot as plt
import seaborn as sb

# Setting Seaborn Style
sb.set(style = 'whitegrid')


# In[33]:


plt.figure(figsize=(8, 6))
sb.distplot(df['CRIM'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['ZN'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['INDUS'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['CHAS'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['NOX'], rug = True)

plt.figure(figsize=(8, 6))
sb.distplot(df['RM'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['AGE'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['DIS'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['RAD'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['TAX'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['PTRATIO'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['B'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['LSTAT'], rug = True)


plt.figure(figsize=(8, 6))
sb.distplot(df['MEDV'], rug = True)


# In[34]:


sns.pairplot(data=df, x_vars=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], 
             y_vars='MEDV', height=7, aspect=0.7)


# In[35]:


cols_predicteurs = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
#predicteurs
X = df[cols_predicteurs]
y = df.MEDV


# In[36]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, 
                                        y , test_size = 0.2, random_state=42)
#detail de chacun des sous-dataset
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[37]:


#estimation des coeeficients du modele lineaire
lm = LinearRegression()
lm.fit(X_train,y_train)
#Afficher les coefficients
print(lm.intercept_)
print(lm.coef_)


# In[38]:


#Afficher l'equation
list(zip(cols_predicteurs, lm.coef_))


# In[39]:


# proceder au test
y_pred = lm.predict(X_test)


# In[40]:


import numpy as np
#comparer les valeurs test et prédites
test_pred_df = pd.DataFrame( { 'Valeurs test': y_test,
                'Valeurs prédites': np.round( y_pred, 2),
                'residuels': y_test - y_pred } )
test_pred_df[0:10]


# In[41]:


# RMSE
mse = np.sqrt(metrics.mean_squared_error(y_test,
                                        y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,
                                        y_pred)))

#Calcul du R-squared
r2 = metrics.r2_score(y_test, y_pred)
print(r2)


# In[42]:


# Write scores to a file,
with open("metrics.txt", 'w') as outfile:
        outfile.write("MSE:  {0:2.1f} \n".format(mse))
        outfile.write("R2: {0:2.1f}\n".format(r2))


# In[ ]:





# In[ ]:




