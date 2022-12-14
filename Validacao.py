#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd


# In[120]:


df = pd.read_csv('cadastro.csv')


# In[121]:


df.head()


# In[122]:


df.shape


# In[123]:


df.isnull().sum().sort_values(ascending=False)


# In[124]:


df.dropna(inplace=True)


# In[125]:


df.isnull().sum().sort_values(ascending=False)


# In[126]:


df['CATEGORIA'].value_counts()


# In[127]:


import seaborn as sns


# In[128]:


sns.countplot(df['CATEGORIA'])


# In[129]:


from sklearn.model_selection import train_test_split


# In[191]:


X_train, X_test, y_train, y_test = train_test_split(df['Nome SKU'], df['EMBALAGEM'], random_state=50)


# In[192]:


print(X_train)


# In[193]:


print(y_train)


# In[194]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df = 3, ngram_range=(1,2)).fit(X_train)


# In[195]:


len(vect.get_feature_names())


# In[196]:


X_train_vectorized = vect.transform(X_train)


# In[197]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[198]:


# Save Predictions
predictions = model.predict(vect.transform(X_test))


# In[199]:


teste = pd.DataFrame(predictions)
teste.to_excel('teste.xlsx')
real = pd.DataFrame(y_test)
real.to_excel('real.xlsx')


# In[140]:





# In[76]:


gabarito = pd.Series(y_test)
teste = pd.DataFrame(predictions)
frames = [gabarito,teste]
results = pd.concat(frames, keys=['gabarito', 'teste'], axis =1)


# In[79]:


results.to_excel('resultados.xlsx')
gabarito.to_excel('gabarito.xlsx')
teste.to_excel('teste.xlsx')


# In[ ]:




