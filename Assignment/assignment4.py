#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# 
# Name : Najma Parveen
# Roll No :963219104022
# 
# 1.Loading Dataset into tool

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


dataframe = pd.read_csv("Downloads/abalone.csv")
dataframe.head()


# In[3]:


import seaborn as sns


# In[5]:


sns.boxplot(dataframe['Diameter'])


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.hist(dataframe['Diameter'])


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.plot(dataframe['Diameter'].head(10))


# In[8]:


plt.pie(dataframe['Diameter'].head(),autopct='%.3f')


# In[19]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[20]:


sns.distplot(dataframe['Diameter'].head(300))


# In[23]:


import matplotlib.pyplot as plt


# In[26]:


plt.scatter(dataframe['Diameter'].head(400),dataframe['Length'].head(400))


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


plt.bar(dataframe['Sex'].head(20),dataframe['Rings'].head(20))
plt.title('Bar plot')
plt.xlabel('Diameter')
plt.ylabel('Rings')


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[55]:


sns.barplot(dataframe ['Sex'], dataframe['Rings'])


# In[127]:


import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[135]:


sns.jointplot(dataframe['Diameter'].head(50))


# In[183]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[172]:


sns.lineplot(dataframe['Diameter'].head())


# In[146]:


sns.boxplot(dataframe['Length'].head(10))


# In[151]:


fig=plt.figure(figsize=(8,5))
sns.heatmap(dataframe.head().corr(),annot=True)


# In[ ]:


sns.pairplot(dataframe.head(),hue='Height')


# In[ ]:


sns.pairplot(dataframe.head())


# In[64]:


dataframe.head()


# In[66]:


dataframe.tail()


# In[69]:


dataframe.describe()


# In[70]:


dataframe.mode().T


# In[71]:


dataframe.shape


# In[73]:


dataframe.kurt()


# In[74]:


dataframe.skew()


# In[75]:


dataframe.var()


# In[76]:


dataframe.nunique()


# In[77]:


dataframe.isna()


# In[78]:


dataframe.isna().any()


# In[79]:


dataframe.isna().sum()


# In[80]:


sns.boxplot(dataframe['Diameter'])


# In[81]:


quant=dataframe.quantile(q=[0.25,0.75])
quant


# In[82]:


iqr=quant.loc[0.75]-quant.loc[0.25]
iqr


# In[84]:


low=quant.loc[0.25]-(1.5*iqr)
low


# In[85]:


up=quant.loc[0.75]+(1.5*iqr)
up


# In[88]:


dataframe['Diameter']=np.where(dataframe['Diameter']<0.155,0.4078,dataframe['Diameter'])
sns.boxplot(dataframe['Diameter'])


# In[86]:


sns.boxplot(dataframe['Length'])


# In[90]:


dataframe['Length']=np.where(dataframe['Length']<0.23,0.52, dataframe['Length'])
sns.boxplot(dataframe['Length'])


# In[91]:


sns.boxplot(dataframe['Height'])


# In[89]:


dataframe['Height']=np.where(dataframe['Height']<0.04,0.139, dataframe['Height'])
dataframe['Height']=np.where(dataframe['Height']>0.23,0.139, dataframe['Height'])
sns.boxplot(dataframe['Height'])


# In[92]:


sns.boxplot(dataframe['Whole weight'])


# In[93]:


dataframe['Whole weight']=np.where(dataframe['Whole weight']>0.9,0.82, dataframe['Whole weight'])
sns.boxplot(dataframe['Whole weight'])


# In[94]:


sns.boxplot(dataframe['Shucked weight'])


# In[95]:


dataframe['Shucked weight']=np.where(dataframe['Shucked weight']>0.93,0.35, dataframe['Shucked weight'])
sns.boxplot(dataframe['Shucked weight'])


# In[96]:


sns.boxplot(dataframe['Viscera weight'])


# In[97]:


dataframe['Viscera weight']=np.where(dataframe['Viscera weight']>0.46,0.18, dataframe['Viscera weight'])
sns.boxplot(dataframe['Viscera weight'])


# In[99]:


sns.boxplot(dataframe['Shell weight'])


# In[104]:


dataframe['Shell weight']=np.where(dataframe['Shell weight']>0.61,0.2388, dataframe['Shell weight'])
sns.boxplot(dataframe['Shell weight'])


# In[105]:


dataframe['Sex'].replace({'M':1,'F':0,'I':2},inplace=True)
dataframe


# In[107]:


x=dataframe.drop(columns= ['Rings'])
y=dataframe['Rings']
x


# In[109]:


y


# In[108]:


from sklearn.preprocessing import scale
x = scale(x)
x


# In[110]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print(x_train.shape, x_test.shape)


# In[112]:


from sklearn.linear_model import LinearRegression
MLR=LinearRegression()


# In[113]:


MLR.fit(x_train,y_train)


# In[114]:


y_pred=MLR.predict(x_test)
y_pred


# In[115]:


pred=MLR.predict(x_train)
pred


# In[ ]:


from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
accuracy


# In[116]:


MLR.predict([[1,0.455,0.365,0.095,0.5140,0.2245,0.1010,0.150]])


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[117]:


from sklearn.linear_model import Lasso, Ridge
#intialising model
lso=Lasso(alpha=0.01,normalize=True)
#fit the model
lso.fit(x_train,y_train)
Lasso(alpha=0.01, normalize=True)
#prediction on test data
lso_pred=lso.predict(x_test)
#coef
coef=lso.coef_
coef


# In[119]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
metrics.r2_score(y_test,lso_pred)


# In[150]:


np.sqrt(y_test,lso_pred)


# In[120]:


#initialising model
rg=Ridge(alpha=0.01,normalize=True)
#fit the model
rg.fit(x_train,y_train)
Ridge(alpha=0.01, normalize=True)
#prediction
rg_pred=rg.predict(x_test)
rg_pred


# In[121]:


rg.coef_


# In[122]:


metrics.r2_score(y_test,rg_pred)


# In[123]:


np.sqrt(mean_squared_error(y_test,rg_pred))

