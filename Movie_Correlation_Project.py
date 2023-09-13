#!/usr/bin/env python
# coding: utf-8

# In[7]:


#import libraries
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8)  # to adjust the configuration of the plots we ll create


# In[5]:


# reading the data
df=pd.read_csv(r'C:\Users\feven\Downloads\Movies\movies.csv')
df.head()


# In[6]:


#check for missing data
missing_values = df.isnull().sum()

# Display the count of missing values
print(missing_values)


# In[8]:


#Data cleaning
#1. Data types for our columns
df.dtypes


# In[ ]:


#change the dtype of columns
df['budget']= df['budget'].astype('int64')
df['gross']= df['gross'].astype('int64')


# In[12]:


#Drop duplicates
df.drop_duplicates()


# In[ ]:


#budget high correlation


# In[16]:


#scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings(in millions)')
plt.ylabel('Budget for film(in millions)')
plt.show()


# In[22]:


#plot budget vs gross using sns
sns.regplot(x='budget', y='gross',data=df, scatter_kws={"color":"green"}, line_kws={"color":"blue"})


# In[26]:


#determining correlation
df.corr() 
#type of correlation
df.corr(method='pearson')  #chose pearson by choosing deafult
#df.corr(method='kendall')
#df.corr(method='spearman')


# In[28]:


#there is a high correlation between budget and gross
correlation_matrix=df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[29]:


#
df_numerized=df
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype=='object'):
        df_numerized[col_name]=df_numerized[col_name].astype('category')
        df_numerized[col_name]=df_numerized[col_name].cat.codes
df_numerized


# In[30]:


df


# In[31]:


correlation_matrix=df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation matrix for numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[33]:


df_numerized.corr()


# In[41]:


correlation_mat=df_numerized.corr()
corr_pairs=correlation_mat.unstack()
corr_pairs
sorted_pairs=corr_pairs.sort_values()
high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#It is seen that votes and budget have the highest correlation to gross earnings 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




