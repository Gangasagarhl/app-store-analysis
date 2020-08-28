#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv("C:/Users/RAGHAVENDRA/Desktop/appstore_games (2).csv")


# In[ ]:


data.shape


# # Performing data cleansing

# In[5]:


d=data.dtypes
print("Before data cleansing\n\n\n\n")
data.info()



s=data.shape
for i in range(s[1]):
    if data.iloc[:,i].isnull().sum()>0:
        if d[i]=="int64" or d[i]=="float64":
            data.iloc[:,i].fillna(value=np.mean(data.iloc[:,i]),inplace=True)
        elif d[i]=="object":
            data.iloc[:,i].fillna(value="",inplace=True)
print("\n\n\nData is now cleaned\n\n\n")

data.info()


# # Finding the genre of the highest user ratings

# In[113]:


#Higher user rating is 5,got to know after analysing the datasets analysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
for j in [5,4.5]:
    higher=pd.DataFrame(data[data["Average User Rating"]== j])
    cv=CountVectorizer()
    cv1=CountVectorizer()
    count_matrix=cv.fit_transform(higher["Genres"])
    count_matrix1=cv.fit_transform(higher["Primary Genre"])
    total=[]
    total1=[]
    for i in cosine_similarity(count_matrix):
        total.append(np.sum(i))
    for i in cosine_similarity(count_matrix1):
        total1.append(np.sum(i))
        
    higher["ttl"]=total
    higher["ttl1"]=total1
    
    higher1=higher[higher.ttl==higher.ttl.max()]
    higher2=higher[higher.ttl1==higher.ttl1.max()]
    printing=list(higher1["Genres"])
    printing1=list(higher2["Primary Genre"])
    
    print("Rating's {} with  Genres is :  '{}'".format(j,printing[0]))
    print("Rating's {} with  Primary Genre is :  '{}'".format(j,printing1[0]))


# In[18]:





# # Identifying trends between "Average User Ratings" and "Price"

# In[107]:


for i in data["Average User Rating"].unique():
    x=data[data["Average User Rating"]==i]["Average User Rating"]
    y=data[data["Average User Rating"]==i]["Price"]
    plt.plot(y,x)

plt.xlabel("\n\nPrice")
plt.ylabel("Average User Rating\n\n")
plt.title("Price vs. Average User Rating")
plt.legend(data["Average User Rating"].unique(),loc="best")
plt.show()


# In[92]:


plt.scatter(x=data["Average User Rating"],y=data["Price"])
plt.show()


# # Stating inferences:
#  
#  ## 1)The most used language is:English
#  ## 2)Different number of games available is:16847
#  ## 3)Highest rating for a)Primary Genre :-"Games"   b)Genres:-"Games, Strategy"
#  ## 4)user rating count a)max-3032734.0  b)min-5.0
#  ## 5)Price a)max-179.99  b)min-0.0
#  ## 6)Size a)max-4005591040.0   b)min-51328.0

# In[108]:


data.head()


# In[110]:


#Different number of games available is:
len(data.Name.unique())


# In[112]:


# diiferent type of primary genre available are
data["Primary Genre"].unique()


# In[116]:


print("User Rating Count")
print("Maximum=",data["User Rating Count"].max())
print("Minimum=",data["User Rating Count"].min())


# In[117]:


print("Price")
print("Maximum=",data["Price"].max())
print("Minimum=",data["Price"].min())


# In[121]:


print("Size")
print("Maximum=",data["Size"].max())
print("Minimum=",data["Size"].min())

