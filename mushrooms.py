
# coding: utf-8

# In[3]:


# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:



mushrooms = pd.read_csv(r'C:\Users\81wingo\Desktop\510 assignment1\Codes\mushroom.csv')


# In[5]:


mushrooms.head(6)


# In[6]:


print(mushrooms.shape)


# In[7]:


mushrooms.describe()


# In[8]:


# show how number of  ediate vs poisonous mushrooms
print(mushrooms.groupby('class').size())


# In[9]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = labelencoder.fit_transform(mushrooms[col])
 
mushrooms.head()


# In[10]:


X = mushrooms.iloc[:,1:23]  # all rows, all the features and no labels
y = mushrooms.iloc[:, 0]  # all rows, label only
X.head()
y.head()


# In[12]:


X.describe()


# In[13]:


X.corr()


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# In[16]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[17]:


covariance=pca.get_covariance()


# In[38]:


explained_variance=pca.explained_variance_
explained_variance


# In[45]:


cap_colors = mushrooms['cap-color'].value_counts()
m_height = cap_colors.values.tolist() #Provides numerical values
cap_colors.axes #Provides row labels
cap_color_labels = cap_colors.axes[0].tolist() #Converts index object to list

#=====PLOT Preparations and Plotting====#
ind = np.arange(10)  # the x locations for the groups
width = 0.7        # the width of the bars
colors = ['brown','gray','red','yellow','#E5E4E2','#F0DC82','pink','#D22D1E','purple','green']
#FFFFF0
fig, ax = plt.subplots(figsize=(10,7))
mushroom_bars = ax.bar(ind, m_height , width, color=colors)

#Add some text for labels, title and axes ticks
ax.set_xlabel("Cap Color",fontsize=20)
ax.set_ylabel('Quantity',fontsize=20)
ax.set_title('Cap Color Quantity',fontsize=22)
ax.set_xticks(ind) #Positioning on the x axis
ax.set_xticklabels(('brown', 'gray','red','yellow','white','buff','pink','cinnamon','purple','green'), fontsize = 12)
plt.show()


# In[20]:


N=mushrooms.values
pca = PCA(n_components=2)

x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[46]:


scatter_matrix(mushrooms)
plt.show()

