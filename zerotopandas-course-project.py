#!/usr/bin/env python
# coding: utf-8

# # Project Title - Suicides in INDIA
# 
# In this project, we analysis the dataset on the suicides in India from year 2001 to 2012. We took the dataset from kaggle.

# ## Downloading the Dataset
# 
# First, we download the dataset from kaggle using opendatasets library.

# In[1]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[2]:


dataset_url = 'https://www.kaggle.com/rajanand/suicides-in-india'


# In[3]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[4]:


data_dir = './suicides-in-india'


# In[5]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[6]:


project_name = "suicides_data_analysis"


# In[7]:


get_ipython().system('pip install jovian --upgrade -q')


# In[8]:


import jovian


# In[9]:


jovian.commit(project=project_name)


# ## Data Preparation and Cleaning
# 
# Now, we prepare data and clean it.
# 
# 

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


data_raw_df = pd.read_csv(data_dir + "/Suicides in India 2001-2012.csv")


# In[63]:


data_raw_df.head(15)


# In[13]:


data_raw_df.info()


# In[14]:


data_raw_df.shape 


# The dataset contains 237519 rows and 7 columns.

# In[15]:


data_raw_df.describe() # describes the dataset using statistics 


# In[16]:


#checking for the unique causes of suicide
print(data_raw_df['Type'].unique())
print("Total unique causes of suicide are ",data_raw_df['Type'].nunique())


# In[17]:


#checking for null values in each column
data_raw_df.isnull().sum()


# In[18]:


import jovian


# In[19]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# Now, we begin to analyse the dataset by plotting different graphs using seaborn and matplotlib.pyplot.

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[20]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# First, we analyse the total number of suicides from year 2001 to 2012.

# In[21]:


# Total suicides from year 2001 to 2012
data_suicides = data_raw_df['Total'].groupby(data_raw_df['Year']).sum()

print("Total number of suicides from Year 2001 to 2012")
data_suicides


# In[22]:


sns.set_theme(style="darkgrid", context="notebook",palette='deep')
fig=plt.figure(figsize=(15,10))
ax=plt.axes()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.grid(linestyle="-", axis="y", color="black")
year = data_raw_df['Year'].unique()
plt.bar(year,data_suicides, color='Blue', width=0.8)
plt.xlabel("Year",fontweight='bold',fontsize=15)
plt.ylabel("Total suicides",fontweight='bold',fontsize=15)
plt.title("Total number of suicides in each year", fontweight='bold',fontsize=25)
plt.xticks(year,fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()


# From the above bar graph, we can see that the year 2011 has the highest number of suicides.

# Now, we plot a histogram for different age group from year 2001-2012.

# In[23]:


age_count = data_raw_df['Age_group'].value_counts()
print("Number of suicides in different age groups :")
age_count


# In[24]:


plt.figure(figsize=(20, 10))
plt.title("Number of suicides in different age groups",fontsize=20)
plt.xlabel('Age Group',fontsize=15 )
plt.ylabel('Number of suicides',fontsize=15)
plt.hist(data_raw_df['Age_group'],color='purple', align='left')
plt.show()


# Now, we look at the number of males and females by plotting pie chart.

# In[25]:


gender_counts = data_raw_df.Gender.value_counts()
gender_counts


# In[26]:


plt.figure(figsize=(15,6))
plt.title("Number of men and women who attempted suicide", fontsize=20)
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=45);


# From the above pie chart, we can see that the number of males is slightly higher than the number of females who attempted suicide.

# Now , we look at total number of people who attempted suicide due to different reasons.

# In[27]:


reasons = data_raw_df["Type_code"].value_counts()
reasons = pd.DataFrame(reasons).reset_index()
reasons.columns = ['Reasons','Total_suicides']
reasons


# In[28]:


plt.figure(figsize=(10, 5))
plt.title("Reasons of suicides",fontsize=20)
plt.xlabel('Reasons',fontsize=15 )
plt.ylabel('Total_suicides',fontsize=15)
sns.barplot(y="Reasons", x="Total_suicides", data=reasons)
plt.show()


# So, highest number of suicides occur due to causes.

# Now, we analyse the total number of suicides from 2001 to 2012 statewise.

# In[29]:


states = data_raw_df['Total'].groupby(data_raw_df['State']).sum()
states = pd.DataFrame(states).reset_index()
states.columns = ['State','Total_suicides']
states = states.drop([31,32,33])
states


# In[30]:


plt.figure(figsize=(20, 10))
plt.title("Number of suicides in different states",fontsize=20)
plt.xlabel('States',fontsize=15 )
plt.ylabel('Total_suicides',fontsize=15)
plt.xticks(rotation=90)
sns.barplot(x="State", y="Total_suicides", data=states)
plt.show()


# From above bar graph, we can see that Maharashtra has highest number of suicides and Lakshadweep has lowest number of suicides during 2001-2012.

# Let us save and upload our work to Jovian before continuing

# In[31]:


import jovian


# In[32]:


jovian.commit()


# ## Asking and Answering Questions
# 
# Now, we try to answer some interesting questions about the dataset.
# 
# 

# #### Q1: Show the change in the suicide rate through year by year analysis.

# In[33]:


sns.set_theme(style='darkgrid', context='notebook')
fig = plt.figure(figsize=(16,8))

ax=plt.axes()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(linestyle="-", axis="y", color="black")

sns.lineplot(data=data_suicides, palette="plasma",linewidth=2.5)
plt.xticks(year,fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Year",fontweight='bold',fontsize=15)
plt.ylabel("Suicide Rate",fontweight='bold',fontsize=15)
plt.title("Suicide Rate in INDIA from year 2001 to 2012",fontsize=20)
plt.show()


# From above lineplot, We can see that there has been an increase in the suicide rate over the years and a slight decrease in the year 2012.

# #### Q2: Which cause was the highest responsible for suicide and which one was the least responsible? 

# In[34]:


causes = data_raw_df['Type'].value_counts()
causes = pd.DataFrame(causes).reset_index()
causes.columns = ['Type','Total_Suicides']
causes.sort_values(by="Total_Suicides", ascending=False)


# From above dataframe, we can see that maximum number of suicides occur due to others and minimum number of suicides occur by other means.

# Also, we can see that second highest number of suicides are due to divorce.

# #### Q3: What is the state-wise report of suicides?

# In[35]:


states.sort_values(by="Total_suicides",ascending=False)


# We can see that Maharashtra has recorded maximum number of suicides from year 2001-2012.

# #### Q4: Which gender has highest number of suicides in different age group?

# In[36]:


gender = data_raw_df.groupby(['Age_group','Gender']).size()
gender = pd.DataFrame(gender).reset_index()
gender.columns = ['Age_group','Gender','Total']
gender


# In[37]:


gender_male = pd.DataFrame(gender, index=[0,2,4,6,8,10])
gender_male


# In[38]:


gender_female = pd.DataFrame(gender, index=[1,3,5,7,9,11])
gender_female


# In[52]:


sns.set_theme(style='darkgrid', context='notebook')
fig = plt.figure(figsize=(14,8))

ax=plt.axes()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(linestyle="-", axis="y", color="black")


width = 0.4
bar1 = np.arange(6)
bar2 = [i+width for i in bar1]
plt.bar(bar1,gender_male['Total'],0.4, label ="Male")
plt.bar(bar2,gender_female['Total'],0.4, label = "Female")
plt.xticks(bar1 + width/2,gender_female['Age_group'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Age_group",fontweight='bold',fontsize=15)
plt.ylabel("Number of suicides",fontweight='bold',fontsize=15)
plt.legend(fancybox=True, loc="upper left",borderpad=1)
plt.title("No. of males and females from different age group",fontsize=20)
plt.show()


# We can see that there is a slight difference in numbers of males and females who commit suicide.

# #### Q5: What is the  number of suicides in Delhi?

# In[70]:


df_delhi = data_raw_df[data_raw_df['State']=='Delhi (Ut)'].copy()
df_delhi.head(20)


# In[90]:


delhi_counts = df_delhi["Type"].value_counts()
delhi_counts =pd.DataFrame(delhi_counts).reset_index()
delhi_counts.columns = ["Type","Total"]
delhi_counts.sort_values("Total", ascending=False)


# In[89]:


sns.set_theme(style='darkgrid', context='notebook')
fig = plt.figure(figsize=(20,10))

ax=plt.axes()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(linestyle="-", axis="y", color="black")


plt.title("Number of suicides in Delhi from 2001 to 2012",fontsize=20)
plt.xlabel('Type',fontsize=15,fontweight='bold')
plt.ylabel('Total_suicides',fontsize=15,fontweight='bold')
plt.xticks(rotation=90)
sns.barplot(x="Type", y="Total", data=delhi_counts)
plt.show()


# Above graph shows that maximum suicides occur due to others in Delhi region.

# Let us save and upload our work to Jovian before continuing.

# In[91]:


import jovian


# In[92]:


jovian.commit()


# ## Inferences and Conclusion
# 
# 1. Maharashtra has the maximum number of suicides and Lakshadweep has the minimum number of suicides from year 2001 to 2012.
# 2. The second highest number of suicides occur due to divorce.

# In[93]:


import jovian


# In[94]:


jovian.commit()


# ## References and Future Work
# 
# In future, this can be used to find correlation between difference types of suicides.

# In[95]:


import jovian


# In[96]:


jovian.commit()


# In[97]:


jovian.submit(assignment="zero-to-pandas-project")


# In[ ]:




