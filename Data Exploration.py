
# coding: utf-8

# In[1]:

# Libraries:
import pandas as pd
import numpy as np
from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')


# In[2]:

# Config:
# In order to display all the columns:
pd.options.display.max_columns = 30
pd.options.display.max_rows = 30


# ### Importing datasets:

# In[3]:

# Training set:
filepath="train.csv" 
data=pd.read_csv(filepath)
# Training set:
filepath_val="validation.csv"
data_val=pd.read_csv(filepath_val)
# Testing set:
filepath_test="test.csv"
data_test=pd.read_csv(filepath_test)


# In[4]:

# Functions for analysing data:
def print_different_values(dataframe):
    print ("total number of datapoints : "+str(len(dataframe)))
    for i in dataframe.columns:
        set_col=len(dataframe[i].value_counts())
        print ("%s has %d different points." %(i,set_col))


# In[5]:

# Ploting:

# Groupby functions:
def plot_ctr(dataframe,y,x,list_advert_tot):
    adv="advertiser"
    # Mean time series:
    plt.figure(figsize=(9,9))
    for list_advert in list_advert_tot:
        mean_data=data.groupby([x,adv]).mean()
        ts_mean=mean_data.unstack(adv)[y][list_advert]

        # STD:
        std_data=data.groupby([x,adv]).std()
        ts_std=std_data.unstack(adv)[y][list_advert]

        # Totals:
        totals=data.groupby([x,adv]).count()
        ts_totals=totals.unstack(adv)[y][list_advert]

        # margin:
        margin=ts_std*2/np.sqrt(ts_totals)
        plt.errorbar(x=ts_mean.index,y=ts_mean, yerr=margin,fmt='--o')
    plt.legend(list_advert_tot, loc=0)
    plt.ylabel(y)
    plt.xlabel(x)


# In[6]:

# Analising:
print (len(data.columns))
data.head()


# In[7]:

# Number of different values for each column:
print_different_values(data)


# In[8]:

# Basic Analysis:
# Num click:
num_clics=len(data[data.click==1])
print ("Clicks: %d"%num_clics)
# number of impressions:
num_impressions=len(data)
print ("Impressions: %d"%num_impressions)
# Click through Rate: number of Clicks / number ofImpressions
CTR=num_clics/num_impressions
print ("CTR: {:.4%}".format(CTR))   # Showing in percentage!


# ### After checkin the paper of ipinYou got some insights:
# #### Some Columns description:
# - logtype: 1 for impression, repeated
# - useragent: device/OS/browser
# - adexchange: 1 to 4 id of the "auction house"
# - urlid: null if is anonymous, in this case they all are
# - slotvisitility: first view means that it appears without the user having to scroll down
# - bidprice: the "optimal" bid price 
# - payprice: highest bid price from competitors, also called "market price" or "auction winning price"
# - advertiser: Corresponds ti a different categorie of the advertiser, this could be: "telecom", "Oil", etc
# 
# #### Some conclusion:
# - All the rows represent impressions (showing the ad after winnign the auction) for which the biding price was higher that the payprice and also higher that the other competitors

# In[ ]:




# In[9]:

# Plotting 
x= "weekday"
y="click"
list_advert=[1458,3358]
adv="advertiser"
plot_ctr(data,y,x,list_advert)


# In[ ]:




# ### Pareto
# #### In Pareto, we can easily see who are the main factors that control the Payprice or the Clicks
# - Main conclusions:
#     1. 3 out of 9 advertisers (33%) are responsible for 51.9% of the pay prices
#     2. 37 out of 370 cities (10%) are responsible for 50.1% of the pay prices
#     3. 14 out of 35 regions (40%) are responsible for 71.4% of the click!
#     

# In[10]:

def pareto(dataframe):
    sumpayprice = data.click.sum()
    for i in dataframe.columns:
        if len(dataframe[i].value_counts()) <= 400 and len(dataframe[i].value_counts())>2 :
            print ("-----------",i,"-----------")
            temp = data.groupby(i)["click"].sum().sort_values(ascending=False)
            temp = temp.to_dict()
            pareto = 0.0
            count = 0.0
            for j in temp:
                count = (count+1)
                countp = (count/len(temp))*100
                pareto = temp[j]/sumpayprice*100 + pareto
                print(j,temp[j],round(temp[j]/sumpayprice*100,2),round(pareto,2),round(countp,2))
                
pareto(data)


# ### Pivot Table
# #### The values shows the clicks each day for each advertiser
# - The main points are :
#     1. Distribution among days
#     2. Some advertisers have no activity in some days
#     3. We can easily change weekday to hours and see the distribution during the day and this important because we can profile our customers eg. kids, adults and etc.

# In[11]:

pivotable = pd.pivot_table(data, values='click', index=['weekday'],columns=['advertiser'], aggfunc=np.sum)
pivotable


# In[12]:

#Click Per Advertiser
CPA = (data.groupby('advertiser')["click"].sum())
#Impression per Advertiser
IPA = (data.groupby('advertiser')["logtype"].sum())
#Bids per Advertiser - Missing -
#Cost
Cost = (data.groupby('advertiser')["payprice"].sum())
#CTR
CTR = (CPA/IPA)*100
#CVR - MISSING -
#CPM
CPM = (Cost/IPA)*1000
#eCPC
eCPC = Cost/CPA

Table3 = [CPA,IPA,Cost,CTR,CPM,eCPC]
Table3 = pd.concat(Table3,axis=1)
Table3.columns = ['Clicks','Impressions','Cost','CTR','CPM','eCPC']
Table3
#all the advertisers has CTR less than 0.1% except for advertiser 2997 (0.46%).
#Although the nine advertisers have similar CPM, their efective cost-per-click (eCPC), i.e. the expected cost for achieving one click, are fairly diferent. This could be caused by the target rule setting (i.e., the target user demographic information, location and time) and the market of each specifc advertiser


# ## Constant bid
# ### Assumptions:
#     - Gaussian distribution
#     - 2 standard deviation should cover 95% of the cases
