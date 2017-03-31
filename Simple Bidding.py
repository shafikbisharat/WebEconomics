
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pd.options.display.max_columns = 30
pd.options.display.max_rows = 10
filepath="train.csv" 
data=pd.read_csv(filepath)
# Training set:
filepath_val="validation.csv"
data_val=pd.read_csv(filepath_val)
# Testing set:
filepath_test="test.csv"
data_test=pd.read_csv(filepath_test)


# 

# In[2]:

#removing useless columns:
def fltr(data,columns_to_drop):
    return(data.drop(columns_to_drop, axis=1, inplace=True))
# Functions to put data in the correct format for estimation:
def get_set_tags(data):
    # Use this function to get the set of tags in the document:
    user_tags=data.usertag
    list_tags=[]
    for i in user_tags:
        for j in i.split(','):
            list_tags.append(j)
    set_tags=list(set(list_tags))
    return set_tags
# removing the null tags:
def remove_objects(list_values,obj="null"):
    return [x for x in list_values if x != obj]

# Functions to separate the usertag columns:
def create_tag_index_dictio(data,column="usertag",split_char=","):
    # Creates a dictionary with key: tag and value: list of rows
    tag_series=data[column].apply(lambda x:x.split(split_char))
    k=0
    tag_index={}
    for tags_line,index_tag in zip(tag_series,tag_series.index):
        k+=1
        for tag in tags_line:
            if tag != "null":
                if tag not in tag_index.keys():
                    tag_index[tag]=[index_tag]
                else:
                    tag_index[tag].append(index_tag)
    return tag_index

def get_tagsDF(dictio_tag_index,training_set):
    # Create the dictionary with key:usertag, value: list with 1 tha impression correspond to this tag 
    N=training_set.shape[0]
    dictio_tags_values={}
    for user_tag in dictio_tag_index.keys():
        sparce_array=[0]*N
        for i in dictio_tag_index[user_tag]:
            sparce_array[i]=1
        dictio_tags_values[user_tag]=sparce_array
    #print ("%d users completed, creating dataframe"%(len(dictio_tags_values.keys())))
    return pd.DataFrame(dictio_tags_values)
#we normalize things we compare like slot price, NOT the city because we don't compare city number 80 with city number
#60. We can change the slot width but because it's 250 200 100 so it's catogerial 
def normalize(dataframe,column="slotprice"): 
    dataframe[column]=dataframe[column]/dataframe[column].std()

columns_to_dummy=["advertiser","adexchange","slotvisibility","slotformat","city", "region","weekday","hour","slotheight","slotwidth"]
columns_to_drop=["logtype","userid","urlid","url","bidprice","keypage","creative","domain","IP"
                 ,"slotid"]
columns_to_drop2=["logtype","userid","urlid","url","keypage","creative","domain","IP","slotid"]
train = (data)
val = (data_val)
test = (data_test)

fltr(train,columns_to_drop)
fltr(val,columns_to_drop)
fltr(test,columns_to_drop2)



# In[3]:

# Getting the set of tags:

list_tags=get_set_tags(train)
list_tags=remove_objects(list_tags) 
# creating dictionaries:
dictio_tag_index=create_tag_index_dictio(train)
dictio_os_browser=create_tag_index_dictio(train,"useragent","_")
# creating new dataframe:
N = train.shape[0]
tags_valuesDF=get_tagsDF(dictio_tag_index,train)
os_browserDF=get_tagsDF(dictio_os_browser,train)
train=pd.get_dummies(train,columns=columns_to_dummy)
clean_data1 = pd.concat([train, tags_valuesDF,os_browserDF], axis=1)
clean_data1.drop(["useragent","usertag"],axis=1,inplace=True)
train=np.array(clean_data1)


# In[4]:

# Getting the set of tags:
list_tags=get_set_tags(val)
list_tags=remove_objects(list_tags) 
# creating dictionaries:
dictio_tag_index=create_tag_index_dictio(val)
dictio_os_browser=create_tag_index_dictio(val,"useragent","_")
# creating new dataframe:
N = val.shape[0]
tags_valuesDF=get_tagsDF(dictio_tag_index,val)
os_browserDF=get_tagsDF(dictio_os_browser,val)
val=pd.get_dummies(val,columns=columns_to_dummy)
clean_data2 = pd.concat([val, tags_valuesDF,os_browserDF], axis=1)
clean_data2.drop(["useragent","usertag"],axis=1,inplace=True)
val=np.array(clean_data2)


# In[5]:

# Getting the set of tags:
test = (data_test)

list_tags=get_set_tags(test)
list_tags=remove_objects(list_tags) 
# creating dictionaries:
dictio_tag_index=create_tag_index_dictio(test)
dictio_os_browser=create_tag_index_dictio(test,"useragent","_")
# creating new dataframe:
N = test.shape[0]
tags_valuesDF=get_tagsDF(dictio_tag_index,test)
os_browserDF=get_tagsDF(dictio_os_browser,test)
test=pd.get_dummies(test,columns=columns_to_dummy)
clean_data3 = pd.concat([test, tags_valuesDF,os_browserDF], axis=1)
clean_data3.drop(["useragent","usertag"],axis=1,inplace=True)
test=np.array(clean_data3)


# In[6]:

train=np.array(train)
val=np.array(val)
test=np.array(test)


# In[7]:

trainBidID = train[:,1:2]
trainSlotPrice = train[:,2:3]/1000
trainPayprice = train[:,3:4]/1000
trainY  = data['click']
trainY  = trainY.as_matrix()
trainX = train[:,4:]

valBidID = val[:,1:2]
valSlotPrice = val[:,2:3]/1000
valPayprice = val[:,3:4]/1000
valY = data_val['click']
valY = valY.as_matrix()
valX = val[:,4:]

testBidID = test[:,0]
testSlotPrice = test[:,1:2]/1000
testX = test[:,2:]


# In[8]:

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

neg_w=np.sum(trainY)/len(trainY)
pos_w=1-neg_w
clf = LogisticRegression(class_weight={1:pos_w,0:neg_w},solver = 'sag')
clf = clf.fit(trainX,trainY)
predict_y = clf.predict(valX)


# In[9]:

counter = 0
for i in range(len(valY)):
    if valY[i] == 1 and predict_y[i] == 1:
        counter += 1
print(counter,sum(predict_y),sum(valY))


# In[10]:

fpr, tpr, thresholds = metrics.roc_curve(valY, predict_y)
print ("The AUC is:",metrics.auc(fpr, tpr))


# In[11]:

def get_pro(dataframe):
    x1 = np.zeros(len(dataframe))
    xx1 = clf.predict_proba(dataframe)
    for i in range(len(xx1)):
        x1[i] = xx1[i][1]
    return x1

trainProb =get_pro(trainX)
ValProb = get_pro(valX)


# In[12]:

trainProb = trainProb.reshape(len(trainProb),-1)
trainX = np.concatenate((trainProb, trainSlotPrice), axis=1)
trainY = trainPayprice

ValProb = ValProb.reshape(len(ValProb),-1)
ValX = np.concatenate((ValProb, valSlotPrice), axis=1)
ValY = valPayprice


# In[ ]:




# In[45]:

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

svr_rbf = SVR(kernel='rbf',C = 1e3,max_iter = 3000)
#dtr = DecisionTreeRegressor()
#dtr.fit(trainX,trainY)
svr_rbf.fit(trainX,trainY)



# In[ ]:




# In[46]:

ValBid = svr_rbf.predict(ValX)


# In[47]:

data_val['myBid'] = ValBid
data_val['myProb'] = ValProb
data_val['PayPrice'] = valPayprice


# In[48]:

data_test['myBid'] = ValBid
data_test['myProb'] = ValProb
data_test['PayPrice'] = valPayprice



# In[49]:

filepath_test="shafik_val.csv"
shafik=pd.read_csv(filepath_test)


# In[ ]:




# In[50]:

CTR = 0
CPC = 0
Clicks = 0
Win = 0
Loss = 0
Budget = 6250
MyBids = 0
for payprice,click,prob,myBid in data_val[['PayPrice','click','myProb','myBid']].values:
    payprice = payprice/1000
    myBid = myBid/1000
    if myBid>=payprice and Budget >=payprice:
        Win = Win + 1
        Budget = Budget - payprice
        Clicks = Clicks + click
        MyBids = MyBids + myBid
    else:
        Loss = Loss + 1
CTR = (Clicks/Win)*100
CTR0 = (data_val['click'].values.sum()/len(data_val))*100
Budget0 = data_val['payprice'].values.sum()/1000


# In[51]:

print("CTR:",CTR,"Orginal:",CTR0)
print("Budget Left:",6250 - Budget,"Orginal:",Budget0)
print("Wins:",Win,"MyBids:",MyBids,"Clicks:",Clicks) # was 0.51933064050779


# In[52]:

print("Impressions:",Win)
print("Clicks:",Clicks)
print("CTR:",CTR)
print("CPM:",(Budget/Win)*1000)
print("eCPC:",Budget/Clicks)
print("Spent:",Budget)


# In[86]:

columns_to_drop3=["weekday","hour","useragent","region","city","adexchange","slotwidth","slotheight","slotvisibility",                "slotformat","usertag","slotprice"]
shafik_val = data_val
fltr(shafik_val,columns_to_drop3)


# In[137]:

shafik_val


# In[126]:

testProb = get_pro(testX)
testProb = testProb.reshape(len(testProb),-1)
testX = np.concatenate((testProb, testSlotPrice), axis=1)
testBid = svr_rbf.predict(testX)
data_test['myBid'] =testBid
data_test['myProb'] = testProb


# In[130]:

shafik_test = data_test
#fltr(shafik_test,columns_to_drop3)


# In[ ]:




# In[129]:

shafik_test.to_csv('shafik_test.csv')


# In[108]:

shafik_val.to_csv('shafik_val.csv')


# In[ ]:



