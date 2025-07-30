#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[52]:


df=pd.read_csv(r"C:\Users\chitr\OneDrive\Documents\liver_cirrhosis.csv")
df.head(10)


# In[53]:


df.columns


# In[54]:


df.shape


# In[55]:


df.dropna(inplace=True)
df.isnull().sum()


# In[56]:


df.describe().transpose()


# In[57]:


df.info()


# In[58]:


numeric_data=df.select_dtypes(include=[np.number])
corr=numeric_data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


# In[59]:


#Converting categorical into numerical

df['Sex']=df['Sex'].replace({'F':1,'M':0})
df['Drug']=df['Drug'].replace({'Placebo':1,'D-penicillamine':0})
df['Ascites']=df['Ascites'].replace({'Y':1,'N':0})
df['Hepatomegaly']=df['Hepatomegaly'].replace({'Y':1,'N':0})
df['Spiders']=df['Spiders'].replace({'Y':1,'N':0})
df['Edema']=df['Edema'].replace({'Y':1,'N':0,'S':2})
df['Status']=df['Status'].replace({'C':0,'D':1,'CL':2})


# In[60]:


df.corr


# In[61]:


correlation=df.corr()
plt.figure(figsize=(16,8))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
plt.show()


# In[62]:


# Visualization of Age distribution by gender
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Age (days)', hue='Sex', element='step', stat='density', common_norm=False)
plt.title('Age distribution by gender')
plt.xlabel('Age (days)')
plt.ylabel('Density')
plt.show()


# In[63]:


# Visualization of Stage based on gender count
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Stage', hue='Sex', palette=['orange', 'blue'])
plt.title('Stage based on gender count')
plt.xlabel('Stage')
plt.ylabel('Count')
plt.show()


# In[31]:


# Visualization of comparing bilirubin,albumin and prothrombin levels between patients with different stage
plt.figure(figsize=(3,4))
sns.boxplot(x='Stage', y='Bilirubin', data=df)
plt.title('Bilirubin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Bilirubin (mg/dL)')

plt.figure(figsize=(3,4))
sns.boxplot(x='Stage', y='Albumin', data=df)
plt.title('Albumin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Albumin')

plt.figure(figsize=(3,4))
sns.boxplot(x='Stage', y='Prothrombin', data=df)
plt.title('Prothrombin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Prothrombin')

plt.show()


# In[14]:


# Visualization of comparing bilirubin,albumin and prothrombin levels between patients with different stage
plt.figure(figsize=(3,4))
sns.scatterplot(x='Stage', y='Bilirubin', data=df)
plt.title('Bilirubin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Bilirubin (mg/dL)')

plt.figure(figsize=(3,4))
sns.scatterplot(x='Stage', y='Albumin', data=df)
plt.title('Albumin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Albumin')

plt.figure(figsize=(3,4))
sns.scatterplot(x='Stage', y='Prothrombin', data=df)
plt.title('Prothrombin Levels by Stage')
plt.xlabel('Stage')
plt.ylabel('Prothrombin')

plt.show()


# In[15]:


#Logistic Regression model

x=df.drop(columns="Stage")
y=df['Stage']
x.shape,y.shape


# In[16]:


#splitting of the data as Training data and Testing data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=5)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[17]:


from sklearn.linear_model import LogisticRegression

model_logistic=LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=10000)
model_logistic.fit(x_train,y_train)
predictions=model_logistic.predict(x_test)


# In[18]:


data = pd.DataFrame(data={"Prediction": predictions.flatten()}) 
print(data) 


# In[19]:


acc_logistic=accuracy_score(y_test,predictions)*100
print("Accuracy of Logistic Reggression:",acc_logistic)


# In[20]:


#finding roc-auc and log loss

from sklearn.metrics import roc_auc_score 
from sklearn.metrics import log_loss

y_pred_proba = model_logistic.predict_proba(x_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
logloss = log_loss(y_test, y_pred_proba)

print("Roc-Auc Score:",roc_auc)
print("Log loss:",logloss)


# In[21]:


conf_mat=confusion_matrix(y_test,predictions)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='coolwarm',xticklabels=[1,2,3],yticklabels=[1,2,3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[22]:


Report=classification_report(y_test,predictions)
print("Report",Report)


# In[23]:


#Dropping ascites to check if there is any correlation with spiders covariate

df=df.drop(labels='Ascites',axis=1)
df.head()


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=5)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression

model_logistic_2=LogisticRegression(class_weight='balanced',multi_class='multinomial',max_iter=10000)
model_logistic_2.fit(x_train,y_train)
prediction=model_logistic_2.predict(x_test)


# In[26]:


print(prediction)


# In[27]:


y_pred_proba = model_logistic_2.predict_proba(x_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
logloss = log_loss(y_test, y_pred_proba)

print("Roc-Auc Score:",roc_auc)
print("Log loss:",logloss)


# In[28]:


print("Accuracy of Logistic Regression:",accuracy_score(y_test,prediction)*100)


# In[29]:


conf_mat=confusion_matrix(y_test,prediction)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='coolwarm',xticklabels=[1,2,3],yticklabels=[1,2,3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[30]:


print("Report:",classification_report(y_test,prediction))


# In[31]:


#Random forest classifier

x=df.drop(columns="Stage")
y=df['Stage']
x.shape,y.shape


# In[32]:


#splitting of the data as Training data and Testing data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=5)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[33]:


from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier(n_estimators=100,random_state=42)
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)


# In[34]:


data = pd.DataFrame(data={"Prediction": y_pred.flatten()}) 
print(data) 


# In[35]:


#finding roc-auc and log loss

from sklearn.metrics import roc_auc_score 
from sklearn.metrics import log_loss

y_pred_proba = model_rf.predict_proba(x_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
logloss = log_loss(y_test, y_pred_proba)

print("Roc-Auc Score:",roc_auc)
print("Log loss:",logloss)


# In[36]:


acc_random=accuracy_score(y_test,y_pred)*100
print("Accuracy of Random Forest:",acc_random)


# In[37]:


conf_mat=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='coolwarm',xticklabels=[1,2,3],yticklabels=[1,2,3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[38]:


report=classification_report(y_test,y_pred)
print("Report",report)


# In[39]:


pip install xgboost


# In[40]:


#XGBoost model

from xgboost import XGBClassifier

df_xg=df.copy()
df_xg['Stage']=df['Stage'] - 1


# In[41]:


x=df_xg.drop(columns='Stage')
y=df_xg['Stage']


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model_xgb=XGBClassifier(objective='multi:softmax',num_class=3,random_state=5)
model_xgb.fit(x_train,y_train)
y_pred=model_xgb.predict(x_test)


# In[43]:


data = pd.DataFrame(data={"Prediction": y_pred.flatten()}) 
print(data) 


# In[53]:


new_data=[[2200,0,1,18454,1,0,1,0,1,0.5,137.0,4.04,227.0,598.0,52.70,57.0,256.0]]
predicted_class=model_xgb.predict(new_data)
print("prediction:",predicted_class)



# In[47]:


#finding roc-auc and log loss

from sklearn.metrics import roc_auc_score 
from sklearn.metrics import log_loss

y_pred_proba = model_xgb.predict_proba(x_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
logloss = log_loss(y_test, y_pred_proba)

print("Roc-Auc Score:",roc_auc)
print("Log loss:",logloss)


# In[48]:


acc_xgb=accuracy_score(y_test,y_pred)*100
print("Accuracy of XGBoost:",acc_xgb)


# In[49]:


conf_mat=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='coolwarm',xticklabels=[1,2,3],yticklabels=[1,2,3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[50]:


report=classification_report(y_test,y_pred)
print("Report",report)


# In[51]:


#Visualizing roc-auc score and log loss

x=np.array(["Roc-Auc_score of Logistic Regression","Roc-Auc_score of Random Forest","Roc-Auc_score of XGBoost"]) 
y=np.array([0.750,0.993,0.995]) 
plt.barh(x,y) 
plt.show() 
x=np.array(["Log Loss of Logistic Regression","Log Loss of Random Forest","Log Loss of  XGBoost"]) 
y=np.array ([0.910,0.209,0.121]) 
plt.barh(x,y) 
plt.show()


# In[52]:


classifiers=["Logistic Regression", "Random Forest", "XGBoost"]
accuracy_scores = [acc_logistic, acc_random, acc_xgb]

plt.figure(figsize=(6,6))
plt.bar(classifiers,accuracy_scores)
plt.title('Accuracy for different classifiers')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy Scores')
plt.ylim(50,100)
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:




