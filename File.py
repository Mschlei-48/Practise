import pandas as pd
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data1=pd.read_csv("titanic_test.csv")
data2=pd.read_csv("titanic_train.csv")
print(data1.shape)
print(data2.shape)
#Combine the data row wise
data=pd.concat([data1,data2])
print(data.shape)
#Check for information regarding the data
data.info()
#Check head of the data
print(data.head())
#Summary of data types

#1. Numerical columns
#Age,SibSp,Parch,Fare,

#2. Categorical columns
#Pclass,Name,Sex,Ticket number,Cabin,Embarked

#Numerical Columns with missing values
#Age,Fare

#Categorical columns with missing values
#Cabin,Embarked

#Check percentage of missing values 
for i in data.columns:
    if data.loc[:,i].isnull().sum()!=0:
        if (data.loc[:,i].isnull().sum()/data.shape[0])*100 > 40:
            print(f"{i} has more than 40% missing values")
        elif (data.loc[:,i].isnull().sum()/data.shape[0])*100 < 40:
            print(f"{i} has less than 40% of missing data")
#Delete Cabin as it has more than 40% missing values
data.drop(columns=["Cabin"],inplace=True)
#Check if it has been deleted
print("Remaining columns are :",data.columns)
#Replace missing values of age and fare with mean
mean_ages=data.loc[:,"Age"].mean()
print("Mean of ages is",mean_ages)
data.loc[:,"Age"].fillna(int(mean_ages),inplace=True)
mean_fare=data.loc[:,"Fare"].mean()
print("Mean of Fare is :",mean_fare)
data.loc[:,"Fare"].fillna(int(mean_fare),inplace=True)
#Replace missing values of Embarked with mode
print("Frequency of values in Embarked column :",data.loc[:,"Embarked"].value_counts())
data.loc[:,"Embarked"].fillna("S",inplace=True)
#Check if the missing values are still there
data.info()
#Check if each columns till has nan
print("Checking if there are any remaining missing values")
for i in data.columns:
    print(data.loc[:,i].isna().sum())
#Check for duplicated rows
duplicates=data.duplicated()
print(duplicates.head(10))
#Remove duplicates
#Check shape of data before dropping duplicates
print(data.shape)
data=data.drop_duplicates()
#Check shape of data after dropping duplicates
print(data.shape)
#Delete the name column as it is useless
data.drop(columns=["Name"],inplace=True)
#Encode the categorical columns to be numerical for modelling stage
data=pd.get_dummies(data,columns=["Sex","Ticket","Embarked"])
print(data.head())
#Modelling
#Get features
features=data.iloc[:,:-1]
labels=data.iloc[:,-1]
#Mkae a matrix from the features and labels
features=features.values
labels=labels.values
#Check the shape of the new matrices
print(features.shape)
print(features[0,:])
print(labels.shape)
print(labels[0:10])
#Remove missing values
nan_mask=np.isnan(data).any(axis=1)
features=features[~nan_mask]
labels=labels[~nan_mask]
#Scale the data
scaler=StandardScaler()
features=scaler.fit_transform(features)
#Build the model
#Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.3,shuffle=True)
model=LogisticRegression()
model.fit(x_train,y_train)
#Make a prediction and get the  accuracy
y_pred=model.predict(x_test)
accu=accuracy_score(y_test,y_pred)  
print("Äccuracy is :",accu)
with open('results.txt','w') as f:
    f.write(f"Accuracy is: {accu}")
#Plot the predictions
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()
plt.savefig("ConfusionMatrix.png")