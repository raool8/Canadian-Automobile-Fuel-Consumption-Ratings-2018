# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:45:36 2019

@author: Rahul
"""

'''
Understanding the Table				
Model	4WD/4X4 = Four-wheel drive			
    	AWD = All-wheel drive			
    	FFV = Flexible-fuel vehicle			
    	SWB = Short wheelbase			
    	LWB = Long wheelbase			
    	EWB = Extended wheelbase			
Transmission	A = Automatic			
            	AM = Automated manual			
            	AS = Automatic with select shift			
            	AV = Continuously variable			
            	M = Manual			
            	3 – 10 = Number of gears			
Fuel Type	X = Regular gasoline			
        	Z = Premium gasoline			
        	D = Diesel			
        	E = Ethanol (E85)			
        	N = Natural gas			
Fuel Consumption	City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per imperial gallon (mpg)											
CO2 Emissions	the tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving											
CO2 Rating	the tailpipe emissions of carbon dioxide rated on a scale from 1 (worst) to 10 (best)											
Smog Rating	the tailpipe emissions of smog-forming pollutants rated on a scale from 1 (worst) to 10 (best)											

'''
#Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset and splitting it into matrices of independent & dependent variables 
missing_values = ('na','NA',' ','--','nan')
dataset = pd.read_csv("my2018_fuel_consumption_ratings.csv",na_values = missing_values,encoding = 'unicode_escape')
X = dataset.iloc[:, [4,11]].values
y = dataset.iloc[:,6].values

#Dropping unnecessary columns 
dataset.drop(['FUEL CONS:COMB (L/100 km)','FUEL CONS:COMB (mpg)'],axis = 1, inplace = True) #On preliminary analysis, combined values were average values of city and hwy values rounded off to the nearest integer. 
                                                                                          #These can be calculated easily to the exact value if and when needed.                                                                  

#Checking for null values
dataset.isnull().sum()

#Deleting unwanted rows containing missing values 
dataset.dropna(inplace = True)

#Exporting dataframe to csv file for input to Tableau for data visualization 
export_csv = dataset.to_csv (r'D:\Data Visualization Projects\Canada Fuel Consumption Ratings 2018\newfile.csv', index = False, header=True)

#Splitting the dataset into training set and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting random forest classification to the training set 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test results 
y_pred = classifier.predict(X_test)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#identifying all unique elements in each of the string columns; can be useful for identifying dummy columns created for categorical variables
dataset['MAKE'].unique()
dataset['VEHICLE CLASS'].unique()
dataset['TRANSMISSION'].unique()
dataset['FUEL TYPE'].unique()

#Encoding the categorical data 
dummy_df = pd.get_dummies(X[['VEHICLE CLASS','TRANSMISSION','FUEL TYPE']])


#






