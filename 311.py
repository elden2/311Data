# Using available data, specify a model that can most accurately classify a complaint. Noise complaints are classified by NYC as one of the following types:

# Collection Truck Noise 
# Noise 
# Noise - Commercial 
# Noise - Helicopter 
# Noise - House of Worship 
# Noise - Park 
# Noise - Street/Sidewalk 
# Noise - Vehicle
# NOTE: The Descriptor variable is a sub-category of Complaint.Type and therefore should be ignored when building your model.


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Lets load our small data:

data = pd.read_csv('small_data.csv')


#Looking at the data colums:
# get rid of columns I don't need

dropped_fields = ['Unique Key','Agency Name','Park Facility Name','School Name','School Number','School Code','School Phone Number','School Address','School City','School State','School Zip','School Region','School Not Found','School or Citywide Complaint','Vehicle Type','Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp','Bridge Highway Segment','Garage Lot Name','Ferry Direction','Ferry Terminal Name','Latitude','Longitude','Location']
data.drop(dropped_fields, axis=1, inplace = True)
# data.to_csv('data_dropped_columns.csv')

# data = pd.read_csv('data_dropped_columns.csv')

#Turn each day into a day of the week.


import datetime
def date(d):
	month, day, year = (int(x) for x in d.split('/'))
	return datetime.date(year,month, day).weekday()

def month(d):
	month, day, year = (int(x) for x in d.split('/'))
	return month

def year(d):
	month, day, year = (int(x) for x in d.split('/'))
	return year



data['Weekday'] = data['Created Date'].apply(date)
data['Year'] = data['Created Date'].apply(year)
data['Month'] = data['Created Date'].apply(month)


dummies = pd.get_dummies(data['Month'])
for m_number in dummies.columns:
	data['month_%s' % m_number] = dummies[m_number]

dummies = pd.get_dummies(data['Year'])
for m_number in dummies.columns:
	data['Year_%s' % m_number] = dummies[m_number]

dummies = pd.get_dummies(data['Weekday'])
for m_number in dummies.columns:
	data['Weekday_%s' % m_number] = dummies[m_number]

data = data.drop('Created Date', axis = 1)
data = data.drop('Weekday', axis = 1)
data = data.drop('Year', axis = 1)
data = data.drop('Month', axis = 1)
# ##


# #deal with agency
# data = data.drop('Closed Date', axis = 1) #i'm just assuming this won't matter.  I'll experiment later

dummies = pd.get_dummies(data['Agency'])
for agent in dummies.columns:
	data['agency_%s' % agent] = dummies[agent]


# Group Zip Codes in nearest 10
def zip(x):
	if np.isnan(x):
		return 10000
	else:
		return x//10
	return x//10

data['Incident Zip'] = data['Incident Zip'].apply(zip)

dummies = pd.get_dummies(data['Incident Zip'])
for a in dummies.columns:
	data['zip_%s' % a] = dummies[a]

data = data.drop('Incident Zip', axis = 1)


# Deal with Avenue/Street
def Ave_Street(name):
	if type(name) != str:
		return 'OTHER'
	if bool(re.search('AVENUE', name)):
		return 'AVENUE'
	if bool(re.search('STREET', name)):
		return 'STREET'	
	if bool(re.search('BROADWAY', name)):
		return 'BROADWAY'
	if bool(re.search('ROAD', name)):
		return 'ROAD'
	else:
		return 'OTHER'
	
data['Street Name'] = data['Street Name'].apply(Ave_Street)

dummies = pd.get_dummies(data['Street Name'])
for a in dummies.columns:
	data['Street_Type_%s' % a] = dummies[a]


data = data.drop('Street Name', axis = 1)


#Make City into a few key bigger sections
def C_name(name):
	cities = ['NEW YORK', 'BROOKLYN','BRONX', 'STATEN ISLAND']
	if name not in cities:
		return 'RARE'
	else:
		return name

data['City'] = data['City'].apply(C_name)


dummies = pd.get_dummies(data['City'])
for a in dummies.columns:
	data['City_%s' % a] = dummies[a]


data = data.drop('City', axis = 1)


#Make Landmarks Binary
def land(x):
	if type(x) == str:
		return 1
	else:
		return 0



data['Landmark'] = data['Landmark'].apply(land)


### Facility Types

dummies = pd.get_dummies(data['Facility Type'])
for a in dummies.columns:
	data['Facility_%s' % a] = dummies[a]


data = data.drop('Facility Type', axis = 1)

## Status


dummies = pd.get_dummies(data['Status'])
for a in dummies.columns:
	data['Status_%s' % a] = dummies[a]

data = data.drop('Status', axis = 1)


## Maybe i'll go back and use community board later - for now its too spread out, and im not sure what to do with it.  I'll check later

data = data.drop('Community Board', axis=1)

##address type
dummies = pd.get_dummies(data['Address Type'])
for a in dummies.columns:
	data['CommunityBoard_%s' % a] = dummies[a]

data = data.drop('Address Type', axis = 1)

##Location Type
dummies = pd.get_dummies(data['Location Type'])
for a in dummies.columns:
	data['AddressType_%s' % a] = dummies[a]

data = data.drop('Location Type', axis = 1)






drop = ['Incident Address', 'Descriptor', 'Closed Date', 'Cross Street 1', 'Cross Street 2','Intersection Street 1', 'Intersection Street 2', 'Due Date', 'Resolution Action Updated Date', 'X Coordinate (State Plane)', 'Y Coordinate (State Plane)', 'Park Borough'] 
data.drop(drop, axis=1, inplace = True)




#Just O.H.A. Borough
dummies = pd.get_dummies(data['Borough'])
for a in dummies.columns:
	data['Borough_%s' % a] = dummies[a]

data = data.drop('Borough', axis = 1)



#agency
dummies = pd.get_dummies(data['Agency'])
for a in dummies.columns:
	data['Agency_%s' % a] = dummies[a]

data = data.drop('Agency', axis = 1)

data.to_csv('final_data.csv')



target = data['Complaint Type']
train = data.drop(['Complaint Type'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.15, random_state=42)

#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


logreg = LogisticRegression()
#svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 3)
gaussian = GaussianNB()
perceptron = Perceptron()
#linear_svc = LinearSVC()
sgd = SGDClassifier()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)

classifiers = [logreg, svc, knn, gaussian,perceptron,linear_svc, sgd, decision_tree, random_forest]

