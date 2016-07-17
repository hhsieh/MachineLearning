## for city_model, suggestion from Kee-Chul Chang
## You should be able to do everything with pandas, using groupby and aggregate. It should be in the fit function of your classifier. If you want to adhere to scikit-learn behavior, your predict method should accept a list of city names and return a list of predicted star values.

       

import json

#read in data
data = []
with open('yelp_data.json') as f:
    for line in f:
        data.append(json.loads(line))
#print(data[:1])
        
#len(data) #37938
#print(data[0]) #take a look of the basic structure of the jason data

# collect variables
attr = []      
for i in range(len(data[0].keys())):
	attr.append(data[0].keys()[i])

#print(attr)

#attr[10] == "stars", the target in the model
#attr[0] == "city", for the city model


city = []
for i in range(len(data)):
	city.append(data[i].values()[0])

	
stars = []
for i in range(len(data)):
	stars.append(data[i].values()[10])
	
bid = []
for i in range(len(data)):
	bid.append(data[i].values()[5])	
	
address = []
for i in range(len(data)):
	address.append(data[i].values()[6])		

review_count = []
for i in range(len(data)):
	review_count.append(data[i].values()[1])	

name = []
for i in range(len(data)):
	name.append(data[i].values()[2])

neighborhoods = []
for i in range(len(data)):
	neighborhoods.append(data[i].values()[3])

type = []
for i in range(len(data)):
	type.append(data[i].values()[4])

state =[]
for i in range(len(data)):
	state.append(data[i].values()[8])
	
longitude = []
for i in range(len(data)):
	longitude.append(data[i].values()[9])

latitude = []
for i in range(len(data)):
	latitude.append(data[i].values()[11])

attributes = []
for i in range(len(data)):
	attributes.append(data[i].values()[12])

open = []
for i in range(len(data)):
	open.append(data[i].values()[13])

categories = []
for i in range(len(data)):
	categories.append(data[i].values()[14])
print([categories[:1][-1]])  #[-1] gives the most general categories, omitting the sub-categories		
print([categories[1:2][-1]])	
print([categories[3:4][-1]])
# to count how many cities there are in the list
unique = set(city)
len(unique) #167
#print(unique)	


import pandas as pd
dd = pd.DataFrame({'city': city, 'stars' : stars})
#print(dd)

city_count = pd.value_counts(dd['city'].values, sort = False)
#print(city_count)
city_mean_stars = dd.groupby(['city']).mean()
#print(city_mean_stars)
#print(len(city_mean_stars))

ladf = pd.DataFrame({'city': city, 'lon': longitude, 'lat': latitude, 'stars': stars})
#print(ladf)

######################################
## below works the same as city_count"
#from collections import Counter
#city_count = Counter(city).items()
#print(len(city_count))
#print(city_count[0][0])
#print(city_count[0][1])
#cityname = []
#for i in range(len(city_count)):
#	cityname.append(city_count[i][0])
#print(cityname)
#citycount = []
#for i in range(len(city_count)):
#	citycount.append(city_count[i][1])
#print(citycount)
#citycount = pd.DataFrame({'cityname': cityname , 'count': citycount})
#print(citycount)	

###################################
## estimator (assign the max value to each x of the test data)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class MajorityClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self):
		self.cityid_ = []
		self.cntX = []
		pass
		
	def X2(self, X):
		self.cityid_, idx = np.unique(X, return_inverse = True)
		self.cntX = map(list(self.cityid_).index, X)
		return self.cntX
		
	def fit(self, X, y):
		self.classes_, indices = np.unique(y*2, return_inverse = True)
		#self.cityid_, idx = np.unique(X, return_inverse = True) 
		#print(self.cityid_)    	
		self.majority_ = np.argmax(np.bincount(indices))
		return self

	def predict(self, X):
		#self.X2 = map(list(self.cityid_).index, X)
		return np.repeat(self.classes_[self.majority_]/2, len(X))
		
		

A = MajorityClassifier()

asncityid = A.X2(dd['city'])

A.fit(asncityid, dd['stars'])
pred = A.predict(asncityid)
#print(pred)

print(A.score(dd['stars'], pred))		
# take data unbalance into consideration while splitting data to training and testing data for cross-validation


## city model

from sklearn.base import RegressorMixin, BaseEstimator
class MeanClassifier(BaseEstimator, RegressorMixin):
	def __init__(self):
		self.cityid_ = []
		self.cntX = []

	def X3(self, X):
		self.cityid_, idx = np.unique(X, return_inverse = True)
		self.cntX = map(list(self.cityid_).index, X)
		return self.cntX
	
	def fit(self, X, y):
		self.meanclasses_, meanindices = np.unique(y, return_inverse = True) #this is mean stars 
		self.cityid_, idx = np.unique(X, return_inverse = True) #this is city
		#print(X)
		self.df = pd.DataFrame({"X": X, "y": y}) #collect city and mean stars and make a pandas dataframe
		#print(self.df)
		#self.mean_ = np.array(self.df.groupby(['X']).mean()) #generate one mean star for each city
		self.means_ = self.df.groupby(['X']).mean()
                #self.m1, self.m2 = np.unique(self.mean_, return_inverse = True)
		#print(type(self.mean_))
		#print(self.group_)
		#print(self.mean_)
		return self
				
	def predict(self, X):
		#print(list(self.cityid_).index(X))
		#print(float(self.mean_[list(self.cityid_).index(X)]))
		#z = self.cityid_  
		#return float(self.mean_.ix[X].values) ##self.mean_[self.cityid_][0]
                return list(self.means_.loc[X, 'y'])       

B = MeanClassifier()

newcityid = B.X3(dd['city'])

B.fit(newcityid, dd['stars'])

Bpred = B.predict(newcityid)
print(Bpred[:5])

bbstars = map(float, list(dd['stars']))
print(bbstars[:5])
#print(list(dd['stars']))
print(B.score(newcityid, dd['stars']))


## lat_long_model
#class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor

class LonLatClassifier(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.knn = KNeighborsRegressor(n_neighbors = 13)
        self.knn.fit(X, y)
        return self        

    def predict(self, X):
        return self.knn.predict(X)



C = LonLatClassifier()
data = pd.DataFrame({"longitude": longitude, "latitude": latitude, "star": stars})
print(data.columns.values)

#sturcture data like this to be read as X in the mode
#[[data.loc[i][1], data.loc[i][2]] for i in range(4)]

#x_columns = ["longitude", "latitude"]
#y_column = ["star"]

LonLat = [[data.loc[i][0], data.loc[i][1]] for i in range(len(data))]
#print(len(LonLat))
stars = [data.loc[i][2] for i in range(len(data))]
#print(len(stars))

Cfit = C.fit(LonLat, stars)  ##not giving newcityid but longitudes and latitudes
#print(Cfit)

Cpred = C.predict(LonLat[:10])
print(Cpred)

print(C.score(LonLat, stars))
