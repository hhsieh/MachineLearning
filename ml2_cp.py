import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

data = []
with open('yelp_data.json') as f:
    for line in f:
        data.append(json.loads(line))

city = []
for i in range(len(data)):
    city.append(data[i].values()[0])

stars = []
for i in range(len(data)):
    stars.append(data[i].values()[10])

dd = pd.DataFrame({"city":city, "star":stars})

city_mean = dd.groupby("city").mean()
#print(city_mean['star'][0]) # 0 is the index of the city
#print(np.array(city_mean))
#print(city_mean["star"]["Madison"])


X = city
y = stars

cityid_, idx = np.unique(X, return_inverse = True)
cntX = map(list(cityid_).index, X)  ## map all city inputs and generate corresponding index values
#print(cntX)

meanclasses_, meanindices = np.unique(y, return_inverse = True)
df = pd.DataFrame({"X":X, "y": y})
#print(df)
mean_ = df.groupby(['X']).mean()
print(mean_['y']['Waterloo']) ## test in the model if this return to the right prediction star with no error


class MeanClassifier(BaseEstimator, ClassifierMixin):
    def __inif__(self):
        self.cityid_ = []
        self.cntX = []

    def X3(self, X):
        self.cityid_, idx = np.unique(X, return_inverse = True)
        self.cntX = map(list(self.cityid_).index, X)
        return self.cntX
    
    def fit(self, X, y):
        self.meanclasses_, meanindicies = np.unique(y, return_inverse = True)
        self.cityid_, idx = np.unique(X, return_inverse = True)
        self.df = pd.DataFrame({"X": X, "y": y})
        self.mean_ = self.df.groupby(['X']).mean()
        
    def predict(self, X):
        return self.mean_.ix[X].values ## using sklearn requires all X inputs


B = MeanClassifier()

asncityid = city

B.fit(asncityid, stars)
pred = B.predict(asncityid[120])
print(pred)


      











##city model
class city_model():


    def __init__(self, X, y):
        self.cities = X  # a list of cities
        self.stars = y # a list of stars assocaited with the cities
 
    def fit(self, X, y):
        self.data = pd.DataFrame({"city": self.cities, "star": self.stars})
        self.mean = self.data.groupby(self.data['city']).mean()
        return self
        
    def predict(self, X):
        return self.mean['star'][X]

CITIES = ["Taipei", "Shanghei", "New York", "Taipei"]
STARS = [2,3,3,1]
        
A = city_model(CITIES, STARS)
A.fit(CITIES, STARS)
pred = A.predict("Taipei")
#print(pred)


## city model based on sklearn
#class MeanClassifier(BaseEstimator, ClassifierMixin):
#    def __init__(self):
#        self.cityid_ = []
#        self.cntX = []
  
