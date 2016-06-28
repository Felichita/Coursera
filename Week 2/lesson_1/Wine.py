from numpy import mean
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

nameTitle = ["PClass",
"Alcohol",
"Malic acid",
"Ash",
"Alcalinity of ash", 
"Magnesium", 
"Total phenols",
"Flavanoids",
"Nonflavanoid phenols", 
"Proanthocyanins",
"Color intensity", 
"Hue",
"OD280/OD315 of diluted wines",
"Proline"]
data = pd.read_csv("wine.txt", names = nameTitle)
classData = data["PClass"]
cleanData = data[data.columns[1:]]

kf = KFold(n=classData.size, n_folds = 5, shuffle = True, random_state = 42)

accuracy = {}

for i in range(1,51):
	classifier = KNeighborsClassifier(i)
	score = cross_val_score(estimator=classifier, X=cleanData, y=classData, cv=kf)
	accuracy[i] = mean(score)
key_accuracy = max(accuracy, key = accuracy.get)
max_accuracy = accuracy[key_accuracy]
print key_accuracy, max_accuracy


accuracy2 = {}
scale_prop = scale(cleanData)

for i in range(1,51):
	classifier = KNeighborsClassifier(i)
	score = cross_val_score(estimator=classifier, X=scale_prop, y=classData, cv=kf)
	accuracy[i] = mean(score)
key_accuracy = max(accuracy, key = accuracy.get)
max_accuracy = accuracy[key_accuracy]

print key_accuracy, max_accuracy




 