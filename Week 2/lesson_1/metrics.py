from sklearn.neighbors import KNeighborsRegressor
from numpy import mean, linspace
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

data = load_boston()

result = data.target
scale_prop = scale(data.data)

kf = KFold(n=result.size, n_folds = 5, shuffle = True, random_state = 42)

accuracy = {}
for i in linspace(start=1,stop=10,num=200):
	regressor = KNeighborsRegressor(weights='distance',p=i)
	score = cross_val_score(regressor, scale_prop, result, scoring='mean_squared_error', cv=kf)
	accuracy[i] = mean(score)

key_accuracy = max(accuracy, key = accuracy.get)
max_accuracy = accuracy[key_accuracy]

print key_accuracy
