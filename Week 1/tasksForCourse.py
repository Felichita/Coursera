import pandas
import re
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')
# First Task - count male and female
print data['Sex'].value_counts()
# Second Task - in percent who survived on Titanic
survived = data['Survived'].value_counts()
allPassenger = data.count()[0]
percent = float(survived[1])/float(allPassenger)
print percent * 100
# Third Task - count who from 1 Pclass 
Pclass = data['Pclass'].value_counts()[1]
percent = float(Pclass)/float(allPassenger)
print percent * 100
# Four Task - midle and median Age
age = data['Age']
print age.mean() , age.median()
# Five Task - Search correlation Pirson
SibSp = data['SibSp']
Parch = data['Parch']
print SibSp.corr(Parch)
# Six Task - Favorite female name on Titanic
femaleName = data[data['Sex'] == 'female']['Name']

def slice (name):
	if re.search("Miss", name):
		return name.split(" ")[2]
	elif re.search("Mrs",name):
		result = re.search("\(([A-z]+)",name)
		if result:
			return result.group(1)
		else:
			return 0
favoriteName = femaleName.map(lambda name: slice(name)).value_counts()
print favoriteName

# statement-importance

dataImportance = data[['Survived','Pclass','Fare', 'Age', 'Sex']].dropna()
survived = dataImportance['Survived']
dataImportance = dataImportance[['Pclass','Fare', 'Age', 'Sex']].replace(to_replace=['male', 'female'], value=[1, 0])

clf = DecisionTreeClassifier(random_state=241)
clf.fit(dataImportance,survived)
importances = clf.feature_importances_
print importances