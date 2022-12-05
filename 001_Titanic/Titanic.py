#Load module
from pydoc import describe
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#read csv
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
result = pd.read_csv('gender_submission.csv')

#data description
print(train_data.info())
print(train_data.describe())
print(train_data.head())

#Missing data
print(train_data.isna().sum())

#train missing

# for age groupby sex
m_age_mean = train_data.groupby('Sex')['Age'].mean()['male']
f_age_mean = train_data.groupby('Sex')['Age'].mean()['female']

def fillna_age(sex_age):
    sex = sex_age[0]
    age = sex_age[1]

    if pd.isnull(age):
        if 'male' in sex:
            return m_age_mean
        if 'female' in sex:
            return f_age_mean
    else:
        return age

train_data['Age'] = train_data[['Sex', 'Age']].apply(fillna_age, axis = 1)

#for gender
sex={'male':0, 'female':1}
train_data = train_data.replace({'Sex':sex})

#for cabin
train_data.drop('Cabin', axis = 1, inplace = True)

#for Embarked
print(train_data['Embarked'].mode())
train_data['Embarked'].fillna('S', inplace = True)

#plot catagory variable and numeric
print(train_data.isna().sum())
print(train_data['Survived'].value_counts())
cate = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
num = ['Age', 'Fare']

for i in cate:
    catplot = sns.countplot(train_data[i])
    plt.title(i)
    plt.show()

for a in num:
    numplot = sns.histplot(train_data[a])
    plt.title(a)
    plt.show()

train_data.drop(['Name', 'Ticket','Embarked'], axis = 1, inplace = True)
print(train_data.head())

#Test data
print(test_data.info())
print(test_data.describe())
print(test_data.head())

#Missing data
print(test_data.isna().sum())

#Fill age
f_age_test = test_data.groupby('Sex')['Age'].mean()[0]
m_age_test = test_data.groupby('Sex')['Age'].mean()[1]

test_data['Age'] = test_data[['Sex', 'Age']].apply(fillna_age, axis = 1)

#for sex
sex={'male':0, 'female':1}
test_data = test_data.replace({'Sex':sex})

#fill fare
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)

#drop variable which are not use for prediction
test_data.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], axis = 1, inplace = True)

#set variable list that decided to use for prediction without passengerid
feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
print(test_data.isna().sum())

#load train_dummy variable
cate_train = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch']
train_data[cate_train] = train_data[cate_train].astype('category')
train_data = pd.get_dummies(train_data, drop_first = True)

#load test_dummy variable
cate_test = ['Pclass', 'Sex', 'SibSp', 'Parch']
test_data[cate_test] = test_data[cate_test].astype('category')
test_data = pd.get_dummies(test_data, drop_first = True)
print(test_data)

#add a column Parch_9 to train_data
train_data['Parch_9'] = 0
print(train_data)

#Standardize
stand = StandardScaler()
train_data[['Age', 'Fare']] = stand.fit_transform(train_data[['Age', 'Fare']])
# X_test[['Age', 'Fare']] = stand.transform(X_test[['Age', 'Fare']])
X = train_data.drop('Survived_1', axis = 1)
y = train_data['Survived_1']

#split train_test data
X_train, X_ex, y_train, y_ex = train_test_split(X, y, test_size=0.3, random_state=10,stratify = y)

#Load model LogisticReg
logit = LogisticRegression(random_state=10)
logit.fit(X_train, y_train)
y_pred1 = logit.predict(X_ex)
acc1 = accuracy_score(y_ex, y_pred1)
print(acc1)

#Load model DecisionTree
dt = DecisionTreeClassifier(max_depth=6, random_state=10)
dt.fit(X_train, y_train)
y_pred2 = dt.predict(X_ex)
acc2 = accuracy_score(y_ex, y_pred2)
print(acc2)

#Load model Ada
ada = AdaBoostClassifier(base_estimator = dt, n_estimators= 1000, random_state = 10)
ada.fit(X_train, y_train)
y_pred3 = ada.predict(X_ex)
acc3 = accuracy_score(y_ex, y_pred3)
print(acc3)

#Prediction
pred_f = dt.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':pred_f})
output.to_csv('submission.csv', index = False)




