import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train
print(train.head())
print(train.describe())
print(train.info())

train['Cabin_Deck'] = train['Cabin'].str.split('/').str[0]
train['Cabin_Num'] = train['Cabin'].str.split('/').str[1]
train['Cabin_Side'] = train['Cabin'].str.split('/').str[2]

print(train.head)

cat_v = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']
num_v = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_Num']
num_v_2 = ['Age']

train[cat_v] = train[cat_v].astype('category')

train[num_v] = train[num_v].astype('float')

train[num_v_2] = train[num_v_2].astype('float')

print(train.info())

#train Graph
for i in cat_v:
    print(train[i].value_counts())
    sns.countplot(data = train, x = i, palette = 'Blues')
    plt.show()

for j in num_v:
    print('The mean of {}:'.format(j), train[j].mean())
    sns.kdeplot(data = train, x = j, palette = 'Greens', hue = 'Transported', fill = True)
    plt.show()

for h in num_v_2:
    print('The mean of {}:'.format(h), train[h].mean())
    sns.kdeplot(data = train, x = h, palette = 'Greens', hue = 'Transported', fill = True)
    plt.show()

# Null cat
train['HomePlanet'] = train['HomePlanet'].fillna('Earth')
train['CryoSleep'] = train['CryoSleep'].fillna(False)
train['Destination'] = train['Destination'].fillna('TRAPPIST-1e')
train['VIP'] = train['VIP'].fillna(False)
train['Cabin_Deck'] = train['Cabin_Deck'].fillna('F')
train['Cabin_Side'] = train['Cabin_Side'].fillna('S')

for i in cat_v:
    le = LabelEncoder()
    le.fit(train[i].astype(str))
    train[i] = le.transform(train[i].astype(str))

train['Transported'] = train['Transported'].astype(int)

print(train.head())

# Null num
for g in num_v:
    train[g] = train[g].fillna(0)

for l in num_v_2:
    train[l] = train[l].fillna(train[l].median())

# drop data
train = train.drop(['Name', 'PassengerId'], axis = 1)

print(train.head())

#test
print(test.describe())
print(test.info())
print(test.isna().sum())

print(test.head)

test['Cabin_Deck'] = test['Cabin'].str.split('/').str[0]
test['Cabin_Num'] = test['Cabin'].str.split('/').str[1]
test['Cabin_Side'] = test['Cabin'].str.split('/').str[2]

test[cat_v] = test[cat_v].astype('category')

test[num_v] = test[num_v].astype('float')

test[num_v_2] = test[num_v_2].astype('float')

print(test.info())

# test Graph
for u in cat_v:
    print(test[u].value_counts())
    sns.countplot(data = test, x = u, palette = 'rocket')
    plt.show()

for o in num_v:
    print('The mean of {}:'.format(o), test[o].mean())
    sns.kdeplot(data = test, x = o, palette = 'rocket', fill = True)
    plt.show()

for p in num_v_2:
    print('The mean of {}:'.format(p), test[p].mean())
    sns.kdeplot(data = test, x = p, palette = 'rocket', fill = True)
    plt.show()

#Null cat
test['HomePlanet'] = test['HomePlanet'].fillna('Earth')
test['CryoSleep'] = test['CryoSleep'].fillna(False)
test['Destination'] = test['Destination'].fillna('TRAPPIST-1e')
test['VIP'] = test['VIP'].fillna(False)
test['Cabin_Deck'] = test['Cabin_Deck'].fillna('F')
test['Cabin_Side'] = test['Cabin_Side'].fillna('S')

# Null num
for p in num_v:
    test[p] = test[p].fillna(0)

for q in num_v_2:
    test[q] = test[q].fillna(train[q].median())

test_pass = test['PassengerId']
test = test.drop(['Name', 'PassengerId'], axis = 1)
print(test.isna().sum())

for i in cat_v:
    le.fit(test[i].astype(str))
    test[i] = le.transform(test[i].astype(str))

test = test.drop('Cabin',axis = 1)
print(test.head())

# ML
stand = StandardScaler()
train[num_v_2] = stand.fit_transform(train[num_v_2])
train = train.drop('Cabin', axis = 1)
train = train.astype(int)

X = train.drop('Transported', axis = 1)
y = train['Transported']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

dt = DecisionTreeClassifier(max_depth=6, random_state=15)
dt_fit = dt.fit(X_train, y_train)
y_pred2 = dt.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print(acc2)

rfc = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 0.12, random_state = 15)
rfc_fit = rfc.fit(X_train, y_train)
y_pred3 = rfc.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
print(acc3)

test = test.astype(int)
dt_pred_fin = dt.predict(test)
dt_pred_fin = dt_pred_fin.astype(bool)

output = pd.DataFrame({'PassengerId':test_pass, 'Transported':dt_pred_fin})
output.to_csv('submission.csv', index = False)

print(output.head())





