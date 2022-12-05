import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
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

train['Cabin_deck'] = train['Cabin'].str.split('/').str[0]
train['Cabin_num'] = train['Cabin'].str.split('/').str[1]
train['Cabin_side'] = train['Cabin'].str.split('/').str[2]


print(train.head)

cat_v = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported', 'Cabin_deck', 'Cabin_side']
num_v = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num']

train[cat_v] = train[cat_v].astype('category')

train[num_v] = train[num_v].astype('float')

print(train.info())

# train Graph
for i in cat_v:
    print(train[i].value_counts())
    sns.countplot(data = train, x = i, palette = 'Blues')
    plt.show()

for j in num_v:
    print('The mean of {}:'.format(j), train[j].mean())
    sns.kdeplot(data = train, x = j, palette = 'Greens', hue = 'Transported', fill = True)
    plt.show()


# Null cat
train['HomePlanet'] = train['HomePlanet'].fillna('Earth')
train['CryoSleep'] = train['CryoSleep'].fillna(False)
train['Destination'] = train['Destination'].fillna('TRAPPIST-1e')
train['VIP'] = train['VIP'].fillna(False)
train['Cabin_deck'] = train['Cabin_deck'].fillna('F')
train['Cabin_side'] = train['Cabin_side'].fillna('S')


# Null num
for g in num_v:
    train[g] = train[g].fillna(train[g].mean())

train_trans = train['Transported']
train = train.drop(['Cabin', 'Name', 'Transported', 'PassengerId'], axis = 1)
print(train.isna().sum())

train = pd.get_dummies(train, drop_first = True)
train['Transported'] = train_trans
print(train.head())

#test
print(test.describe())
print(test.info())
print(test.isna().sum())

test['Cabin_deck'] = test['Cabin'].str.split('/').str[0]
test['Cabin_num'] = test['Cabin'].str.split('/').str[1]
test['Cabin_side'] = test['Cabin'].str.split('/').str[2]


print(test.head)

cat_v_t = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_deck', 'Cabin_side']
num_v_t = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num']

for e in cat_v_t:
    test[e] = test[e].astype('category')

for r in num_v_t:
    test[r] = test[r].astype('float')

print(test.info())

# test Graph
for u in cat_v_t:
    print(test[u].value_counts())
    sns.countplot(data = test, x = u, palette = 'rocket')
    plt.show()

for o in num_v_t:
    print('The mean of {}:'.format(o), test[o].mean())
    sns.kdeplot(data = test, x = o, palette = 'crest', fill = True)
    plt.show()


#Null cat
test['HomePlanet'] = test['HomePlanet'].fillna('Earth')
test['CryoSleep'] = test['CryoSleep'].fillna(False)
test['Destination'] = test['Destination'].fillna('TRAPPIST-1e')
test['VIP'] = test['VIP'].fillna(False)
test['Cabin_deck'] = test['Cabin_deck'].fillna('F')
test['Cabin_side'] = test['Cabin_side'].fillna('S')


# Null num
for p in num_v:
    test[p] = test[p].fillna(test[p].mean())

test_pass = test['PassengerId']
test = test.drop(['Cabin', 'Name', 'PassengerId'], axis = 1)
print(test.isna().sum())

test = pd.get_dummies(test, drop_first = True)
print(test.head())

# ML
stand = StandardScaler()
train[num_v] = stand.fit_transform(train[num_v])
train = train.astype(int)

X = train.drop('Transported', axis = 1)
y = train['Transported']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

logist = LogisticRegression(random_state=25)
log_fit = logist.fit(X_train, y_train)
y_pred1 = logist.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
print(acc1)

dt = DecisionTreeClassifier(max_depth=6, random_state=25)
dt_fit = dt.fit(X_train, y_train)
y_pred2 = dt.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print(acc2)

rfc = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 0.12, random_state = 25)
rfc_fit = rfc.fit(X_train, y_train)
y_pred3 = rfc.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
print(acc3)

test = test.astype(int)
log_pred_fin = dt.predict(test)
log_pred_fin = log_pred_fin.astype(bool)

output = pd.DataFrame({'PassengerId':test_pass, 'Transported':log_pred_fin})
output.to_csv('submission.csv', index = False)

print(output.head())





