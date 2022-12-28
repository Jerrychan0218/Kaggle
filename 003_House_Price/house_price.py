import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/JC/Desktop/Python/House_Prices/train.csv')
test = pd.read_csv('C:/Users/JC/Desktop/Python/House_Prices/test.csv')

print(train.head())
print(train.info())
print(train.describe())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(train.isna().sum())

cat_v = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
         'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond',
         'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond', 'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

for i in cat_v:
    sns.countplot(data = train, x = i, palette = 'Greens', fill = True)
    plt.show()

sns.histplot(data = train, x = 'SalePrice', kde= True)
plt.show()
sns.lineplot(data = train, x = 'YrSold', y = 'SalePrice')
plt.show()

# print(train.info())
# print(train[cat_v].isna().sum())

#fill 0
train['Alley'] = train['Alley'].fillna(0)
train['BsmtQual'] = train['BsmtQual'].fillna(0)
train['BsmtCond'] = train['BsmtCond'].fillna(0)
train['BsmtExposure'] = train['BsmtExposure'].fillna(0)
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(0)
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(0)
train['FireplaceQu'] = train['FireplaceQu'].fillna(0)
train['GarageType'] = train['GarageType'].fillna(0)
train['GarageFinish'] = train['GarageFinish'].fillna(0)
train['GarageQual'] = train['GarageQual'].fillna(0)
train['GarageCond'] = train['GarageCond'].fillna(0)
train['PoolQC'] = train['PoolQC'].fillna(0)
train['Fence'] = train['Fence'].fillna(0)
train['MiscFeature'] = train['MiscFeature'].fillna(0)

# fill with mode
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode().iloc[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode().iloc[0])


# num
# fill with mean
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
print(train.isna().sum())

# LabelEncoder
for i in cat_v:
    le = LabelEncoder()
    le.fit(train[i].astype(str))
    train[i] = le.transform(train[i].astype(str))

print(train.head())

# test
print(test.isna().sum())

# for cat_v
# fill with mode
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode().iloc[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode().iloc[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode().iloc[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode().iloc[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode().iloc[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode().iloc[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode().iloc[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode().iloc[0])

# fill with 0
test['Alley'] = test['Alley'].fillna(0)
test['BsmtQual'] = test['BsmtQual'].fillna(0)
test['BsmtCond'] = test['BsmtCond'].fillna(0)
test['BsmtExposure'] = test['BsmtExposure'].fillna(0)
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(0)
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(0)
test['FireplaceQu'] = test['FireplaceQu'].fillna(0)
test['GarageType'] = test['GarageType'].fillna(0)
test['GarageFinish'] = test['GarageFinish'].fillna(0)
test['GarageQual'] = test['GarageQual'].fillna(0)
test['GarageCond'] = test['GarageCond'].fillna(0)
test['PoolQC'] = test['PoolQC'].fillna(0)
test['Fence'] = test['Fence'].fillna(0)
test['MiscFeature'] = test['MiscFeature'].fillna(0)

# for num_v
# fill with mean
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean())
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean())
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

print(test[cat_v].isna().sum())

# LabelEncoder
for i in cat_v:
    le = LabelEncoder()
    le.fit(test[i].astype(str))
    test[i] = le.transform(test[i].astype(str))

# Model Prediction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error as MSE

test1 = test.drop('Id', axis = 1)
X = train.drop(['SalePrice', 'Id'], axis = 1)
y = train['SalePrice']

kf = KFold(n_splits = 5, shuffle = True, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=10)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_cv = - cross_val_score(lr, X_train, y_train, cv = kf, scoring = 'neg_mean_squared_error')
print(lr_cv.mean())

ypred = lr.predict(X_test)
print(MSE(y_test, ypred))

ypredlr = lr.predict(test1)
print(ypredlr)

output = pd.DataFrame({'Id':test['Id'], 'SalePrice':ypredlr})
output.to_csv('submission.csv', index = False)

print(output.head())




