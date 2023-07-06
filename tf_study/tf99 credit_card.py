import numpy as np
import pandas as pd

#1. 데이터
path = './data/creditcard/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')





print(train.columns, test.columns) #
# Index(['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
#        'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
#        'Is_Lead'],

# Index(['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
#        'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active'],
#       )

x = [['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = [['Is_Lead']]

# print(train.head(7), test.head(7))

print(train.shape) #(245725, 11)
print(test.shape) #(105312, 10)