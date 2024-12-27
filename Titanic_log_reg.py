import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('C:\\Users\\danil\\Downloads\\train.csv')

print(train_df.head())

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
train_df['Age'] = train_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

X = train_df.drop('Survived',axis=1).drop( 'Ticket',axis = 1 ).drop('Name', axis = 1).drop("Cabin",axis = 1)
y = train_df['Survived']

print(X.head())

model = LogisticRegression(max_iter = 1000)
model.fit(X,y)
model.coef_[0]

test_df = pd.read_csv('C:\\Users\\danil\\Downloads\\test.csv')
test_res = pd.read_csv('C:\\Users\\danil\\Downloads\\gender_submission.csv')

test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)
test_df['Age'] = test_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))
test_df['Fare'] = test_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))

X_test = test_df.drop( 'Ticket',axis = 1 ).drop('Name', axis = 1).drop("Cabin",axis = 1)
y_test = test_res['Survived']

print(X_test.head())

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

prediction = pd.DataFrame({'PassengerId': test_res['PassengerId'], 'Survived': y_pred})
prediction.to_csv('predict.csv', index=False)

#This yelds a 0.75837 points on kaggle
