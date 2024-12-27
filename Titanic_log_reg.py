import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv('C:\\Users\\danil\\Downloads\\train.csv')
test_df = pd.read_csv('C:\\Users\\danil\\Downloads\\test.csv')
test_res = pd.read_csv('C:\\Users\\danil\\Downloads\\gender_submission.csv')

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
train_df['IsAlone'] = (train_df['SibSp'] + train_df['Parch'] == 0).astype(int)
train_df['IsMinor'] = (train_df['Age'] <= 18).astype(int)
train_df['FamSize'] = (train_df['SibSp'] + train_df['Parch'] + 1).astype(int)
train_df['Age'] = (train_df['Age'] - min(train_df['Age']))/(max(train_df['Age']) - min(train_df['Age']))
train_df['Fare'] = (train_df['Fare'] - min(train_df['Fare']))/(max(train_df['Fare']) - min(train_df['Fare']))

test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)
test_df = test_df.merge(test_res[['PassengerId', 'Survived']], on='PassengerId', how='left')
test_df['Fare'] = test_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))
test_df['IsAlone'] = (test_df['SibSp'] + test_df['Parch'] == 0).astype(int)
test_df['IsMinor'] = (test_df['Age'] <= 18).astype(int)
test_df['FamSize'] = (test_df['SibSp'] + test_df['Parch'] + 1).astype(int)
test_df['Age'] = (test_df['Age'] - min(test_df['Age']))/(max(test_df['Age']) - min(test_df['Age']))
test_df['Fare'] = (test_df['Fare'] - min(test_df['Fare']))/(max(test_df['Fare']) - min(test_df['Fare']))

combined_df = pd.concat([train_df, test_df], axis=0)
known_age = combined_df[combined_df['Age'].notnull()]
unknown_age = combined_df[combined_df['Age'].isnull()]

model_input = LinearRegression()
model_input.fit(known_age[['Pclass', 'Sex','SibSp', 'Parch', 'Fare','Embarked_Q','Embarked_S','IsAlone','IsMinor','FamSize']] ,known_age['Age'])

predicted_ages = model_input.predict(unknown_age[['Pclass', 'Sex','SibSp', 'Parch', 'Fare','Embarked_Q','Embarked_S','IsAlone','IsMinor','FamSize']])
predicted_ages = np.maximum(predicted_ages, 0)

combined_df.loc[combined_df['Age'].isnull(), 'Age'] = predicted_ages

train_df = combined_df.iloc[:len(train_df)]
test_df = combined_df.iloc[len(train_df):]



X = train_df.drop('Survived',axis=1).drop( 'Ticket',axis = 1 ).drop('Name', axis = 1).drop("Cabin",axis = 1)
y = train_df['Survived']

X_test = test_df.drop( 'Ticket',axis = 1 ).drop('Name', axis = 1).drop("Cabin",axis = 1).drop('Survived',axis=1)
y_test = test_df['Survived']

model = LogisticRegression(max_iter = 10000)
model.fit(X,y)
model.coef_[0]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

prediction = pd.DataFrame({'PassengerId': test_res['PassengerId'], 'Survived': y_pred})
prediction.to_csv('predict.csv', index=False)

print(sum(abs(y_test - y_pred)))
