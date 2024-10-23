import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv("loan_approval_dataset.csv")

# print(data.head())
# print(data.info())

X = data.drop(columns=[' loan_status', ' education',' self_employed','loan_id'])
Y = data[' loan_status']
print(X)

# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

LabelEncoder_Y=LabelEncoder()
Y_train=LabelEncoder_Y.fit_transform(Y_train)

# print(Y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lm = LinearRegression()
lm.fit(X_train, Y_train)
predictions= lm.predict(X_test)
# train_score = lm.score(X_train, Y_train)
# print(train_score)


pickle.dump(lm, open('./model.sav','wb'))
