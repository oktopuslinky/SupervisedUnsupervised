import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



df = pd.read_csv("austin_weather2.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

print(df)


X = df[['DayOfYear', 'Month', 'Day', 'Year']]

y = df['TempAvgF']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = mean_squared_error(y_test, predictions)

df2 = pd.DataFrame(X_test, columns=['DayOfYear', 'Month', 'Day', 'Year'])
df2['Prediction'] = predictions
df2['Actual Value'] = y_test

print(df2)
print("\nThe MSE was:", score)
