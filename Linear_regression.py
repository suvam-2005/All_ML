import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cars.csv")

print("Dataset Preview:")
print(df.head())
print("\nDataset Shape:", df.shape)

X = df[['Volume', 'Weight']]   
y = df['CO2']   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Parameters:")
print("Intercept:", model.intercept_)
print("Coefficient for Volume:", model.coef_[0])
print("Coefficient for Weight:", model.coef_[1])

y_pred = model.predict(X_test)

print("\nActual CO2 values:")
print(y_test.values)

print("\nPredicted CO2 values:")
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

new_car = [[1300, 2300]]
predicted_co2 = model.predict(new_car)

print("\nPrediction for New Car:")
print("Volume = 1300, Weight = 2300")
print("Predicted CO2 Emission:", predicted_co2[0])
