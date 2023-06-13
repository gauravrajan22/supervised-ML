import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("Sparks.txt", sep=' ')

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)

# Predict score for hours=9.25
hours_value = 9.25
hours_value = [[hours_value]]
score_pred = model.predict(hours_value)
print("Predicted Score for 9.25 hours:", score_pred)

# Plotting
y_pred = model.predict(X)
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.scatter(hours_value, score_pred, color="green", label="Predicted for 9.25 hours")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Linear Regression")
plt.legend()

# Display the plot
plt.show()
