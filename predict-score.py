import pandas as pd 

df = pd.read_csv("StudentPerformanceFactors.csv")
print(df.head())

#creating a X dataframe of numeric inputs columns 
X = df[["Attendance","Hours_Studied","Previous_Scores","Tutoring_Sessions"]] 
#creating a y dataframe of numeric output column
y = df["Exam_Score"]

print(X.head())
print()
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# checking the shape of the training and testing datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Now we make predictions 

predictions = model.predict(X_test)

print("Predicated Scores:",predictions[:5])              # printing first 5 predictions
print("Actual Scores:",y_test[:5].values)                # printing first 5 actual scores"

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)