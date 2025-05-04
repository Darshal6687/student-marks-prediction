import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\darsh\OneDrive\Desktop\FSDS\4_APRIL\FSDS_24_04\student mark price prediction\student_info.csv")

df.head()

df.shape

# visualize the data to gain insights

df.info()

df.describe()

plt.scatter(x = df.study_hours,y = df.student_marks )
plt.xlabel('Students Study Hours')
plt.ylabel('Students Marks')
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()


# Data Cleaning
df.isnull().sum()

df.mean()

df2 = df.fillna(df.mean())

df2.isnull().sum()


# Split the dataset

X = df2.drop("student_marks",axis="columns")
y = df2.drop("study_hours",axis="columns")
print("shape of X =" , X.shape)
print("Shape of y =" , y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
print("shape of X_train = ", X_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", X_test.shape)
print("shape of y_test = ", y_test.shape)


# Select the model & Train it
# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)

lr.coef_

lr.intercept_

# m = 3.93
# c = 50.45

lr.predict([[4]])[0][0].round(2)

y_pred = lr.predict(X_test)
y_pred

pd.DataFrame(np.c_[X_test,y_test,y_pred],columns = ["study_hours","student_marks_original","student_marks_predicted"])


#Fine-tune model

lr.score(X_test, y_test)

plt.scatter(X_train,y_train)
plt.show()

plt.scatter(X_test,y_test)
plt.plot(X_train, lr.predict(X_train),color = "r")
plt.show()

import joblib
joblib.dump(lr, "Student_mark_predictor.pkl")


model = joblib.load("Student_mark_predictor.pkl")


