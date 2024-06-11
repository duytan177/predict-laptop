import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('./College.csv', delimiter=",")

# Question 1(Regression)
# a. Perform a multiple linear regression using all the features to predict Apps.
X = data.drop(['Apps', 'Private', 'Unnamed: 0'], axis=1) # Loại bỏ cột 'Apps' và 'Private' và 'Unnamed' khỏi ma trận đặc trưng
y = data['Apps'] # Biến phụ thuộc
linear_model = LinearRegression()
linear_model.fit(X, y)
print("Câu hỏi 1a: Hệ số hồi quy tuyến tính:", linear_model.coef_)

# b. Perform a polynomial regression using a number of meaningful features of the original data to predict Apps.
poly_features = ['Accept', 'Enroll', 'F.Undergrad']
X_poly = data[poly_features]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_poly)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
print("Câu hỏi 1b: Hệ số hồi quy đa thức:", poly_model.coef_)

# Question 2 (K nearest neighbor)
# a. Create a binary attribute Apps01 that contains a 1 if Apps contains a value equal to or above its median,
# and a 0 if Apps contains a value below its median. Create a single data set d containing both Apps01 and
# the other College features. Split the data into d.train training set and d.test test set 80:20.
median = data['Apps'].median()
data['Apps01'] = np.where(data['Apps'] >= median, 1, 0)
d = data.drop(['Apps', 'Private','Unnamed: 0'], axis=1) # Loại bỏ cột 'Apps' và 'Private' và 'Unnamed' khỏi ma trận đặc trưng
X_train, X_test, y_train, y_test = train_test_split(d.drop('Apps01', axis=1), d['Apps01'], test_size=0.2, random_state=42)

# b. Perform k-NN with several values of k in order to predict Apps01 using all the features
k_values = [3, 5, 7]
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Câu hỏi 2b: Độ chính xác của k-NN với k =", k, ":", accuracy)

#c. Display the comparison of validation error rate for different values of k = 11.

error_rates = []

# Thử các giá trị k = 11 khác nhau
for k in range(1, 11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rates.append(1 - accuracy)

# Biểu đồ đường tỷ lệ lỗi xác thực cho các giá trị k khác nhau
plt.plot(range(1, 21), error_rates, marker='o')
plt.title('Tỷ lệ lỗi xác thực cho các giá trị k khác nhau')
plt.xlabel('Giá trị k')
plt.ylabel('Tỷ lệ lỗi xác thực')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()
