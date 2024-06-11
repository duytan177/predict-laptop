import drive
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import display,image
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
def main():
    data = pd.read_csv("../data/Humidity_Temp_Prediction.csv",delimiter=";")
    #change  type object data['date_time'] to type date_time
    data['date_time'] = pd.to_datetime(data['date_time'],format="%Y-%m-%d %H:%M:%S")

    #get hour,minute,day of week,month from data['date_time]
    data['Hour'] = data['date_time'].dt.hour
    data['Minute'] = data['date_time'].dt.minute
    data['Day_of_week'] = data['date_time'].dt.day_name()
    data['Month'] = data['date_time'].dt.month
    data['Minute'] = data['Minute'] + data['Hour'] * 60

    NumDescripticeStats = data.describe(include=[np.number])
    CatDescripticeStats = data.describe(exclude=[np.number])

    print(data.head())
    print(data.tail())
    print(data.shape)
    print(data.info())

    #định lượng
    #std do lech || 50% trung vi || trung bình với trung vị gần = nhau
    print(NumDescripticeStats)
    #25 q1 50 q2 75 q3
    print(CatDescripticeStats)

    feature_x = "Minute"
    feature_y = "temp"
    # GroupByDF = data.loc[:, [feature_x, feature_y]].groupby(feature_x, as_index=False).mean() # vì temp nhiều cột nên tính trung bình nó ra bằng mean()
    # print(GroupByDF)
    # plt.figure(figsize=(15,7))                                                                    #vẽ biểu đồ theo chiê dài dọc
    # plt.scatter(GroupByDF[feature_x],GroupByDF[feature_y])                                      # biểu đồ theo dạng hình những điểm tròn cột
    # plt.title(f"{feature_x} vs {feature_y} Plotting")
    # plt.xlabel(feature_x,size=15)
    # plt.ylabel(feature_y,size=15)
    # plt.savefig(f'{feature_x} vs {feature_y} Plotting.jpg')                           # save lại hình ở folder.
    # plt.show()


    DataX = data.loc[:, [feature_x, feature_y]].groupby(feature_x, as_index=False).mean()
    print(DataX)
    m1 = DataX[[feature_x]].values
    m2 = DataX[[feature_y]].values
    print(m1)
    print(m2)

    #build model với 4 thuộc tính
    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(m1)
    print(x_poly[:5])

    lin_reg = LinearRegression()
    lin_reg.fit(x_poly,m2)

    m2_predict = lin_reg.predict(poly_reg.fit_transform(m1))
    print(m2_predict)
    print(m2)
    print(x_poly)
    print(NumDescripticeStats.describe())


    # Sử dụng matplotlib để vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.scatter(x_poly[:, 1], m2, color='blue', label='Actual')  # Dữ liệu thực tế
    plt.plot(x_poly[:, 1], m2_predict, color='red', label='Predicted')  # Dự đoán từ mô hình
    plt.title('Actual vs Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()