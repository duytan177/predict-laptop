import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Question 1. Create two vector and test : dot, cosine, plus,
# Question 2. Solving y=x+1 and y=-2x+1
# Question 3. Test Covariance, correlation of Matrix
# Question 4:
# We have 3 stock of Apple, Microsoft and Google. Let's find
# the relationship among them.Try to show by chart
AppleStock = [70 , 72 , 80 , 75 , 69 ,  85]
MSStock    = [120, 125, 130, 128, 110, 150]
GOStock    = [200, 150, 120, 170, 250, 280]
def main():
    #question1
    list1 = [[1,2],
             [6,7],
             [8,9]]
    list2 = [[2,3,4],
             [6,7,8 ]]
    vector1 = np.array(list1)
    vector2= np.array(list2)
    dot = np.dot(vector1,vector2)
    print(f"Nhân 2 ma trận \n{dot}")

    listPlus1 = [[1,2],
                [20,30]]
    listPlus2 = [[10, 20],
                 [6, 7]]
    vector1 = np.array(listPlus1)
    vector2= np.array(listPlus2)
    plus = np.add(vector1,vector2)
    print(f"Cộng 2 ma trận \n{plus}")

    # np.linalg.norm là norm/ euclidean khoảng cách
    #cosine l khoảng cách giữa 2 vector
    cosine = np.dot(vector1,vector2) / np.linalg.norm(vector1) * np.linalg.norm(vector2)
    print(f"cosine 2 ma trận \n{cosine}")
    print(f"{np.linalg.norm(vector1) * np.linalg.norm(vector2) * cosine}")

    # [ - 1  1] [x] = [ 1]
    # [2  1] [y] = [ 1]
    A = np.array([[-1, 1],
                  [2, -1]])
    B = np.array([1, 1])

    # Giải hệ phương trình
    print(np.dot(A.T,B))

    # Tạo một ma trận dữ liệu (mỗi hàng là một biến, mỗi cột là một quan sát)
    data = np.array([3,4,5,6,7])
    # Tính ma trận hiệp phương sai
    covariance_matrix = np.cov(data, rowvar=False)
    # Tính ma trận tương quan
    correlation_matrix = np.corrcoef(data, rowvar=False)
    print("Covariance Matrix:")
    print(covariance_matrix)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Xây dựng ma trận từ dữ liệu
    matrix = np.array([AppleStock, MSStock, GOStock])
    # Tính ma trận tương quan Pearson
    correlation_matrix = np.corrcoef(matrix)
    print(correlation_matrix)
    # Trực quan hóa mối quan hệ bằng biểu đồ heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
    # Tạo DataFrame từ ma trận tương quan
    column_names = ['Apple', 'Microsoft', 'Google']
    df = pd.DataFrame(correlation_matrix, index=column_names, columns=column_names)
    # Chỉnh sửa DataFrame để có đường chéo chính là 1 và nửa trên và nửa dưới giống nhau
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            df.iat[i, j] = ''
    #vector riêng là chiều cuar 2 ghướng mới
    #giá trị riêng dùng để xác định xem chiều nào bỏ cái nào lớn hơn lấy


if __name__ == '__main__':
    main()
