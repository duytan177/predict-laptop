import matplotlib.pyplot as plt
import numpy as np

def question2():
    #PCA algorithm bằng tay
    #step 0 Create Data
    X = np.random.randint(10,100,100)
    X = X.reshape(20,5)

    #Step 1 Standarization
    mean =  np.mean(X,axis=0) #giá trị trung bình
    std = np.std(X,axis=0) # độ lệch chuẩn
    XStandarization = (X-mean)/std
    # print(f"Chuẩn hóa data {XStandarization}")

    #Step 2 Covariance
    covX = np.cov(XStandarization)
    # print(f"covariance\n{covX}")

    #Step 3 Eigen vector Eigent value
    e,v = np.linalg.eig(covX) # giá trị riêng vector riêng
    print("\nEigenvalues:")
    print(e)
    print("\nEigenvectors:")
    print(v)

    # Step 4: Feature vector
    # Chọn các vector riêng tương ứng với các giá trị riêng lớn nhất
    top_eigenvectors = v[:, np.argsort(e)[-2:]]
    #Step 5 Data
    plt.figure(figsize=(8, 5))
    plt.scatter(XStandarization[:,0],XStandarization[:,1], color="b", alpha=0.2)
    for ie, iv in zip(e, top_eigenvectors.T):
        print(iv)
        plt.plot([0, 3 * ie * iv[0]], [0, 3 * ie * iv[1]], "r-", lw=3)
    plt.show()

def main():
    # Question 1: Vẽ vector riêng và giá trị riêng của ma trận bất kì
    CenterPoint = np.array([0, 0])
    CovPoint = np.array([[0.6, 0.2], [0.2, 0.2]])
    n = 1000
    Points = np.random.multivariate_normal(CenterPoint, CovPoint, n).T
    print(Points)
    # np.display(Points)
    print(f"Matrix Points: {Points}")
    print(f"Shape of Matrix {Points.shape}")
    # ma trận hiệp phương sai
    A = np.cov(Points)
    print(A)
    e, v = np.linalg.eig(A)
    print(f"Eigen values:{e}")
    print(f"Eigen vector:{v}")

    plt.figure(figsize=(10, 8))
    # data x,y, color ,độ sáng
    plt.scatter(Points[0, :], Points[1, :], color="b", alpha=0.2)
    for ie, iv in zip(e, v.T):
        print(iv)
        plt.plot([0, 3 * ie * iv[0]], [0, 3 * ie * iv[1]], "r-", lw=3)
    plt.show()

if __name__ == '__main__':
    # main()
    question2()