import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

def bac2():
    x = np.array([2,3,4])
    poly = PolynomialFeatures(3,include_bias=False)
    poly.fit_transform(x[:,None])

    poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = np.sin(x)+0.1 * rng.rand(50)

    poly_model.fit(x[:,np.newaxis],y)
    yfit = poly_model.predict(x[:,np.newaxis])

    plt.scatter(x,yfit)
    plt.plot(x,yfit)
    plt.show()

def main():
    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = 2 * x -5 + rng.rand(50)

    model = LinearRegression(fit_intercept=True)
    model.fit(x[:,np.newaxis],y)

    xfit = np.linspace(0,10,1000)
    yfit = model.predict(xfit[:,np.newaxis])

    print("Model slope ",model.coef_[0])
    print("Model intercept ",model.intercept_)

    plt.scatter(x,y)
    plt.plot(xfit,yfit)
    plt.show()

    rng = np.random.RandomState(1)
    x = 10 * rng.rand(100,3)
    y = 0.5+ np.dot(x,[1.5,-2.,-1.])
    model = LinearRegression(fit_intercept=True)
    model.fit(x,y)

    print(model.intercept_)
    print(model.coef_)
    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    plt.scatter(x, y)
    plt.plot(xfit, yfit)
    plt.show()
# def humidity():



if __name__ == '__main__':
    main()
    # bac2()
    # humidity();