import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


class LinearRegression:  # 선형회귀 클래스
    def __init__(self, dataset):
        self.p = len(dataset)  # 데이터 개수(순서쌍 개수)

        self.dataset = dataset

        self.coeff = ()

    def draw(self, graph=0, dimension=1):  # 0 : 점만 표현, 1 : 그래프만 표현 2 : 점과 그래프를 함께 표현
        if dimension is 1:
            if graph is 0:
                plt.plot([self.dataset[i][0] for i in range(self.p)], [self.dataset[i][1] for i in range(self.p)], 'ro')  # 산포도
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

            elif graph is 1:  # 회귀식 그래프
                x_data = [self.dataset[i][0] for i in range(self.p)]
                plt.plot(x_data, [self.coeff[1] * x_data[i] + self.coeff[1] for i in range(self.p)], '-')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

            elif graph is 2:
                x_data = [self.dataset[i][0] for i in range(self.p)]
                y_data = [self.dataset[i][1] for i in range(self.p)]
                plt.plot(x_data, y_data, 'ro')  # 회귀식 그래프와 산포도를 함께 표현
                plt.plot(x_data, [self.coeff[0] * x_data[i] + self.coeff[1] for i in range(self.p)], '-')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
        elif dimension is 2:
            x = np.arange(0, 10, 0.1)  # points in the x axis
            y = np.arange(0, 10, 0.1)  # points in the y axis
            z = [self.coeff[0]*x[i] + self.coeff[1]*y[i]+self.coeff[2] for i in range(self.p)]  # points in the z axis

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, 'b-')
            plt.show()

    def regress(self, dimension=1, a=0.005, tolerance = 0.00001):
        var = sp.symbols(' '.join([chr(ord('A')+i) for i in range(dimension)]))
        sym = sp.symbols(' '.join([chr(ord('a')+i) for i in range(dimension+1)]))
        if type(var) is not tuple:
            var = var,

        f = sum([sym[i] * var[i] for i in range(dimension)]) + sym[-1]

        cost = sp.expand(
            (1 / (2 * self.p)) * sum(
                [(self.dataset[i][-1] - f.subs(
                    {
                        var[l]:self.dataset[i][l] for l in range(dimension)
                    }
                )) ** 2 for i in range(self.p)
                 ]
            )
        )

        self.coeff = self.argmin(dimension+1, cost, sym, a, tolerance)

    def argmin(self, k, func, sym, a=0.005, tolerance=0.00001):  # 경사감소법 옵티마이저
        v = np.array([0 for _ in range(k)]).T
        nv = np.array([-1 for _ in range(k)]).T  # 새로운 v 벡터

        while abs(nv[0] - v[0]) > tolerance:  # v 변화량이 허용범위보다 작을 때 까지
            v = nv
            nc = np.array(
                [sp.diff(func, sym[i]).subs({sym[l]: v[l] for l in range(k)}) for i in range(k)]
            ).T  # Nabla C : 목표함수의 w, b에 대한 편미분 값 벡터의 전치행렬
            nv = v - a * nc  # 새로운 v 벡터 할당
        return v


class LogisticRegression:
    pass


class NonLinearRegression:
    pass
