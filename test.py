import expection
import numpy as np

p = 100
p_set = []
for i in range(p):  # 산포도 생성
    x = np.random.normal(0.0, 0.5)
    y = 3*x + np.random.normal(3.0, 0.5)
    p_set.append([x, y])

AI = expection.LinearRegression(p_set)
AI.regress()

print(AI.coeff)
