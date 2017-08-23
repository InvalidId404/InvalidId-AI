import expection
import numpy as np
import tools.excel as xl

p = 1000
p_set = []
for i in range(p):  # 산포도 생성
    x = np.random.normal(0.0, 0.5)
    y = np.random.normal(3.0, 0.4)*x + np.random.normal(0.0, 0.4)
    p_set.append([x, y])


AI = expection.LinearRegression(p_set)
AI.regress()
AI.draw(graph=0)
AI.draw(graph=1)
AI.draw(graph=2)

print(AI.coeff)
# 회귀

