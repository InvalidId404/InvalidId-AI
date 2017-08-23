import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class KMeansCluster:
    def __init__(self):
        self.clustering_data = []
        self.centroids = []

    def cluster(self, num_of_centroids=4, tolerance=0.05):
        centroids = [
            [max(self.clustering_data[0]), max(self.clustering_data[1])],
            [max(self.clustering_data[0]), min(self.clustering_data[1])],
            [min(self.clustering_data[0]), max(self.clustering_data[1])],
            [min(self.clustering_data[0]), min(self.clustering_data[1])]
        ]  # 임의로 설정
        while True:
            groups = [[] for _ in range(num_of_centroids)]
            for data in self.clustering_data:
                distance = []
                for k in centroids:
                    distance.append(
                        sum([data[i]-k[i] for i in range(len(k))])**2
                    )
                groups[distance.index(min(distance))].append(data)
            new_centroids = [
                [
                    sum(groups[i][x])/len(groups[i]) for x in range(2)
                ] for i in range(len(groups))
            ]
            if new_centroids[0][0]-centroids[0][0] < tolerance:
                break
            else:
                centroids = new_centroids
                self.centroids = centroids
                self.draw()
                continue
        return new_centroids

    def draw(self):
        df = pd.DataFrame({'x': [v[0] for v in self.clustering_data],
                           'y': [v[1] for v in self.clustering_data]})
        sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, legend=False)
        plt.plot([c[0] for c in self.centroids], [c[1] for c in self.centroids], 'ro')
        plt.show()


class DBSCANCluster:
    def __init__(self):
        self.training_data = []
        # [(x, y), 소속 클러스터]

    def cluster(self, radios, threshold):
        d = lambda x, y: (x[0]-y[0])**2+(x[1]-y[1])**2
        c = 0

        def core(data, r=radios, t=threshold):
            for record in data:
                if not record[1]:
                    n = [p for p in self.training_data if d(p[0], record[0]) < radios]
                    if len(n) < threshold:
                        record[1] = 'Noise or Border'
                    else:
                        record[1] = 'Core'
                        core(n, r, t)
                else:
                    continue
                return 0
        core(self.training_data)


