# -*- coding: utf-8 -*-

import random


class GeneticMethod:
    def __init__(self, generation, cost):
        self.generation = generation
        self.length = len(generation)  # 세대에 속한 염색체 수
        self.cost = cost

    def evolution(self, method):  # 진화 메서드
        parents = self.select(method=method[0], generation=self.generation)  # 부모 유전자 선택
        offspring = self.cross(method=method[1], parents=parents)  # 자손 유전자 생성
        result = self.mutant(method=method[2], target=offspring)  # 세대에 변이 적용

        return result

    def select(self, method, generation, selective_pressure=0.6):  # 선택 메서드
        # 토너먼트 선택
        if method is 0:
            result = []

            for i in range(self.length):
                r = []
                for _ in range(2):
                    if random.random() >= selective_pressure:
                        rev = False
                    else:
                        rev = True
                    r.append(
                        sorted([random.choice(generation), random.choice(generation)], key=self.cost, reverse=rev)[0]
                    )

                result.append(r)
            return result

        # 품질 비례 룰렛휠 선택
        if method is 1:  # 적합도 계산, 비례하여 룰렛휠 배분, 선택
            costs = [self.cost(x) for x in generation]
            fitnesses = [min(costs)-cost + (min(costs)-max(costs))/(selective_pressure-1) for cost in costs]
            # 적합도 = (최소 가치-현재 가치)+ (최소 가치-최대 가치)/(선택압-1)

            result = []

            for i in range(self.length):
                r = []
                for _ in range(2):
                    dart = random.uniform(0, sum(fitnesses))
                    flow = 0
                    for fit in fitnesses:
                        flow += fit
                        if dart <= flow:
                            r.append(generation[i])
                result.append(r)

            return result

    def cross(self, method, parents, selective_pressure=0.6):  # 교차 메서드

        # 균등 교차
        if method is 0:
            result = []
            for parent in parents:
                r = []
                for i in range(len(parent[0])):
                    if random.random() >= selective_pressure:
                        rev = False
                    else:
                        rev = True
                    r.append(
                        sorted(parent, key=self.cost, reverse=rev)[0][i]
                    )
                result.append(r)
            return result

        # 다점 교차
        if method is 1:
            pass

        # PMX
        if method is 2:
            pass

    def mutant(self, method, target):  # 변이 메서드
        return 0


def main():
    first_gen = []
    for i in range(100):
        first_gen.append([random.choice([0, 1]) for _ in range(6)])

    solution = [0, 1, 1, 1, 0, 1]

    def cost(chromo):
        score = 6
        for gene, sol in zip(chromo, solution):
            if gene is sol:
                score -= 1
        return score

    generation = GeneticMethod(generation=first_gen, cost=cost)

    for i in range(5000):
        generation.generation = generation.evolution(method=[0, 0])

    print(
        sorted(generation.generation, key=cost, reverse=True)[0]
    )


if __name__ == '__main__':
    main()
