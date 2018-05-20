import random

'''
세대 : 염색체 집합, Generation
염색체 : 유전자 집합, Chromosome
유전자 : 정보 집합, Gene
'''


class GeneticMethod:
    def __init__(self, generation):
        self.generation = generation
        self.length = len(generation)  # 세대에 속한 염색체 수

    def evolution(self, method):  # 진화 메서드
        '''
        인자 : 세대, (선택, 교차, 변이)
        :return: 진화된 해집합        '''

        parents = self.select(method=method[0], generation=self.generation)  # 부모 유전자 선택
        offspring = self.cross(method=method[1], parents=parents)  # 자손 유전자 생성
        #  result = self.mutant(method=method[2], target=offspring)  # 세대에 변이 적용
        result = offspring

        return result


    def select(self, method, generation, cost, selective_pressure=0):  # 선택 메서드
        '''
        :param method: 선택 연산의 방식(0-토너먼트, 1-품질 비례 룰렛휠)
        :param generation: 선택 연산을 적용할 세대
        :param cost: 염색체 비용 함수
        :param selective_pressure: 선택압
        :return: 두 쌍의 부모 염색체
        '''

        # 토너먼트 선택
        if method is 0:
            result = []

            for i in range(self.length):
                r = []
                for _ in range(2):
                    if random.random()>=selective_pressure:
                        r.append(max(random.choice(generation), random.choice(generation)))
                    else:
                        r.append(min(random.choice(generation), random.choice(generation)))
                result.append(r)
            return result

        # 품질 비례 룰렛휠 선택
        if method is 1:  #  적합도 계산, 비례하여 룰렛휠 배분, 선택
            costs = [cost(x) for x in generation]
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
                        if dart<=flow:
                            r.append(generation[i])
                result.append(r)

            return result


    def cross(self, method, parents, selective_pressure=0, points = 0):  # 교차 메서드
        '''
        :param method: 교차 연산의 방식(0-균등 교차, 1-다점 교차, 2-PMX)
        :param parents: 교차 연산을 적용할 부모 염색체 쌍의 집합
        :param selective_pressure: 선택압(균등 교차)
        :param points: 교차점(다점 교차)
        :return:
        '''

        # 균등 교차
        if method is 0:
            result = []
            for parent in parents:
                r = []
                for i in len(parent[0]):
                    if random.random() >= selective_pressure:
                        r.append(max(parent[0][i], parent[1][i]))
                    else:
                        r.append(min(parent[0][i], parent[1][i]))
                result.append(r)
            return result

        # PMX


    def mutant(self, method, target):  # 변이 메서드
        pass


def main():
    first_gen = []
    for i in range(100):
        first_gen.append([random.choice(0, 1) for _ in range(6)])

    generation = GeneticMethod(generation=first_gen)
    solution = [0, 1, 1, 1, 0, 1]

    def cost(chromo):
        score = 6
        for gene, sol in zip(chromo, solution):
            if gene is sol:
                score -= 1
        return score


    for i in range(5000):
        generation.generation = generation.evolution(method=[0, 0])

    print(
        sorted(generation.generation, key=cost, reverse=1)[0]
    )


if __name__ == '__main__':
    main()
