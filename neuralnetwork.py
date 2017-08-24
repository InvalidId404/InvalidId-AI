import numpy as np
import scipy.special


class NeuralNetwork:
    activate_function = {
        'Logistic_Sigmoid': lambda x: scipy.special.expit(x)
    }


class FeedFowardNeuralNetwork(NeuralNetwork):
    def __init__(self, nodes):
        '''
        :keyword 생성자
        :param nodes: 노드 개수(List)
        '''
        self.nodes = nodes  # 계층별 노드 개수 - List
        self.classes = len(nodes)  # 계층 수
        self.weight = [
            np.random.normal(
                0.0, self.nodes[i]**(-1/2), (self.nodes[i+1], self.nodes[i])
            ) for i in range(self.classes-1)
        ]  # 가중치 초기화
        self.dataset = []  # 학습 데이터세트, (0 : 입력 데이터, 1 : 레이블)

    def query(self, input):
        '''
        :keyword 순전파 메서드
        :param input: 입력 신호
        :return: 결과값 리스트(0~n)
        '''
        result = []  # 출력값
        input_data = np.array([input]).T  # 입력 데이터 - 열벡터
        for i in range(self.classes):
            if i is 0:  # 초기값
                result.append(input_data)
            else:  # 순전파
                result.append(
                    self.weight[i-1]@result[-1]
                )
        return result

    def descent(self, input, target, lr, tole, weight_defined=None):
        '''
        :keyword 확률적 경사감소 메서드
        :param target: 목표 신호
        :param learning_rate 학습률
        :param tolerance 오차 허용치
        :param weight_defined 사용자 지정 가중치
        :return: 새로운 가중치 행렬
        '''
        input_data = np.array([input]).T
        outputs = self.query(input_data)
        target_data = np.array([target]).T
        weight = weight_defined if not weight_defined else self.weight
        errors = []
        for i in range(self.classes):
            if i is 0:
                errors.insert(0, target_data-outputs[-1])
            else:
                errors.insert(0, weight[-i].T@errors[0])
            weight[-i-1] += learning_rate * (
                errors[0]*outputs[-i-1]*(1.0-outputs[-i-1])@outputs[-i-2]
            )
        return weight

    def train(self, epoch, learning_rate, tolerance, weight_defined=None, dataset_defined=None):
        '''
        :keyword 학습 메서드
        :param epoch: 주기
        :param learning_rate: 학습률
        :param tolerance: 오차 허용치
        :return: 학습된 가중치 행렬
        '''
        weight = weight_defined if not weight_defined else self.weight
        dataset = dataset_defined if not dataset_defined else self.dataset
        for _ in range(epoch):
            for record in dataset:
                weight = self.descent(
                    input=record[0],
                    target=record[1],
                    lr=learning_rate,
                    tole=tolerance
                )
        return weight
