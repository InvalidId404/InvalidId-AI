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
        self.nodes = nodes
        self.classes = len(nodes)
        self.weight = [
            np.random.normal(
                0.0, self.nodes[i]**(-1/2), (self.nodes[i+1], self.nodes[i])
            ) for i in range(self.classes-1)
        ]
        pass

    def query(self, input):
        '''
        :keyword 전파 메서드
        :param input: 입력 신호
        :return: 결과값 리스트(0~n)
        '''
        result = []
        input_data = np.array([input]).T
        for i in range(self.classes):
            if i is 0:
                result.append(input_data)
            else:
                result.append(
                    self.weight[i-1]@result[-1]
                )
        return result

    def descent(self, input, target, learning_rate, tolerance, weight_argument=None):
        '''
        :keyword 확률적 경사감소 메서드
        :param target: 목표 신호
        :param learning_rate 학습률
        :param tolerance 오차 허용치
        :param weight_argument 사용자 지정 가중치
        :return: 새로운 가중치 행렬
        '''
        input_data = np.array([input]).T
        outputs = self.query(input_data)
        target_data = np.array([target]).T
        weight = weight_argument if not weight_argument else self.weight
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

    def train(self, epoch, target, learning_rate, tolerance):
        '''
        :keyword 학습 메서드
        :param epoch: 주기
        :param target: 목표 신호
        :param learning_rate: 학습률
        :param tolerance: 오차 허용치
        :return: 학습된 가중치 행렬
        '''

nodes = [7, 3, 2]
AI = FeedFowardNeuralNetwork(nodes)