import numpy as np
import scipy.special
import tensorflow as tf


class NeuralNetwork:
    activate_function = {
        'Logistic_Sigmoid': lambda x: 1/(1+np.e**(-x))
    }


class FeedfowardNeuralNetwork(NeuralNetwork):
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
        self.dataset = []  # 학습 데이터세트, (0 : 목표 신호, 1 : 입력 신호)

    def query(self, input):
        '''
        :keyword 순전파 메서드
        :param input: 입력 신호
        :return: 결과값 리스트(0~n)
        '''
        result = []  # 출력값
        input_data = np.array(input)  # 입력 데이터 - 열벡터
        for i in range(self.classes):
            if i is 0:  # 초기값
                result.append(input_data)
            else:  # 순전파
                result.append(
                    self.activate_function['Logistic_Sigmoid'](self.weight[i-1]@result[-1])
                )
        return result

    def descent(self, input, target, lr, weight_defined=None):
        '''
        :keyword 확률적 경사감소 메서드
        :param target: 목표 신호
        :param lr 학습률
        :param weight_defined 사용자 지정 가중치
        :return: 새로운 가중치 행렬
        '''
        input_data = np.array([input]).T
        outputs = self.query(input_data)
        target_data = np.array([target]).T
        weight = weight_defined if weight_defined else self.weight[:]
        for i in range(self.classes-1):
            if i is 0:
                error = target_data-outputs[-1]
            else:
                error =  weight[-i].T@prev_error
            weight[-i-1] += lr * (
                error*outputs[-i-1]*(1.0-outputs[-i-1])@outputs[-i-2].T
            )
            prev_error = error
        return weight

    def train(self, epoch, learning_rate, weight_defined=None, dataset_defined=None):
        '''
        :keyword 학습 메서드
        :param epoch: 주기
        :param learning_rate: 학습률
        :param weight_defined 사용자 지정 가중치
        :param dataset_defined 사용자 지정 데이터세트
        :return: 학습된 가중치 행렬
        '''
        weight = weight_defined if weight_defined else self.weight[:]
        dataset = dataset_defined if dataset_defined else self.dataset[:]
        for _ in range(epoch):
            for record in dataset:
                weight = self.descent(
                    input=record[1],
                    target=record[0],
                    lr=learning_rate,
                    weight_defined=weight
                )
        return weight


class FeedfowardNeuralNetwork_Dropout(FeedfowardNeuralNetwork):
    pass


class FeedfowardNeuralNetwork_Minibatch(FeedfowardNeuralNetwork):
    def __init__(self, nodes, minibatch):
        FeedfowardNeuralNetwork.__init__(self, nodes)
        self.minibatch = minibatch

    def descent(self, batch, lr, weight_defined=None):
        '''
        :keyword 확률적 경사감소 메서드
        :param batch 배치
        :param lr 학습률
        :param weight_defined 사용자 지정 가중치
        :return: 새로운 가중치 행렬
        '''
        weight = weight_defined if weight_defined else self.weight[:]
        error = [0 for i in range(self.classes - 1)]

        for target_data, input_data in batch:
            input = np.array([input_data]).T
            target = np.array([target_data]).T
            outputs = self.query(input)

            for i in range(self.classes - 1):
                if i is 0:
                    error[-1-i] += (target - outputs[-i-1])
                else:
                    error[-1-i] += (weight[-i].T @ error[-i])

        for i, error_r in enumerate(reversed(error)):
            weight[-i-1] += lr * (
                (error_r/len(batch)) * outputs[-i-1] * (1.0 - outputs[-i-1]) @ outputs[-i - 2].T
            )
        return weight

    def train(self, epoch, learning_rate, weight_defined=None, dataset_defined=None, batch_defined=None):
        '''
        :keyword 학습 메서드
        :param epoch: 주기
        :param learning_rate 학습률
        :param weight_defined 사용자 지정 가중치
        :param dataset_defined 사용자 지정 데이터세트
        :param batch_defined 사용자 지정 배치(n(D))
        :return: 학습된 가중치 행렬
        '''
        weight = weight_defined if weight_defined else self.weight[:]
        dataset = dataset_defined if dataset_defined else self.dataset[:]
        batch = batch_defined if batch_defined else self.minibatch
        batch_set = [[dataset.pop(0) for _ in range(batch)] for __ in range(len(dataset)//batch)]
        dataset = None;
        for _ in range(epoch):
            for minibatch in batch_set:
                weight = self.descent(
                    batch=minibatch,
                    lr=learning_rate,
                    weight_defined=weight
                )
        return weight


class FeedfowardNeuralNetwork_WeightAttenuation(FeedfowardNeuralNetwork):
    def __init__(self, nodes, attenuation_constant=10**(-3)):
        FeedfowardNeuralNetwork.__init__(self, nodes)
        self.att_const = attenuation_constant

    def descent(self, input, target, lr, weight_defined=None):
        '''
                :keyword 확률적 경사감소 메서드
                :param target: 목표 신호
                :param lr 학습률
                :param weight_defined 사용자 지정 가중치
                :return: 새로운 가중치 행렬
                '''
        input_data = np.array([input]).T
        outputs = self.query(input_data)
        target_data = np.array([target]).T
        weight = weight_defined if weight_defined else self.weight[:]
        for i in range(self.classes - 1):
            if i is 0:
                error = target_data - outputs[-1]
            else:
                error = weight[-i].T @ prev_error
            weight[-i - 1] += lr * (
                error * outputs[-i - 1] * (1.0 - outputs[-i - 1]) @ outputs[-i - 2].T
            )-(self.att_const/2)*(weight[-i-1])
            prev_error = error
        return weight

    def sum_of_squares(self, iterable):
        avg = sum(sum(iterable))/(len(iterable)*len(iterable[0]))
        return sum(sum((iterable-avg)**2))


        