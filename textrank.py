# -*- coding: utf-8 -*-
import networkx
import gensim
from collections import Counter
from itertools import combinations
from konlpy.tag import Okt
from nltk.tag import pos_tag
import re

corpus = '''
존경하는 국민 여러분, 먼저 이번 최순실 씨 관련 사건으로 이루 말할 수 없는 큰 실망과 염려를 끼쳐드린 점 다시 한 번 진심으로 사과드립니다.

무엇보다 저를 믿고 국정을 맡겨주신 국민 여러분께 돌이키기 힘든 마음의 상처를 드려서 너무나 가슴이 아픕니다.

저와 함께 헌신적으로 뛰어주셨던 정부의 공직자들과 현장의 많은 분들, 그리고 선의의 도움을 주셨던 기업인 여러분께도 큰 실망을 드려 송구스럽게 생각합니다.

국가 경제와 국민의 삶에 도움이 될 것이라는 바람에서 추진된 일이었는데 그 과정에서 특정 개인이 이권을 챙기고 여러 위법 행위까지 저질렀다고 하니 너무나 안타깝고 참담한 심정입니다.

이 모든 사태는 모두 저의 잘못이고, 저의 불찰로 일어난 일입니다. 저의 큰 책임을 가슴 깊이 통감하고 있습니다.

어제 최순실 씨가 중대한 범죄 혐의로 구속되었고 안종범 전 정책조정수석이 체포되어 조사를 받는 등 검찰 특별수사 본부에서 철저하고 신속하게 수사를 진행하고 있습니다.

앞으로 검찰은 어떠한 것에도 구애받지 말고 명명백백하게 진실을 밝히고 이를 토대로 엄정한 사법처리가 이루어져야 할 것입니다.

저는 이번 일의 진상과 책임을 규명하는데 있어서 최대한 협조하겠습니다.

이미 청와대 비서실과 경호실에도 검찰의 수사에 적극 협조하도록 지시하였습니다.

필요하다면 저 역시 검찰의 조사에 성실하게 임할 각오이며 특별검사에 의한 수사까지도 수용하겠습니다.

국민 여러분, 저는 청와대에 들어온 이후 혹여 불미스러운 일이 생기지는 않을까 염려하여 가족 간의 교류마저 끊고 외롭게 지내왔습니다.

홀로 살면서 챙겨야 할 여러 개인사들을 도와줄 사람조차 마땅치 않아서 오랜 인연을 갖고 있었던 최순실 씨로부터 도움을 받게 되었고, 왕래하게 되었습니다.

제가 가장 힘들었던 시절에 곁을 지켜주었기 때문에 저 스스로 경계의 담장을 낮추었던 것이 사실입니다.

돌이켜 보니 개인적 인연을 믿고 제대로 살피지 못한 나머지 주변사람들에게 엄격하지 못한 결과가 되고 말았습니다.

저 스스로를 용서하기 어렵고 서글픈 마음까지 들어 밤잠을 이루기도 힘이 듭니다.

무엇으로도 국민들의 마음을 달래드리기 어렵다는 생각을 하면 ‘내가 이러려고 대통령을 했나’ 하는 자괴감이 들 정도로 괴롭기만 합니다.

국민의 마음을 아프지 않게 해 드리겠다는 각오로 노력해 왔는데 이렇게 정반대의 결과를 낳게 되어 가슴이 찢어지는 느낌입니다. 심지어 제가 사이비 종교에 빠졌다거나 청와대에서 굿을 했다는 이야기까지 나오는데 이는 결코 사실이 아니라는 점을 분명히 말씀드립니다.

우리나라의 미래 성장 동력을 만들기 위해 정성을 기울여온 국정과제들까지도 모두 비리로 낙인찍히고 있는 현실도 참으로 안타깝습니다. 일부의 잘못이 있었다고 해도 대한민국의 성장 동력 만큼은 꺼트리지 말아 주실 것을 호소드립니다.

다시 한 번 저의 잘못을 솔직하게 인정하고, 국민 여러분께 용서를 구합니다.

이미 마음으로는 모든 인연을 끊었지만 앞으로 사사로운 인연을 완전히 끊고 살겠습니다.

그동안의 경위에 대해 설명을 드려야 마땅합니다만 현재 검찰의 수사가 진행 중인 상황에서 구체적인 내용을 일일이 말씀드리기 어려운 점을 죄송스럽게 생각합니다. 자칫 저의 설명이 공정한 수사에 걸림돌이 되지 않을까 염려하여 오늘 모든 말씀을 드리지 못하는 것뿐이며 앞으로 기회가 될 때 밝힐 것입니다.

또한 어느 누구라도 이번 수사를 통해 잘못이 드러나면 그에 상응하는 책임을 져야할 것이며 저 역시도 모든 책임을 질 각오가 되어 있습니다.

국민 여러분, 지금 우리 안보가 매우 큰 위기에 직면해 있고, 우리 경제도 어려운 상황입니다. 국내외의 여러 현안이 산적해 있는 만큼 국정은 한시라도 중단되어서는 안 됩니다.

대통령의 임기는 유한하지만 대한민국은 영원히 계속되어야만 합니다.

더 큰 국정 혼란과 공백 상태를 막기 위해 진상 규명과 책임 추궁은 검찰에 맡기고 정부는 본연의 기능을 하루속히 회복해야만 합니다.

국민들께서 맡겨주신 책임에 공백이 생기지 않도록 사회 각계의 원로분들과 종교 지도자 분들, 여야 대표님들과 자주 소통하면서 국민 여러분과 국회의 요구를 더욱 무겁게 받아들이겠습니다.

다시 한 번 국민 여러분께 깊이 머리 숙여 사죄드립니다.
'''


def summarize(document, unit='sentence', lines_to_summarize=3, language='kor'):
    class Sentence:
        def __init__(self, sentence, index):
            kkma = Okt()
            self.sentence = sentence
            self.nouns = kkma.nouns(self.sentence)
            self.index = index

        def __eq__(self, other):
            return isinstance(other, Sentence) and other.index == self.index

        def __hash__(self):
            return self.index

        def __str__(self):
            return self.sentence

    class Word:
        def __init__(self, word, index):
            self.word = word
            self.index = index

        def __eq__(self, other):
            return isinstance(other, Word) and other.index == self.index

        def __hash__(self):
            return self.index

        def __str__(self):
            return self.word

    def segment(text):
        corpus = re.split('(?<!\d)\.|\.(?!\d)', text)
        result = []
        for sentence in corpus:
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        return result
    
    def make_vocab(text):
        twitter = Okt()
        vocab = []
        for sentence in text:
            vocab.append(*twitter.pharse(sentence))
        vocab=list(set(vocab))
        retrun vocab

    def jac_index(a: Sentence, b: Sentence):
        mult_a = Counter(a.nouns)
        mult_b = Counter(b.nouns)
        if sum((mult_a | mult_b).values()) == 0:
            return 0
        return sum((mult_a & mult_b).values()) / sum((mult_a | mult_b).values())

    def tf_idf(a, b):
        pass
    
    def 

    def textrank(sentences, func=jac_index):
        graph = networkx.Graph()
        graph.add_nodes_from(sentences)
        pairs = combinations(sentences, 2)
        for a, b in pairs:
            graph.add_edge(a, b, weight=func(a, b))
        page_rank = networkx.pagerank(graph, weight='weight')
        return {sentence: page_rank.get(sentence) for sentence in sentences}

    document = [Sentence(sentence, i) for i, sentence in enumerate(segment(corpus))]
