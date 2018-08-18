# -*- coding: utf-8 -*-
import networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from itertools import combinations
from konlpy.tag import Okt

import re

corpus = '''
일 끝나서 친구들과 한잔.
내일은 노는 토요일이니깐.
일 얘기 사는 얘기 재미난 얘기.
시간가는 줄 모르는 이 밤.
술기운이 올라오니 사내놈들끼린.
결국엔 여자 얘기.
적적해서 서로의 전화기를 꺼내.
번호목록을 뒤져보지.
너는 지금 머해 자니 밖이야.
뜬금없는 문자를 돌려보지 난.
어떻게 해볼까란 뜻은 아니야.
그냥 심심해서 그래.
아니 외로워서 그래.
술자리가 끝나가 3차로.
이동하기 전인데 문자는.
일분 칠분 십분 이십분.
담배와 애만 태우는 지금.
답장 왔어 오빠 나 남친 생겼어.
늦었어 좀 일찍 연락하지 그랬어.
담에 봐 그냥 담에 낮에 봐.
후회할거 알면서 전 여자친구에게
너는 지금 머해 자니 밖이야
뜬금없는 문자를 보내보지. 난
어떻게 해볼까란 뜻은 아니야.
그냥 심심해서 그래.
아니 외로워서 그래.
아 진짜 술만 들어가면
왜 이렇게 들뜨는지.
나도 잘 몰라 왜 난 그녀들을
부르는지 갑자기 허전해.
작업을 걸어대지 여기저기 오늘 밤
나 자존심 다 버렸네.
전 여친한테 더럽게 달라붙어
봤지만 그녀는 버럭해.
너 진짜 철없게 언제까지 이럴래.
미안해 갑자기 외로운걸 어떡해.
껄떡대 껄떡대 나 여기저기 다
맘껏 들쑤시고 다녀. 온 거릴 거릴 다
또 잠들었어. 프라이머리가
텐션 떨어진다. 동훈아 넌 저리가.
이제 해가 나올 시간이 되니까
눈이 녹듯이 사그라드는 기대감.
너무 지치고 피곤해 자고 싶어.
이제 나 첫차를 타고 졸며 집에가.
창밖에 앉아 밖을 바라보네.
나는 꾸벅꾸벅 조는데
사람들은 하루를 시작해.
눈부셔 아침해를 보는게.
정신은 맑아지지 않는 기분.
아직도 바쁜 내 손가락
아직 손은 바쁘게 움직이지.
해장국이나 먹고 갈래 오빠랑.
너는 지금 머해 자니 밖이야.
뜬금없는 문자를 보내보지 난
어떻게 해볼까란 뜻은 아니야.
그냥 심심해서 그래.
아니 외로워서 그래.
가지마 제발 심심해서 그래.
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
            word = twitter.morphs(sentence)
            for w in word:
                vocab.append(w)
        return list(set(vocab))

    doc = [Sentence(stc, i) for i, stc in enumerate(segment(document))]
    wrd = [Word(wrd, i) for i, wrd in enumerate(make_vocab(document))]
    sen = segment(document)

    def jac_index(a: Sentence, b: Sentence):
        mult_a = Counter(a.nouns)
        mult_b = Counter(b.nouns)
        if sum((mult_a | mult_b).values()) == 0:
            return 0
        return sum((mult_a & mult_b).values()) / sum((mult_a | mult_b).values())

    def tf_idf(a, b):
        pass

    def freq(a, b):
        result = 0
        for sentence in sen:
            if a.word in sentence and b.word in sentence:
                result += 1 
        return result/sum([
            a.word in line or b.word in line for line in segment(document)
        ])

    def textrank(sentences, func=jac_index):
        graph = networkx.Graph()
        graph.add_nodes_from(sentences)
        pairs = combinations(sentences, 2)
        for a, b in pairs:
            graph.add_edge(a, b, weight=func(a, b))
        page_rank = networkx.pagerank(graph, weight='weight')
        return {sentence: page_rank.get(sentence) for sentence in sentences}

    doc_by_tr = textrank(doc)
    wrd_by_tr = textrank(wrd, func=freq)

    li_doc = [(word, tr) for word, tr in doc_by_tr.items()]
    li_wrd = [(word, tr) for word, tr in wrd_by_tr.items()]

    sorted_doc = sorted(li_doc, key=lambda x: x[1], reverse=True)
    sorted_wrd = sorted(li_wrd, key=lambda x: x[1], reverse=True)

    return sorted_doc if unit=='sentence' else 'word'

print(summarize(corpus))