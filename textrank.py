import networkx
from collections import Counter
from itertools import combinations

# 1) 입력 데이터를 문장 단위로 분리

# 2) 문장 간 유사도 계산 함수 정의


def jac_index(a, b):
    mult_a = Counter(a)
    mult_b = Counter(b)
    return sum((mult_a & mult_b).values())/sum((mult_a | mult_b).values())  # 예외처리


def tf_idf(a, b):
    pass

# 3) (2)에서 정의한 함수로 textrank 그래프 구축


def textrank(sentences, func=jac_index):
    graph = networkx.Graph()
    graph.add_nodes_from(sentences)
    pairs = combinations(sentences, 2)
    for a, b in pairs:
        graph.add_edge(a, b, weight=func(a, b))
    page_rank = networkx.pagerank(graph, weight='weight')
    return {sentence: page_rank.get(sentence) for sentence in sentences}

# 4) (3)에서 구축한 그래프 lookup table
