#테스트 진행 위한 준비
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import itertools
import string
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
#문서 전처리 함수
def preprocess(df_add):
    doc_list = []
    for doc in df_add:
        #구두점 제거
        doc1 = "".join([i for i in doc if i not in string.punctuation]).strip()

        #숫자 제거
        doc2 = "".join([i for i in doc1 if not i.isdigit()])

        #월 제거
        month = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr',
                 'may', 'jun','jul', 'aug', 'sep', 'oct', 'nov', 'dec']   

        doc3 = " ".join([i for i in doc2.split() if i not in month])
        doc_list.append(doc3)
    
    return doc_list

#문서 바이그램 단위로 나누는 함수 + nltk 제공하는 불용어 제거
def bigram(doc_list):
    n_gram_range = (2, 2)
    stop_words = "english"

    candidate_list = []
    for doc in doc_list:
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
        candidate = count.get_feature_names_out()
        candidate_list.append(candidate)
    
    return candidate_list

#전처리 문서와 바이그램 embedding 후 유사도 높은 20개 키워드 추출 함수
def bigram_embedding1(doc_list, candidate_list):
    bigram_keywords1 = []
    top_n = 20

    for i in range(len(doc_list)):
        doc_embeddings = model.encode([doc_list[i]])
        candidate_embeddings = model.encode(candidate_list[i])
        distances = cosine_similarity(doc_embeddings, candidate_embeddings)
        bigram_keywords1.append([candidate_list[i][index] for index in distances.argsort()[0][-top_n:]])
        
    return bigram_keywords1

#원본 문서(전처리 X)와 바이그램 embedding 후 유사도 높은 20개 키워드 추출 함수
def bigram_embedding2(doc_list, candidate_list):
    bigram_keywords2 = []
    top_n = 20

    for i in range(len(doc_list)):
        doc_embeddings = model.encode([df_add[i]])
        candidate_embeddings = model.encode(candidate_list[i])
        distances = cosine_similarity(doc_embeddings, candidate_embeddings)
        bigram_keywords2.append([candidate_list[i][index] for index in distances.argsort()[0][-top_n:]])
        
    return bigram_keywords2

#추출 키워드 embedding 함수
def keyword_embedding(bigram_keywords):
    bigram_embeddings = []
    for i in range(len(doc_list)):
        bigram_embedding = []
        for keyword in bigram_keywords[i]:
            bigram_embedding.append(model.encode(keyword))
        bigram_embeddings.append(bigram_embedding)
    
    return bigram_embeddings

#선정 키워드(단어)와 embedding 값 불러오기
def get_keyword1():
    df_keyword = pd.read_csv("호주_키워드_HS_KSIC(description 추가)", index_col = False)
    keyword = list(df_keyword["번역"])
    keyword_embeddings = []

    for ele in keyword:
        keyword_embeddings.append(model.encode(ele))
    
    return keyword, keyword_embeddings

#선정 키워드(단어)와 embedding 값 불러오기
def get_keyword2():
    df_keyword = pd.read_csv("호주_키워드_HS_KSIC(description 추가).csv", index_col = False)
    keyword = list(df_keyword["description"])
    keyword_embeddings = []

    for ele in keyword:
        keyword_embeddings.append(model.encode(ele))
    
    return keyword, keyword_embeddings

#추출 키워드와 선정 키워드 유사도 비교
def similarity_test(bigram_embeddings, bigram_keywords, keyword_embeddings, keyword):
    bigram_result = []
    keyword_result = []
    cosine_result = []
    for index, bigram in enumerate(bigram_embeddings): #2번 반복(총 2개 문서)

        b_result = []
        k_result = []
        c_result = []

        for i in range(len(bigram)): 
            for j in range(len(keyword_embeddings)): 
                distances = cosine_similarity([bigram[i]],[keyword_embeddings[j]]) #유사도 비교
                if distances[0][0] > 0.48:
                    b_result.append(bigram_keywords[index][i])
                    k_result.append(keyword[j])
                    c_result.append(str(round(float(distances),3)))

        bigram_result.append(b_result)
        keyword_result.append(k_result)
        cosine_result.append(c_result)
        
    return bigram_result, keyword_result, cosine_result

#유사도 높은 순으로 df 만들기
def make_df(bigram_result, keyword_result, cosine_result):
    df_final = pd.DataFrame()

    bigram_list = []
    keyword_list = []
    distance = []

    for i in range(len(df_add)):
        B, K, D = [], [], []
        b_result,k_result, d_result = "", "", ""
        df_check = pd.DataFrame()
        df_check['bigram'] = pd.Series(bigram_result[i])
        df_check['keyword'] = pd.Series(keyword_result[i])
        df_check['distance'] = pd.Series(cosine_result[i])

        df_check = df_check.sort_values(by="distance", ascending=False)

        B = df_check['bigram'].tolist()
        K = df_check['keyword'].tolist()
        D = df_check['distance'].tolist()

        for b in B:
            b_result = b_result + "/" + b
        for k in K:
            k_result = k_result + "/" + k
        for d in D:
            d_result = d_result + "/" + d


        bigram_list.append(b_result[1:])
        keyword_list.append(k_result[1:])
        distance.append(d_result[1:])

        del df_check

    df_final['bigram'] = bigram_list
    df_final['keyword'] = keyword_list
    df_final['distance'] = distance

    return df_final
