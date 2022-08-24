from dashboard.models import MTI4_table, MTI6_table, Vietnam_Customs
from bs4 import BeautifulSoup
from selenium import webdriver
import re
from time import sleep
import pandas as pd
from selenium.webdriver.common.by import By
from .papago_translate import translation
from google.cloud import translate_v2 as translate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from .extract_keyword import bigram, bigram_embedding, keyword_embedding, similarity_test, get_keyword4, get_keyword6, final_keyword6, final_keyword4, make_df

def preprocess(doc):
    doc_list = []
    for i in doc:
        res = re.sub('[^a-zA-Z]', ' ', str(i))
        doc1 = ''.join([i for i in res]).lower()
        
        mystopwords = ['goods', 'chairman', 'customs', 'document', 'vietnam', 'uk', 'hy', 'pm', 'processed', 'process']
        
        doc2 = " ".join([i for i in doc1.split() if i not in mystopwords])
        doc_list.append(doc2)
        
    return doc_list


def vietnam_trans_text(texts):
    trans_texts = []
    for i in range(len(texts)):
        client = translate.Client()
        result = client.translate(texts[i], target_language='en')
        trans_texts.append(result['translatedText'])
    return trans_texts

def scrap_vietnam(cur_title):
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument('window-size=1920x1080')
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    url = 'https://www.customs.gov.vn/index.jsp?pageId=4&cid=30'
    driver.get(url)
    sleep(15)
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    dates = []
    links = []
    titles = []
    
    try:
        rows = soup.find('div', {'class': re.compile('^content-list')}).find_all('div', {'class': 'content_item'})
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False
    
    for row in rows[0:10]:
        ps = row.find('p', {'class':'note-layout'})

        if ps != None:
            date = ps.get_text().strip()
            dates.append(date)
            
    for row in rows[0:10]:
        h3s = row.find('h3', {'class':'content_title'})

        if h3s != None:
            link = 'https://www.customs.gov.vn/' + row.a['href']
            title = h3s.get_text().strip()
            links.append(link)
            titles.append(title) 
    
    #print('vietnam cur title : ', cur_title)
    #print('vietnam titles[0] :', titles[0])

    if len(titles) != 0:
        if cur_title == titles[0]:
            return False
        else:
            try:
                idx = titles.index(cur_title)
            except:
                print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
                return False
    else:
        return False
    
    texts = []
    title_translation = []

    for i in range(idx):
        link = links[i]
        driver.get(link)
        sleep(5)
        text = driver.find_element(By.ID, 'detail-news')
        
        while text.text == '':
            text = driver.find_element(By.ID, 'detail-news')

        texts.append(text.text)
        title_translation.append(translation(titles[i], 'vi'))  

    # text --> 영어 번역
    contents_translated = vietnam_trans_text(texts)

    # 영어 번역 text --> 키워드 추출
    print('keyword 추출을 진행합니다')
    model = SentenceTransformer('all-mpnet-base-v2')

    doc_list = preprocess(contents_translated)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list)
    bigram_embeddings = keyword_embedding(model, bigram_keywords)

    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '3')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '3')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.52, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword = final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(dates[:idx], format = "%d/%m/%Y %H:%M %p")
    df['link'] = links[:idx]
    df['title'] = titles[:idx]
    df['text'] = texts[:idx]
    df['translated'] = title_translation
    df['content_trans'] = contents_translated

    df['date'] = df['date'].dt.date

    dataframe = pd.concat([df,df2], axis=1)

    dataframe = dataframe[::-1]

    for row in dataframe.itertuples():
        Vietnam_Customs.objects.create(
                date = row[1],
                link = row[2],
                title = row[3],
                content = row[4],
                title_translated = row[5],
                content_translated = row[6],
                bigram = row[7],
                mti4 = row[8],
                mti6 = row[9],
                similarity = row[10]
            )
    return True
