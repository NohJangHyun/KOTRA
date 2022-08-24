from dashboard.models import America_Customs
from bs4 import BeautifulSoup
from selenium import webdriver
import re
import pandas as pd
from .papago_translate import translation
import io
import requests
from .get_pdf_text import pdf_to_text, preprocess_america
from .extract_keyword import bigram, bigram_embedding, keyword_embedding, similarity_test, get_keyword4, get_keyword6, final_keyword6, final_keyword4, make_df
from PyPDF2 import PdfFileReader
import string
from sentence_transformers import SentenceTransformer


def preprocess(df_add):
    doc_list = []
    for doc in df_add:
        doc = re.sub("Notice is hereby given pursuant to CBP regulations that", "", doc)
        doc = re.sub("The Bureau of Customs and Border Protection CBP of the Department of Homeland Security", "", doc)
        doc = re.sub("Pursuant to section c Tariff Act of U S C c as amended by section of Title VI Customs Modernization of the North American Free Trade Agreement Implementation Act Pub L Stat", "", doc)
        doc = re.sub("Pursuant to CFR f", "", doc)
        
        #구두점 제거
        doc1 = "".join([i for i in doc if i not in string.punctuation]).strip()

        #숫자 제거
        doc2 = "".join([i for i in doc1 if not i.isdigit()])

        #월 제거
        month = ['JANUARY', 'FEBUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']   
        doc3 = " ".join([i for i in doc2.split() if i not in month])
        
        #미국 문서 불용어 제거
        voca1 = ['BUREAU', 'OF', 'CUSTOMS', 'AND', 'BORDER', 'PROTECTION', 'BULLETIN', 'AGENCY', 'ACTION', 'SUMMARY'
               , 'DECISIONS', 'VOL', 'NO', 'SUPPLEMENTARY', 'FOR', 'LOCATION', 'ADDRESS', 'EFFECTIVE', 'DATE']
        
        doc4 = " ".join([i for i in doc3.split() if i.upper() not in voca1])
        
        #필요없는 단어 제거
        voca2 = ['customs', 'process', 'commodity', 'establish', 'establishes', 'establishment', 'cbp', 'purpose'
                 ,'concerning', 'tariff', 'pursuant', 'states', 'document', 'processing', 'test', 'pertaining', 'entering', 'government',
                'entered', 'testing', 'producing', 'certain', 'containing', 'preceding', 'regarding', 'bringing']
        doc5 = " ".join([i for i in doc4.split() if i.lower() not in voca2])
        doc_list.append(doc5)
    
    return doc_list

def scrap_america(cur_title):
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    browser = webdriver.Chrome(options=options)

    browser.get('https://www.cbp.gov/trade/rulings/bulletin-decisions')
    browser.implicitly_wait(10)
    html = browser.page_source

    soup = BeautifulSoup(html, 'html.parser')

    try:
        div= soup.find('div', {'id': re.compile('a[0-9]')})
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False

    date, link, title, content = [], [], [], []
    
    a_list = div.select('ul a')
    link_list = ['https://www.cbp.gov/'+a['href'] for a in a_list]

    for site in link_list[:5]:
        browser.get(site)
        inner_html = browser.page_source
        soup = BeautifulSoup(inner_html, 'html.parser')
        date.append(soup.select_one('.view-content span.field-content').get_text())
        link.append('https://www.cbp.gov'+ soup.select_one('div.field-content a.survey-processed')['href'])
        title.append(soup.select_one('h1#page-title').get_text().strip())
        browser.back()

    #print('america1 cur title :', cur_title)
    #print('america1 titles[0] :', title[0])
    #print(title)

    if len(title) != 0:
        if cur_title.strip() == title[0]:
            return False
        else:
            try:
                idx = title.index(cur_title.strip())
            except:
                print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
                return False
    else:
        return False
    

    pos = 0
    title_translation = []

    for i in range(idx):
        title_translation.append(translation(title[i], 'en'))  

    for pdf in link:
        response = requests.get(pdf)
        pdf_memory_file = io.BytesIO()
        pdf_memory_file.write(response.content)
        page_num = PdfFileReader(pdf_memory_file).numPages
        text = pdf_to_text(pdf_memory_file, page_num)
        content.append(preprocess_america(text))

        pos += 1
        if pos == idx: break
    
    model = SentenceTransformer('all-mpnet-base-v2')

    doc_list = preprocess(content)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list)
    bigram_embeddings = keyword_embedding(model, bigram_keywords)
    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '4')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '4')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.5, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword= final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(date[:idx])
    df['link'] = link[:idx]
    df['title'] = title[:idx]
    df['text'] = content[:idx]
    df['translated'] = title_translation[:idx]

    dataframe = pd.concat([df,df2], axis=1)

    dataframe = dataframe[::-1]

    for row in dataframe.itertuples():
        America_Customs.objects.create(
                date = row[1],
                link = row[2],
                title = row[3],
                content = row[4],
                title_translated = row[5],
                bigram = row[6],
                mti4 = row[7],
                mti6 = row[8],
                similarity = row[9]
            )
    return True