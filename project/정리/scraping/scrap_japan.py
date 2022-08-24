from dashboard.models import Japan_Customs
from bs4 import BeautifulSoup
from selenium import webdriver
import re
from time import sleep
import pandas as pd
from selenium.webdriver.common.by import By
from .papago_translate import translation
from sentence_transformers import SentenceTransformer
import io
import requests
from .get_pdf_text import pdf_to_text_for_jap
from .extract_keyword import bigram, bigram_embedding, keyword_embedding, similarity_test, get_keyword4, get_keyword6, final_keyword6, final_keyword4, make_df
from google.cloud import translate_v2 as translate

def preprocess(doc):
    doc_list = []
    for i in doc:
        res = re.sub('[^a-zA-Z]', ' ', str(i))
        doc1 = ''.join([i for i in res]).lower()
        
        mystopwords = ['japan', 'japanese', 'products', 'production']
        
        doc2 = " ".join([i for i in doc1.split() if i not in mystopwords])
        doc_list.append(doc2)
        
    return doc_list

def japan_trans_text(texts):
    trans_texts = []
    for i in range(len(texts)):
        client = translate.Client()
        result = client.translate(texts[i], target_language='en')
        trans_texts.append(result['translatedText'])
    return trans_texts

def scrap_japan(news_title, cus_title):
    x = scrap_japan_news(news_title)
    print('2단계 진입')
    y = scrap_japan_cus(cus_title)

    return x and y

def scrap_japan_news(cur_title):
    options= webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.headless = True
    driver = webdriver.Chrome(options = options)
    url= "https://www.meti.go.jp/press/category_02.html"

    driver.get(url)
    sleep(10)

    soup = BeautifulSoup(driver.page_source, 'lxml')

    try:
        dl = soup.find("dl", attrs={"class":"date_sp b-solid"})
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False

    dates = []

    dt = dl.find_all("dt")
    for d in range(len(dt)):
        date = dt[d].get_text().replace("年", "-").replace("月","-").replace("日", "-")
        dates.append(date[:-1])

    titles = []
    links = []

    A = dl.find_all("a")
    for a in A:
        title = a.get_text().strip()
        link = "https://www.meti.go.jp/"+a['href']
        titles.append(title)
        links.append(link)
    
    print('japan cur title :', cur_title)
    print('japan links[0] :', links[0])

    if len(links) != 0:
        if cur_title.strip() == links[0]:
            return False
        else:
            try:
                idx = links.index(cur_title)
            except:
                print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다2')
                return False
    else:
        return False
    
    CC = []
    title_translation = []

    for i in range(idx):
        url= links[i]
        title_translation.append(translation(titles[i], 'ja'))  

        driver.get(url)
        sleep(1)

        str_con = ""
        soup = BeautifulSoup(driver.page_source, 'lxml')
        content = soup.find("div", attrs={"class":"main w1000"})
        for con in content:
            c = con.get_text()
            if len(c) > 1:
                if c != "担当": 
                    str_con = str_con + " " + c
                else:
                    break
        str_con = str_con.replace("\n", "")
        CC.append(str_con)

    contents_translated = japan_trans_text(CC)
    title_eng = japan_trans_text(titles[:idx])

    total_contents = []
    for a,b in zip(title_eng, contents_translated):
        c = a + " " +b
        total_contents.append(c)

    print('keyword 추출 시작')
    model = SentenceTransformer('all-mpnet-base-v2')

    doc_list = preprocess(total_contents)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list) 
    bigram_embeddings = keyword_embedding(model, bigram_keywords) 
    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '5')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '5')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.6, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword = final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)

    new = pd.DataFrame()

    new['date'] = dates[:idx]
    new['title'] = titles[:idx]
    new['link'] = links[:idx]
    new['content'] = CC[:idx]
    new['translated'] = title_translation
    new['content_translated'] = contents_translated

    dataframe = pd.concat([new,df2], axis=1)

    dataframe = dataframe[::-1] 

    for row in dataframe.itertuples():
        Japan_Customs.objects.create(
                date = row[1],
                link = row[3],
                title = row[2],
                content = row[4],
                title_translated = row[5],
                content_translated = row[6],
                bigram = row[7],
                mti4 = row[8],
                mti6 = row[9],
                similarity = row[10],
                site = 1
            )
    return True

def scrap_japan_cus(cur_title):
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    browser = webdriver.Chrome(options=options)

    url = 'https://www.meti.go.jp/policy/external_economy/trade_control/wnlist.html'
    browser.get(url)

    sleep(1)

    soup = BeautifulSoup(browser.page_source, 'lxml')

    try:
        trs = soup.find('table', attrs={'class' : 'csv2table-table'}).find_all('tr')
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False
    
    trs = trs[1:]

    titles = []
    links = []
    dates = []

    for tr in trs[:5]:
        tds = tr.find_all('td')
        date = tds[0].get_text().replace('年', '-').replace('月', '-').replace('日', '')
        
        a_temp = tds[1].find_all('a')
        
        for a in a_temp:
            if a.has_attr('class') and 'inlink' in a['class']:
                continue
            elif a.has_attr('class') == False:
                continue
                
            title = a.get_text()
            
            if a['href'][0] == '/':
                link = 'https://www.meti.go.jp' + a['href']
            else:
                link = 'https://www.meti.go.jp/' + a['href']
            if 'PDF' in title:
                title = title[:title.index('PDF')-1]
            
            titles.append(title)
            links.append(link)
            dates.append(date)
    
    if len(links) != 0:
        if cur_title == links[0]:
            return False
        else:
            try:
                idx = links.index(cur_title)
            except:
                print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다2')
                return False
    else:
        return False
    
    print(idx)
    
    contents = []
    title_translation = []
    for i in range(idx):      
        title_translation.append(translation(titles[i], 'ja')) 
        url = links[i]
        response = requests.get(url)
        pdf_memory_file = io.BytesIO()
        pdf_memory_file.write(response.content)
        
        try:
            pdf_text = pdf_to_text_for_jap(pdf_memory_file)
        except:
            print('pdf 긁기 실패')
            pdf_text = ''
        
        contents.append(pdf_text)
    
    contents_translated = japan_trans_text(contents)
    title_eng = japan_trans_text(titles[:idx])

    total_contents = []
    for a,b in zip(title_eng, contents_translated):
        c = a + " " +b
        total_contents.append(c)

    print('keyword 추출 시작')
    model = SentenceTransformer('all-mpnet-base-v2')

    doc_list = preprocess(total_contents)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list) 
    bigram_embeddings = keyword_embedding(model, bigram_keywords) 
    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '5')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '5')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.6, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword = final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)


    new = pd.DataFrame()

    new['date'] = dates[:idx]
    new['title'] = titles[:idx]
    new['link'] = links[:idx]
    new['content'] = contents[:idx]
    new['translated'] = title_translation
    new['content_translated'] = contents_translated

    dataframe = pd.concat([new,df2], axis=1)

    dataframe = dataframe[::-1] 

    for row in dataframe.itertuples():
        Japan_Customs.objects.create(
                date = row[1],
                link = row[3],
                title = row[2],
                content = row[4],
                title_translated = row[5],
                content_translated = row[6],
                bigram = row[7],
                mti4 = row[8],
                mti6 = row[9],
                similarity = row[10],
                site = 2
            )
    return True