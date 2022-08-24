from dashboard.models import Australia_Customs
from bs4 import BeautifulSoup
from selenium import webdriver
import re
from time import sleep
import pandas as pd
from .papago_translate import translation
from sentence_transformers import SentenceTransformer
import io
import requests
from .get_pdf_text import pdf_to_text, preprocess_australia
import string
from .extract_keyword import bigram, bigram_embedding, keyword_embedding, similarity_test, get_keyword4, get_keyword6, final_keyword6, final_keyword4, make_df

def preprocess(doc):
    doc_list = []
    for i in doc:
        doc1 = "".join([i for i in i if i not in string.punctuation])

        doc2 = " ".join([i for i in doc1.split() if not i.isdigit()])

        month = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr',
                 'may', 'jun','jul', 'aug', 'sep', 'oct', 'nov', 'dec']   
        
        doc3 = " ".join([i for i in doc2.split() if i not in month])
        
        stopword = ["australian", "australia" ,"duty", "abfgovauimportingexportingand", "manufacturingimportinghowtoimportdisposingunenteredabandonedgoods", 
           "declare", "consigment", "sorted", "license","goods", "products", "quota", "ii", "russia", "httpswwwabfgovauimporting", "customs",
                   "indexation", "working", "available", "subheadings", "cpi", "wwwabfgovau", "tariff", "office", "rates", "spirits", "rules", "blue",
                   "manufacturingtariffclassificationharmonizedsystemchanges", "www", "abf", "bureau", "acn", "gov", "au" ]

        doc4 = " ".join([i for i in doc3.split() if i not in stopword])
        doc_list.append(doc4)
    
    return doc_list

def scrap_australia(cur_title):
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument('window-size=1920x1080')
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=options)
    
    url = 'https://www.abf.gov.au/help-and-support/notices/australian-customs-notices'

    driver.get(url)
    sleep(5)
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    dates = []
    titles = []
    links = []

    try:
        date_rows = soup.select('div.col-sm-3 span')
        title_rows = soup.find('tbody').find_all('tr', attrs = {'class' : re.compile('^accordion-header')})
        link_rows = soup.find('tbody').find_all('tr', attrs = {'class' : re.compile('^accordion-content')})
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False

    for row in date_rows:
        date = row.text
        dates.append(date)

    for row in title_rows:
        tds = row.find_all('td')
        title = tds[1].get_text().replace('Press Enter to show more details.', '')
        titles.append(title)

    for row in link_rows:
        link = 'https://www.abf.gov.au' + row.a['href']
        links.append(link)

    # print('titles : ',  titles[0])
    # print('cur_title : ', cur_title)

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
    hscodes= []
    for i in range(idx):
        url = links[i]
        response = requests.get(url)
        pdf_memory_file = io.BytesIO()
        pdf_memory_file.write(response.content)
        content = pdf_to_text(pdf_memory_file, 2)
        content_temp, hscode = preprocess_australia(content)
        texts.append(content_temp)
        hscodes.append(hscode)
        title_translation.append(translation(titles[i], 'en'))    

    hscode_list = []
    for ele in hscodes:
        aus_hs = ''
        for e in ele:
            aus_hs += str(e) + '/'
        
        if len(aus_hs) >= 1: aus_hs = aus_hs[:-1]

        hscode_list.append(aus_hs)

    print('keyword 추출을 진행합니다')
    model = SentenceTransformer('all-mpnet-base-v2')
    
    doc_list = preprocess(texts)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list)
    bigram_embeddings = keyword_embedding(model, bigram_keywords)
    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '2')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '2')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.5, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword = final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(dates[:idx])
    df['link'] = links[:idx]
    df['title'] = titles[:idx]
    df['text'] = texts
    df['translated'] = title_translation
    df['hscode'] = hscode_list
    
    dataframe = pd.concat([df,df2], axis=1)

    dataframe = dataframe[::-1]

    for row in dataframe.itertuples():
        Australia_Customs.objects.create(
                date = row[1],
                link = row[2],
                title = row[3],
                content = row[4],
                title_translated = row[5],
                hs_code = row[6],
                bigram = row[7],
                mti4 = row[8],
                mti6 = row[9],
                similarity = row[10]
            )
    return True