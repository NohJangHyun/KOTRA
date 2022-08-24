from dashboard.models import China_Customs
from bs4 import BeautifulSoup
from selenium import webdriver
import re
from time import sleep
import pandas as pd
from .papago_translate import translation
from sentence_transformers import SentenceTransformer
from .get_pdf_text import pdf_to_text, preprocess_australia
import string
from .extract_keyword import bigram, bigram_embedding, keyword_embedding, similarity_test, get_keyword4, get_keyword6, final_keyword6, final_keyword4, make_df
from google.cloud import translate_v2 as translate

def preprocess(df_add):
    doc_list = []
    for doc in df_add:
        #구두점 제거
        doc1 = "".join([i for i in doc if i not in string.punctuation]).strip()

        #숫자 제거
        doc2 = "".join([i for i in doc1 if not i.isdigit()])

        #월 제거
        month = ['JANUARY', 'FEBUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']   
        doc3 = " ".join([i for i in doc2.split() if i not in month])
        
        #중국 문서 불용어 제거
        voca1 = ['commodity', 'customs', 'including', 'used', 'processing']
       
        doc4 = " ".join([i for i in doc3.split() if i.lower() not in voca1])
        doc_list.append(doc4)
    
    return doc_list

def china_trans_text(texts):
    trans_texts = []
    for i in range(len(texts)):
        client = translate.Client()
        result = client.translate(texts[i], target_language='en')
        trans_texts.append(result['translatedText'])
    return trans_texts


def scrap_china(cur_title):
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument("disable-blink-features=AutomationControlled")
    options.add_argument('User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36')
    browser = webdriver.Chrome(options=options)

    browser.get('http://www.customs.gov.cn/customs/302249/302266/index.html')
    sleep(40)
    soup = BeautifulSoup(browser.page_source, 'lxml')
    
    titles = []
    dates = []
    links = []
  
    try:
        ul = soup.find('ul', attrs = {'class' : 'conList_ull'}).find_all('li')
    except:
        print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다')
        return False

    for li in ul:
        title = li.a.get_text().strip()
        link = 'http://www.customs.gov.cn'+li.a['href']
        date = li.span.get_text().strip()

        titles.append(title)
        links.append(link)
        dates.append(date)

    #print('china cur title :', cur_title)
    #print('china1 titles[0] :', titles[0])

    if len(titles) != 0:
        if cur_title == titles[0]:
            return False
        else:
            try:
                idx = titles.index(cur_title)
            except:
                print('서버 통신 문제로 크롤링을 중단합니다. 추후 다시 시도 부탁드립니다2')
                return False
    else:
        return False

    contents = []
    title_translation = []

    for i in range(idx):
        browser.get(links[i])
        sleep(2)
        soup = BeautifulSoup(browser.page_source, 'lxml')
        
        try:
            content = soup.find('div', attrs={'id' : 'easysiteText'}).get_text().strip()
        except:
            print('본문 크롤링 실패')
            return False
        
        contents.append(content)
        title_translation.append(translation(titles[i], 'zh-CN'))  

    contents_translated = china_trans_text(contents)


    # 중국 내부 .xls에 hscode가 있는지 파악
    page_links = []
    xls_links = []
    for index, content in enumerate(contents):
        if '.xls' in content:
            page_links.append(links[index])
        else:
            page_links.append('')
    
    for link in page_links:
        if link == '':
            xls_links.append([])
        else:
            browser.get(link)
            sleep(10)
            try:
                soup = BeautifulSoup(browser.page_source, 'lxml')
            except:
                xls_links.append([])
                continue

            all_a = soup.find_all('a')
            xls = []
            for a in all_a:
                if a.has_attr('href') and 'xls' in a['href']:
                    xls.append('http://www.customs.gov.cn'+a['href'])
            
            xls_links.append(xls)

    x = re.compile('\d{10}')
    hs_codes = []
    for a in xls_links:
        hs = ''
        for link in a:
            try:
                excel = pd.read_excel(link)
            except:
                print('.xls를 열 수 없습니다')
                continue
    
            for row in excel.itertuples():
                lst = list(row)
                for ele in lst:
                    temp = x.match(str(ele))
                    if temp:
                        hs += temp.group() + '/'
        if len(hs) >= 1: hs = hs[:-1] 
        hs_codes.append(hs)

    print('keyword 추출 시작')
    model = SentenceTransformer('all-mpnet-base-v2')

    doc_list = preprocess(contents_translated)
    candidate_list = bigram(doc_list)
    bigram_keywords = bigram_embedding(model, doc_list, candidate_list)
    bigram_embeddings = keyword_embedding(model, bigram_keywords)
    code4, keyword4, keyword_embeddings4 = get_keyword4(model, '1')
    code6, keyword6, keyword_embeddings6 = get_keyword6(model, '1')
    bigram_result4, keyword_result4, cosine_result4 = similarity_test(0.6, bigram_embeddings, keyword_embeddings4, bigram_keywords, keyword4)
    bigram_result6, keyword_result6, cosine_result6 = final_keyword6(model, keyword_result4, keyword4, code4, bigram_result4, keyword6, code6, keyword_embeddings6)
    mti_4_keyword = final_keyword4(keyword_result6, code6, keyword6, keyword4, code4)
    df2 = make_df(bigram_result6, mti_4_keyword, keyword_result6, cosine_result6)

    df = pd.DataFrame()
    df['date'] = dates[:idx]
    df['link'] = links[:idx]
    df['title'] = titles[:idx]
    df['content'] = contents[:idx]
    df['translated'] = title_translation
    df['content_translated'] = contents_translated
    df['hscode'] = hs_codes

    dataframe = pd.concat([df,df2], axis=1)

    dataframe = dataframe[::-1]

    for row in dataframe.itertuples():
        China_Customs.objects.create(
                date = row[1],
                link = row[2],
                title = row[3],
                content = row[4],
                title_translated = row[5],
                content_translated = row[6],
                hs_code = row[7],
                bigram = row[8],
                mti4 = row[9],
                mti6 = row[10],
                similarity = row[11]
            )
    return True