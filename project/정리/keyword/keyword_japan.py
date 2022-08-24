import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

from dashboard.models import Japan_Customs

with open('Jap_news_keyword.csv', encoding='utf-8') as csv_file:
    rows = csv.reader(csv_file)
    next(rows, None)
    for row in rows:
        Japan_Customs.objects.create(
            date = row[0],
            link = row[1],
            title = row[2],
            content = row[3],
            title_translated = row[4],
            content_translated = row[5],
            bigram = row[6],
            mti4 = row[7],
            mti6 = row[8],
            similarity = row[9],
            site = 1
        )
        print(row)

with open('Jap_cus_keyword.csv', encoding='utf-8') as csv_file:
    rows = csv.reader(csv_file)
    next(rows, None)
    for row in rows:
        Japan_Customs.objects.create(
            date = row[0],
            link = row[1],
            title = row[2],
            content = row[3],
            title_translated = row[4],
            content_translated = row[5],
            bigram = row[6],
            mti4 = row[7],
            mti6 = row[8],
            similarity = row[9],
            site = 2
        )
        print(row)
        