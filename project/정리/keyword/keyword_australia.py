import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

from dashboard.models import Australia_Customs

with open('AUS_keyword.csv', encoding='utf-8') as csv_file:
    rows = csv.reader(csv_file)
    next(rows, None)
    for row in rows:
        Australia_Customs.objects.create(
            date = row[0],
            link = row[1],
            title = row[2],
            content = row[3],
            title_translated = row[4],
            hs_code=row[5],
            bigram = row[6],
            mti4 = row[7],
            mti6 = row[8],
            similarity = row[9]
        )
        print(row)
        