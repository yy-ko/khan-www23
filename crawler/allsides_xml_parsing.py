import xml.etree.ElementTree as ET
import pandas as pd

xml_path = 'github/khan/data/allsides_metadata.xml'
xml_file = open(xml_path, 'rt', encoding='UTF8')
result_path = 'github/khan/data'

anno = ET.parse(xml_file)
xml_articles = anno.getroot()

article_list = []
# title_list = []
# url_list = []
# label_list = []
# event_list = []
# date_list = []

for article in xml_articles.findall('article'):
    title = article.find('title').text
    # title_list.append(title)
    url = article.find('url').text
    # url_list.append(url)
    label = article.find('news-source-bias').text
    source = article.find('news-source').text
    # label_list.append(label)
    event = article.find('event').text
    # event_list.append(event)
    date = article.find('date-of-publication').text
    # date_list.append(date)
    article_list.append({'title': title, 'url': url, 'label': label, 'event': event, 'date': date, 'source': source})

# print('======================== event check ========================')
# event_list_deduplication = list(set(event_list))
# print('event count: ',len(event_list))
# print('event types: ',len(event_list_deduplication))
# print('=============================================================')

# title_df = pd.DataFrame(title_list, columns = ['title'])
# url_df = pd.DataFrame(url_list, columns = ['url'])
# label_df = pd.DataFrame(label_list, columns = ['label'])
# event_type_df = pd.DataFrame(event_list_deduplication, columns = ['event'])
# date_df = pd.DataFrame(date_list, columns = ['date'])
article_df = pd.DataFrame(article_list)

# print(len(title_df))
# print(len(url_df))
# print(len(label_df))
# print(len(date_df))

# print(title_df.head())
# print(url_df.head())
# print(label_df.head())
# print(date_df.head())
print(len(article_df))
print(article_df.head())

# event_list = event_list[7770:]
# check_df = pd.concat([title_df, date_df, url_df, label_df], axis = 1)
# check_df_01 = check_df.iloc[7770:,]

# event_count={}
# for i in event_list:
#     try: event_count[i] += 1
#     except: event_count[i]=1
# print('======================== event count ========================')
# print(pd.DataFrame([event_count]))
# event_count_df = pd.DataFrame([event_count])
# event_count_df.to_csv('/home/yyko/workspace/political_pre/github/khan/data' + '/event_count_10385_01.csv', index=False, encoding='utf-8-sig')
# print('=============================================================')
# check_df_01.to_csv(result_path + '/check_01.csv', index=False, encoding='utf-8-sig')


# title_df.to_csv(result_path + '/parsed_title.csv', index=False, encoding='utf-8-sig')
# url_df.to_csv(result_path + '/parsed_url.csv', index=False, encoding='utf-8-sig')
# label_df.to_csv(result_path + '/parsed_label.csv', index=False, encoding='utf-8-sig')
article_df.to_csv(result_path + '/kcd_allsides_url_re.csv', index=False, encoding='utf-8-sig')
