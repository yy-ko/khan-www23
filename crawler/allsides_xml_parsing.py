import xml.etree.ElementTree as ET
import pandas as pd

xml_path = 'HLSTM-main/data/article_metadata.xml'
xml_file = open(xml_path, 'rt', encoding='UTF8')
result_path = 'HLSTM-main/xml_parsing_result'

anno = ET.parse(xml_file)
xml_articles = anno.getroot()

title_list = []
url_list = []
label_list = []
event_list = []

for article in xml_articles.findall('article'):
    title = article.find('title').text
    title_list.append(title)
    url = article.find('url').text
    url_list.append(url)
    label = article.find('news-source-bias').text
    label_list.append(label)
    event = article.find('event').text
    event_list.append(event)

event_list = list(set(event_list))
print(len(event_list))

title_df = pd.DataFrame(title_list, columns = ['title'])
url_df = pd.DataFrame(url_list, columns = ['url'])
label_df = pd.DataFrame(label_list, columns = ['label'])
event_df = pd.DataFrame(event_list, columns = ['event'])

print(len(title_df))
print(len(url_df))
print(len(label_df))

print(title_df.head())
print(url_df.head())
print(label_df.head())

title_df.to_csv(result_path + '/parsed_title.csv', index=False, encoding='utf-8-sig')
url_df.to_csv(result_path + '/parsed_url.csv', index=False, encoding='utf-8-sig')
label_df.to_csv(result_path + '/parsed_label.csv', index=False, encoding='utf-8-sig')
