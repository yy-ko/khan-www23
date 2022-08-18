# Named Entity Recognition (NER) apply to Reddit triplets

###################################### NER - Flair ######################################
# from flair.data import Sentence
# from flair.models import SequenceTagger

# # load tagger
# tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

# # make example sentence
# # sentence = Sentence("On September 1st George won 1 dollar while watching Game of Thrones.")
# sentence = Sentence("Google LLC is a multinational technology company. It was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in Califonia.")

# # predict NER tags
# tagger.predict(sentence)

# # print sentence
# print(sentence)

# # print predicted NER spans
# print('The following NER tags are found:')
# # iterate over entities and print
# for entity in sentence.get_spans('ner'):
#     print(entity)

###################################### NER - SpaCy ######################################
import spacy
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")
# article = nlp("Google LLC is a multinational technology company. It was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in Califonia.")
# article = nlp("President Joe Biden departs Holy Trinity Catholic Church in the Georgetown section of Washington, after attending a Mass in Washington on July 17, 2022.")

# for X in article.ents:
#     print("Text:", X.text, "\tLabel:", X.label_)

###################################### NER - Reddit #####################################
import pandas as pd
from tqdm import tqdm

# path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_liberal.csv'
# path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_conservative.csv'
# dest = '/home/yyko/workspace/political_pre/KG_construction/NER_Type'
path = '/home/yyko/workspace/political_pre/KG_construction/new_reddit(220720)/reddit_conservative_NF_02(220720).csv'
dest = '/home/yyko/workspace/political_pre/KG_construction/new_reddit(220720)/ner'

reddit_df = pd.read_csv(path, encoding='utf-8')

print("============================= Before Triplet Filtering ==============================")
print(len(reddit_df))

filtered_list_all = []
filtered_list_person = [] # PERSON (person name)
filtered_list_org = [] # ORG (organization name)
filtered_list_norp = [] # NORP (affiliation)
filtered_list_event = [] # EVENT (event name)
filtered_list_gpe = [] # GPE (geo-political entity)
filtered_list_loc = [] # LOC (location name)
filtered_list_law = []
filtered_list_fac = []
filtered_list_lang = []
filtered_list_prod = []
filtered_list_quant = []
filtered_list_date = []
filtered_list_money = []
filtered_list_percent = []
filtered_list_time = []
filtered_list_card = []
filtered_list_ord = []
filtered_list_wka = []

for sen in tqdm(reddit_df.filtered.values.tolist()):
    triplet = nlp(sen)
    for x in triplet.ents:
        if x.label_ == 'PERSON' or x.label_ == 'ORG' or x.label_ == 'NORP' or \
            x.label_ == 'EVENT' or x.label_ == 'GPE' or x.label_ == 'LOC' or \
                x.label_ == 'LAW' or x.label_ == 'FAC' or x.label_ == 'LANGUAGE' or \
                    x.label_ == 'PRODUCT' or x.label_ == 'QUANTITY' or x.label_ == 'DATE' or \
                        x.label_ == 'MONEY' or x.label_ == 'PERCENT' or x.label_ == 'TIME' or \
                            x.label_ == 'CARDINAL' or x.label_ == 'ORDINAL' or x.label_ == 'WORK_OF_ART':
            filtered_list_all.append(sen)
        # if x.label_ == 'PERSON' or x.label_ == 'ORG' or x.label_ == 'NORP' or x.label_ == 'EVENT' or x.label_ == 'GPE' or x.label_ == 'LOC':
        if x.label_ == 'PERSON':
            # print(sen)
            filtered_list_person.append(sen)
        if x.label_ == 'ORG':
            filtered_list_org.append(sen)
        if x.label_ == 'NORP':
            filtered_list_norp.append(sen)
        if x.label_ == 'EVENT':
            filtered_list_event.append(sen)
        if x.label_ == 'GPE':
            filtered_list_gpe.append(sen)
        if x.label_ == 'LOC':
            filtered_list_loc.append(sen)
        if x.label_ == 'LAW':
            filtered_list_law.append(sen)
        if x.label_ == 'FAC':
            filtered_list_fac.append(sen)
        if x.label_ == 'LANGUAGE':
            filtered_list_lang.append(sen)
        if x.label_ == 'PRODUCT':
            filtered_list_prod.append(sen)
        if x.label_ == 'QUANTITY':
            filtered_list_quant.append(sen)
        if x.label_ == 'DATE':
            filtered_list_date.append(sen)
        if x.label_ == 'MONEY':
            filtered_list_money.append(sen)
        if x.label_ == 'PERCENT':
            filtered_list_percent.append(sen)
        if x.label_ == 'TIME':
            filtered_list_time.append(sen)
        if x.label_ == 'CARDINAL':
            filtered_list_card.append(sen)
        if x.label_ == 'ORDINAL':
            filtered_list_ord.append(sen)
        if x.label_ == 'WORK_OF_ARTRSON':
            filtered_list_wka.append(sen)

print('before deduplication(all type): ', len(filtered_list_all))
filtered_list_all_deduplication = list(set(filtered_list_all))
print('after deduplication(all type): ', len(filtered_list_all_deduplication))

filtered_list_person_deduplication = list(set(filtered_list_person))
filtered_list_org_deduplication = list(set(filtered_list_org))
filtered_list_norp_deduplication = list(set(filtered_list_norp))
filtered_list_event_deduplication = list(set(filtered_list_event))
filtered_list_gpe_deduplication = list(set(filtered_list_gpe))
filtered_list_loc_deduplication = list(set(filtered_list_loc))
filtered_list_law_deduplication = list(set(filtered_list_law))
filtered_list_fac_deduplication = list(set(filtered_list_fac))
filtered_list_lang_deduplication = list(set(filtered_list_lang))
filtered_list_prod_deduplication = list(set(filtered_list_prod))
filtered_list_quant_deduplication = list(set(filtered_list_quant))
filtered_list_date_deduplication = list(set(filtered_list_date))
filtered_list_money_deduplication = list(set(filtered_list_money))
filtered_list_percent_deduplication = list(set(filtered_list_percent))
filtered_list_time_deduplication = list(set(filtered_list_time))
filtered_list_card_deduplication = list(set(filtered_list_card))
filtered_list_ord_deduplication = list(set(filtered_list_ord))
filtered_list_wka_deduplication = list(set(filtered_list_wka))

filtered_all_df = pd.DataFrame({'filtered' : filtered_list_all_deduplication})
filtered_person_df = pd.DataFrame({'filtered' : filtered_list_person_deduplication})
filtered_org_df = pd.DataFrame({'filtered' : filtered_list_org_deduplication})
filtered_norp_df = pd.DataFrame({'filtered' : filtered_list_norp_deduplication})
filtered_event_df = pd.DataFrame({'filtered' : filtered_list_event_deduplication})
filtered_gpe_df = pd.DataFrame({'filtered' : filtered_list_gpe_deduplication})
filtered_loc_df = pd.DataFrame({'filtered' : filtered_list_loc_deduplication})
filtered_law_df = pd.DataFrame({'filtered' : filtered_list_law_deduplication})
filtered_fac_df = pd.DataFrame({'filtered' : filtered_list_fac_deduplication})
filtered_lang_df = pd.DataFrame({'filtered' : filtered_list_lang_deduplication})
filtered_prod_df = pd.DataFrame({'filtered' : filtered_list_prod_deduplication})
filtered_quant_df = pd.DataFrame({'filtered' : filtered_list_quant_deduplication})
filtered_date_df = pd.DataFrame({'filtered' : filtered_list_date_deduplication})
filtered_money_df = pd.DataFrame({'filtered' : filtered_list_money_deduplication})
filtered_percent_df = pd.DataFrame({'filtered' : filtered_list_percent_deduplication})
filtered_time_df = pd.DataFrame({'filtered' : filtered_list_time_deduplication})
filtered_card_df = pd.DataFrame({'filtered' : filtered_list_card_deduplication})
filtered_ord_df = pd.DataFrame({'filtered' : filtered_list_ord_deduplication})
filtered_wka_df = pd.DataFrame({'filtered' : filtered_list_wka_deduplication})

print("============================= After Triplet Filtering ==============================")
print('filtered_all_df: ', len(filtered_all_df))
print('filtered_person_df: ', len(filtered_person_df))
print('filtered_org_df: ', len(filtered_org_df))
print('filtered_norp_df: ', len(filtered_norp_df))
print('filtered_event_df: ', len(filtered_event_df))
print('filtered_gpe_df: ', len(filtered_gpe_df))
print('filtered_loc_df: ', len(filtered_loc_df))
print('filtered_law_df: ', len(filtered_law_df))
print('filtered_fac_df: ', len(filtered_fac_df))
print('filtered_lang_df: ', len(filtered_lang_df))
print('filtered_prod_df: ', len(filtered_prod_df))
print('filtered_quant_df: ', len(filtered_quant_df))
print('filtered_date_df: ', len(filtered_date_df))
print('filtered_money_df: ', len(filtered_money_df))
print('filtered_percent_df: ', len(filtered_percent_df))
print('filtered_time_df: ', len(filtered_time_df))
print('filtered_card_df: ', len(filtered_card_df))
print('filtered_ord_df: ', len(filtered_ord_df))
print('filtered_wka_df: ', len(filtered_wka_df))

# filtered_df.to_csv(dest + "/Reddit_NF_FINAL_conservative.csv", index=False, encoding='utf-8-sig')
filtered_all_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_ALL.csv", index=False, encoding='utf-8-sig')
filtered_person_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_PERSON.csv", index=False, encoding='utf-8-sig')
filtered_org_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_ORG.csv", index=False, encoding='utf-8-sig')
filtered_norp_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_NORP.csv", index=False, encoding='utf-8-sig')
filtered_event_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_EVENT.csv", index=False, encoding='utf-8-sig')
filtered_gpe_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_GPE.csv", index=False, encoding='utf-8-sig')
filtered_loc_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_LOC.csv", index=False, encoding='utf-8-sig')
filtered_law_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_LAW.csv", index=False, encoding='utf-8-sig')
filtered_fac_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_FAC.csv", index=False, encoding='utf-8-sig')
filtered_lang_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_LANGUAGE.csv", index=False, encoding='utf-8-sig')
filtered_prod_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_PRODUCT.csv", index=False, encoding='utf-8-sig')
filtered_quant_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_QUANTITY.csv", index=False, encoding='utf-8-sig')
filtered_date_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_DATE.csv", index=False, encoding='utf-8-sig')
filtered_money_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_MONEY.csv", index=False, encoding='utf-8-sig')
filtered_percent_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_PERCENT.csv", index=False, encoding='utf-8-sig')
filtered_time_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_TIME.csv", index=False, encoding='utf-8-sig')
filtered_card_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_CARDINAL.csv", index=False, encoding='utf-8-sig')
filtered_ord_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_ORDINAL.csv", index=False, encoding='utf-8-sig')
filtered_wka_df.to_csv(dest + "/Reddit_NF_conservative(220720)_ner_WORK_OF_ART.csv", index=False, encoding='utf-8-sig')

print("=================================== Finished!!! ====================================")
