#This is a (quite messy) file where I fixed some bugs in the csv files and did some minor changes



'''years_imatinib = filter_papers_for_pki('IMATINIB', papers)
years = sorted(years_imatinib)
years = [i for i in years if len(i) == 4]
w = collections.Counter(years)
plt.plot(w.keys(), w.values())
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xticks(rotation=30, ha='right')

plt.show()'''

'''import os
import pandas
directory = os.fsencode('/Users/dj_jnr_99/Downloads/timelines')

years = list(reversed(['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005']))
for file in os.listdir(directory):
    data = pandas.read_csv('/Users/dj_jnr_99/Downloads/timelines/'+str(file)[2:-1], names=['Year', 'Count'])
    count = list(data['Count'][2:])
    if len(count) != len(years):
        zeros = (len(years)-len(count))*[0]
        count.extend(zeros)
    count = [int(line) for line in count]
    plt.plot(years, list(reversed(count)), label=str(file)[2:-5])


plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xticks(rotation=30, ha='right')
plt.legend(framealpha=1, frameon=True);
plt.show()'''

#['TOFACITINIB', 'LAPATINIB', 'DASATINIB', 'AFATINIB', 'RUXOLITINIB']:

'''plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.legend(framealpha=1, frameon=True);
plt.xticks(rotation=30, ha='right')

plt.show()'''


'''from tools.scraper import getArticleIdsForKeyword, extractArticlesData, split
from tools.paper_infos import get_PMC_info, get_PM_info
from drug_attributes import get_all_pki_synonyms
import csv
import ast
import pandas as pd
import sys'''
#csv.field_size_limit(sys.maxsize)

'''print()
print("GPL example:")
for gpl_name, gpl in gse.gpls.items():
    print("Name: ", gpl_name)
    print("Metadata:",)
    for key, value in gpl.metadata.items():
        print(" - %s : %s" % (key, ", ".join(value)))
    print("Table data:",)
    print(gpl.table.head())
    break'''




'''
from network_feed import get_rcts_from_searchterm
from NER import NER_diseases

relations_db = []
diseases_db = []
papers_db = []
for pki in pki_reduced:
    dic = {}
    RCTs = get_rcts_from_searchterm(pki)
    for rct in RCTs:
        if rct not in papers_db:
            papers_db.append(rct)
        diseases = NER_diseases(rct[0])
        for disease in diseases:
            if disease not in diseases_db:
                diseases_db.append(disease)
            if disease[1] in dic.keys():
                dic[disease[1]].append(rct[4])
            else:
                dic[disease[1]] = [rct[4]]
    for x in dic:
        relation = pki[-1] +'+'+ x
        print([relation, pki[-1], x, dic[x]])
        relations_db.append([relation, pki[-1], x, dic[x]])

print('RELATIONS: ', len(relations_db))
print('PAPERS: ', len(papers_db))'''




#26341741 27979923 27151992
'''
with open('data/papers_db_updated_properly.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    papers = list(csv_reader)

update = []
for i in papers:
    data = extractArticlesData('PubMed', i[4])
    data = get_PM_info(data, i[4])
    if data != []:
        update.append([i[0], data[0][1], i[2], i[3], i[4]])
    else:
        data = extractArticlesData('PubMed', i[4])
        data = get_PM_info(data, i[4])
        if data != []: #try it twice because sometimes server just fails shortly
            update.append([i[0], data[0][1], i[2], i[3], i[4]])
        else:
            update.append([i[0], ['NO ABSTRACT FOUND'], i[2], i[3], i[4]])
            print('FAIL', i[4])

print(len(update))

with open("data/papers_db_updated.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(update)'''

'''
data = extractArticlesData('PubMed', '10574370')
data = get_PM_info(data, '10574370')
print(data)'''


'''
update = []
for i in papers:
    if i[1] == "['NO ABSTRACT FOUND']" or len(i[1]) < 8:
        data = extractArticlesData("PubMed", i[4])
        data = get_PM_info(data, i[4])
        if data != []:
            update.append([i[0], data[0][1], i[3], i[4]])
            print([i[0], data[0][1], i[3], i[4]])
        else:
            update.append(i)
    else:
        update.append(i)

for i in update:
    print(i)

with open("data/papers_db_updated_properly2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(update)'''

'''
with open('data/relations_db_updated.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    relations_db = list(csv_reader)

with open('data/papers_db_updated.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    papers_db_updated = list(csv_reader)

paper_aktuell = []
for rel in relations_db:
    x = ast.literal_eval(rel[3])
    paper_aktuell = paper_aktuell + x

neuer_file_ids = []
neuer_file = []
for papa in papers_db_updated:
    if len(papa) == 0:
        continue
    if len(papa) == 1:
        papa = ast.literal_eval(papa[0])
    if papa[4] in paper_aktuell:
        neuer_file_ids.append(papa[4])
        neuer_file.append(papa)


missing = list(set(paper_aktuell) - set(neuer_file_ids))
letzte_additions = []
for missing_id in missing:
    ids = getArticleIdsForKeyword("PMC", missing_id+'[pmid]')
    data = extractArticlesData("PMC", ids)
    data = get_PMC_info(data)
    #print(missing_id, data)
    if data == []:
        data = extractArticlesData("PubMed", missing_id)
        data = get_PM_info(data, missing_id)
        if data != []:
            data = list(data[0])
            letzte_additions.append(data)
        #print(missing_id, data)
    else:
        letzte_additions.append(data[0])

neuer_file = neuer_file + letzte_additions


with open("data/papers_db_updated_properly.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(neuer_file)



df_papers = pd.read_csv('data/papers_db2.csv', header=None)

def get_papersids_for_synlist(synlist):
    main_syn = synlist[-1]
    paperids = []
    for relation in relations_db:
        if relation[1] == main_syn:
            new = ast.literal_eval(relation[3])
            paperids = paperids + new
    return paperids

def get_paper_from_id(paper_id):

    row = df_papers.loc[df_papers[4] == int(paper_id)]
    title = row.iloc[0, 0]
    abstract = row.iloc[0, 1]
    full_text = row.iloc[0, 2]
    return title

get_paper_from_id('14517187')


synonyms = get_all_pki_synonyms().values()

wrong_papers = []
faulty_pkis = []
for liste in synonyms:
    if [e for e in liste if ' ' in e] != []:
        faulty_pkis.append(liste[-1])
        pki_papers = get_papersids_for_synlist(liste)
        for paper_id in pki_papers:
            title = get_paper_from_id(paper_id)
            if [e for e in liste if e in title.upper()] == []:
                wrong_papers.append(paper_id)


updated_relations = []
for relation in relations_db:
    relation_papers = ast.literal_eval(relation[3])
    wronguns = list(set(wrong_papers).intersection(relation_papers))
    if wronguns != [] and relation[1] in faulty_pkis:
        eliminated = [x for x in relation_papers if x not in wronguns]
        if eliminated != []:
            updated_relations.append([relation[0], relation[1], relation[2], eliminated])
    else:
        updated_relations.append([relation[0], relation[1], relation[2], relation_papers])

print(len(wrong_papers))
print(len(updated_relations))

with open("data/relations_db_updated.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(updated_relations)'''



'''
nums = {}
for pki in lines:
    #print(pki)
    rcts = get_rcts_from_searchterm(pki)
    print(pki, len(rcts))
    nums[pki] = (len(rcts))
    for rct in rcts:
        diseases = NER_diseases(rct[0])
        print(rct[0], diseases)'''

'''
import json
f = open('data/PKIs')
pkis = json.load(f)
for pki in pkis:
    drug_indications = pki.get('drug_indications')
    if len(drug_indications) != 0:
        for disease in drug_indications:
            get_rcts_from_searchterm(pki.get('pref_name'), disease)'''