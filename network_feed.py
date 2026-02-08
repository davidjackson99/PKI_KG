# structure and feed data into network


from tools.scraper import getArticleIdsForKeyword, extractArticlesData, split
from tools.paper_infos import get_PMC_info, get_PM_info
from drug_attributes import get_all_pki_synonyms
from NER import NER_diseases
from paper_review import robotreview
import math
import csv
import ast
import pandas as pd

def get_rcts_from_searchterm(synonym_list, disease=None):
    search_term = synonym_list[0] + '[Title]'
    all_data_listed = []
    for i in range(1, len(synonym_list)):
        search_term = search_term + ' OR ' + synonym_list[i] + '[Title]'
    if disease==None:
        search_term = search_term+' AND (Randomized Controlled Trial[Publication Type] OR Clinical Trial[Publication Type])'
    else:
        search_term = search_term+' AND '+disease+'[Title/Abstract] AND (Randomized Controlled Trial[Publication Type] OR Clinical Trial[Publication Type])'
    ids = getArticleIdsForKeyword('PubMed', search_term)
    # there might be too many articles for 1 xml file, split if necessary
    if len(ids) == 0:
        return []
    n = math.ceil(len(ids) / 300)
    a_sets = split(ids, n)
    for i in range(len(a_sets)):
        data = extractArticlesData("PubMed", a_sets[i - 1])

        data_listed = get_PM_info(data, a_sets[i - 1])
        all_data_listed = all_data_listed + data_listed

    return all_data_listed

def pipeline_feed():

    #take all pki's (incl synonyms), search for all RCT's they appear in
    #for all rct's where pki + some disease appears in title, save pki+rct+paper

    all_pkis = (get_all_pki_synonyms().values())
    all_pkis = list(all_pkis)
    relations_db = []
    diseases_db = []
    papers_db = []
    for pki in all_pkis:
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

    '''
    import csv
    with open("data/relations_dbnew.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(relations_db)

    with open("data/diseases_db.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(diseases_db)

    with open("data/papers_db2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(papers_db)'''

    return relations_db, diseases_db, papers_db


def network_review_feed():
    from paper_review import extract_PFS, extract_OS, extract_HR
    df_papers = pd.read_csv('data/prev_data/papers_db_updated.csv', header=None)
    relplusrev = []
    with open('data/relations_db_updated.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        relations_db = list(csv_reader)
    pmc_ids = [[el[3]] for el in relations_db]
    pki_names = [[el[1]] for el in relations_db]
    relation_entry = relations_db
    for x in range(len(pmc_ids)):
        pmc_ids[x] = ast.literal_eval(pmc_ids[x][0])
    for i in range(len(pmc_ids)):
        subset = pmc_ids[i]
        pki = pki_names[i]
        sub_reviews = []
        for paper_id in subset:
            row = df_papers.loc[df_papers[4] == int(paper_id)]
            title = row.iloc[0,0]
            abstract = row.iloc[0,1]
            full_text = row.iloc[0, 2]
            review = robotreview([title, abstract, full_text], with_bias=False)
            if review != None:
                review = review[0]
            else:
                print('FAIL: ', paper_id)
                continue
            review['PFS'] = extract_PFS(abstract, pki)
            review['OS'] = extract_OS(abstract, pki)
            review['HR'] = extract_HR(abstract)
            review['PAPER_ID'] = paper_id

            print(review['effect'], ' ++++ ', review['punchline_text'], paper_id, review['is_rct'])
            sub_reviews.append(review)
        relplusrev.append(relation_entry[i] + [sub_reviews])

    with open("data/rels_with_rev.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(relplusrev)


network_review_feed()


#27013651 27915408 27915408 24990615

def add_PMC_full_text():
    import sys
    csv.field_size_limit(sys.maxsize)
    updated_file = []
    with open('data/prev_data/papers_db2.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        papers_db = list(csv_reader)
    for x in papers_db:
        ids = getArticleIdsForKeyword('PMC', x[4]+'[pmid]')
        if ids != []:
            data = extractArticlesData("PMC", ids)
            data = get_PMC_info(data)
            updated_file.append(data)
        else:
            x.insert(2,'No PMC data')
            del x[4] #delete article type as it is unnecessary from here and we want all rows to have same entries
            updated_file.append(x)

    with open("data/prev_data/papers_db_updated.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(updated_file)


'''with open('data/papers_db2.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    papers_db = list(csv_reader)'''


'''update = []
for i in papers_db_updated:
    if len(i) == 5:
        update.append(i)
    elif len(i) == 1:
        x = ast.literal_eval(i[0])
        update.append(x)


#no abstracts
for i in update:
    if i[4] in ['3861626', '3519789']:
        print(i)
    #if len(i[1]) < 8:
        #print(i)'''

'''
import ast
nu_db = []
for i in papers_db_updated:
    if len(i) == 1:
        x = ast.literal_eval(i[0])
        nu_db.append(x)
    else:
        nu_db.append(i)

old_ids = [el[4] for el in papers_db]
new_ids = []
for i in nu_db:
    if len(i) == 5:
        new_ids.append(i[4])

print(old_ids)
print(new_ids)
print(len(old_ids), len(new_ids))

main_list = list(set(old_ids) - set(new_ids))

for i in main_list:
    ids = getArticleIdsForKeyword('PMC', i+'[pmid]')
    if ids != []:
        data = extractArticlesData("PMC", ids)
        data = get_PMC_info(data)
        try:
            z = data[0][1]
        except:
            print(i, data)
        if data[0][1] != ['NO ABSTRACT FOUND']:
        print(data[0][4], data[0][2])'''