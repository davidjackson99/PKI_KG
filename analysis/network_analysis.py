# basic measures for network analysis


import csv
import sys
import ast
import networkx as nx
import math

########## NDC ############

csv.field_size_limit(sys.maxsize)

def get_relations():
    with open('//data/relations_db_updated.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        return list(csv_reader)

def get_papers():
    with open('//data/papers_db.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        return list(csv_reader)

def get_diseases():
    with open('//data/diseases_db_updated.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        return list(csv_reader)

def get_PKIS():
    import json
    f = open('data/PKIs')
    pkis = json.load(f)
    idlist = []
    for pki in pkis:
        chembl_id = pki.get('pref_name')
        if chembl_id != None:
            idlist.append(chembl_id)
    return idlist


diseases = get_diseases()
relations = get_relations()
G = nx.Graph()

from drug_attributes import get_name_for_chemb_id

for i in relations:
    if len(ast.literal_eval(i[3])) > 10:
        G.add_edge(i[1], i[2], weight=len(ast.literal_eval(i[3])))


diseases = get_diseases()
relations = get_relations()
G = nx.Graph()

for i in relations:
    G.add_edge(i[1], i[2], weight=len(ast.literal_eval(i[3])))

g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data='weight')}
nx.set_edge_attributes(G, g_distance_dict, 'distance')

closeness = nx.closeness_centrality(G)
closeness = {k: v for k, v in sorted(closeness.items(), key=lambda item: item[1])}

def get_name_from_conc(concept):
    for i in diseases:
        if i[1] == concept:
            return i[0]

import pandas as pd
import numpy as np
def get_recommendation(root):

    pkis = get_PKIS()
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if e2 in pkis:
                commons = commons_dict.get(e2)
                if commons == None:
                    commons_dict.update({e2: [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2: commons})
    movies = []
    weight = []
    for key, values in commons_dict.items():
        w = 0.0
        for e in values:
            w = w + 1 / math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)
    return result

#result = get_recommendation('ERLOTINIB')
#print(result.to_string())

import matplotlib.pyplot as plt

def get_all_adj_nodes(list_in):
    sub_graph=set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):
                sub_graph.add(e)
    return list(sub_graph)


def draw_sub_graph(sub_graph):
    pkis = get_PKIS()
    subgraph = G.subgraph(sub_graph)
    colors=[]
    sizes = []
    for e in subgraph.nodes():
        if e not in pkis:
            sizes.append(50)
            colors.append('blue')
        else:
            sizes.append(10)
            colors.append('red')

    nx.draw(subgraph, with_labels=False, font_weight='bold',node_color=colors, node_size=sizes)
    plt.show()


'''reco=list(result.index[3:5].values)
print(list(result.index[:10]))
reco.extend(["C0007131"])
sub_graph = get_all_adj_nodes(reco)
draw_sub_graph(sub_graph)'''




'''


t = sorted(G.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)

import re
def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))'''


'''count = 0
countcanc = 0
for i in x_cond:
    name = get_name_from_conc(i[0])
    if 'neoplasm' not in name.lower()  and 'cancer' not in name.lower() and 'carcinoma' not in name.lower() and 'tumor' not in name.lower() and name.lower()[-3:] != 'oma' and 'lymphoma' not in name.lower() and 'sarcoma' not in name.lower() and 'lioblastoma' not in name.lower() and 'melanoma' not in name.lower() and 'thymoma' not in name.lower():
        count += i[1]
    else:
        countcanc += i[1]'''




########## NDD ############

'''from affinities import get_weighted_adjacency_matrix
import numpy as np
import collections

df = get_weighted_adjacency_matrix()
df.values[[np.arange(df.shape[0])]*2] = 0

fulllist=[]
for i in df.columns:
    col_list = df[i].values.tolist()
    fulllist = fulllist + col_list

fulllist = list(filter((0.0).__ne__, fulllist))

import matplotlib.pyplot as plt
import numpy as np

# To generate an array of x-values
x = fulllist


G = nx.Graph()
from drug_attributes import get_name_for_chemb_id

drugnames=[]
for i in df.columns:
    drugnames.append(get_name_for_chemb_id(i))

df = df.set_axis(drugnames, axis=1, inplace=False)


for i in df.columns:
    G.add_node(i)

import itertools
df.index = df.columns
for a, b in itertools.combinations(df.columns, 2):
    G.add_edge(a,b, weight=df.loc[a, b])

#print(nx.density(G))

dataFrame = df.sum(axis = 0)
#print(sorted(dataFrame.values.tolist()))
#print(dataFrame.to_string())

for i in df.columns:
    print(i, df[i].sum())'''

'''xA = fulllist

yA = np.random.normal(1, 0.1, len(xA))
import random

rand_list=[]
for i in range(len(yA)):
    rand_list.append(random.randint(20,50))

plt.scatter(xA, yA, s=rand_list, alpha=0.5)
plt.show()'''


'''import pandas as pd
from drug_attributes import get_chembl_id_for_name
import pickle
from adverse_events import calc_PRR
from affinities import create_affinities_matrix
from drug_attributes import get_name_for_chemb_id

affinities_df = create_affinities_matrix()
names = affinities_df['CHEMBL_ID'].tolist()
names = [get_name_for_chemb_id(i) for i in names]
affinities_df['Names'] = names
affinities_df.set_index('Names', inplace=True)
affinities_df = affinities_df.drop('CHEMBL_ID', axis=1)
print(affinities_df)
pkis, aes, pki_ids = [], [], []
full_list = []
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/total_disease_occurences.pkl', 'rb') as handle:
    dis_nums = pickle.load(handle)
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/PKI-AE-occurrences.pkl', 'rb') as handle:
    PKIAE = pickle.load(handle)
for PKI in PKIAE:
    pki_dict = calc_PRR(PKI[1], dis_nums)
    pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
    full_list.append([PKI[0], pki_dict])

xxx = []'''


'''for i in full_list:
    if 'PNEUMONITIS' in i[1].keys():
        #print(i[0], i[1]['PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME'])
        try:
            s = affinities_df.loc[i[0]]
            s = s[s == 1]
            xxx = xxx + list(s.index)
            print('x')
            #print(affinities_df.loc[i[0]]['BRAF'])
        except:
            continue
'''

def get_all_common_neighbors():
    relations = get_relations()
    drugs = list(set([i[1] for i in relations]))
    drugdiseasedict = {}
    for i in relations:
        if i[1] in drugdiseasedict.keys():
            drugdiseasedict[i[1]] = drugdiseasedict[i[1]] + [i[2]] * len(ast.literal_eval(i[3]))
        else:
            drugdiseasedict[i[1]] = [i[2]]

    similaritydicts = {}
    min_rels = 4
    import itertools
    for v1, v2 in itertools.combinations(drugs, 2):
        #common_neighbors = len(list(set(drugdiseasedict[v1]) & set(drugdiseasedict[v2]))) #weights disregarded
        #common_neighbors = len(list((Counter(drugdiseasedict[v1]) & Counter(drugdiseasedict[v2])).elements())) #weights regarded
        if len(drugdiseasedict[v1]) < min_rels or len(drugdiseasedict[v2]) < min_rels:
            continue
        common_neighbors = len(list(set(drugdiseasedict[v1]) & set(drugdiseasedict[v2])))
        alpha = 0.6
        result = alpha * common_neighbors / (len(drugdiseasedict[v1]) + len(drugdiseasedict[v2])) + (1 - alpha) * common_neighbors / 300
        similaritydicts[v1+'+'+v2] = common_neighbors
        #similaritydicts[v1 + '+' + v2] = result

    similaritydicts = {k: v for k, v in sorted(similaritydicts.items(), key=lambda item: item[1])}
    return similaritydicts

#get_all_common_neighbors()


#print(common_neighbor_similarity('SUNITINIB', 'IMATINIB'))

#disease_scores = get_all_common_neighbors()

from drug_attributes import get_mechanism_df, extract_mechanisms
import pandas as pd
import pickle
import numpy as np
from adverse_events import calc_PRR

def pearson_correlation_event_gene(event='METASTASES TO LUNG'):
    mech_df = get_mechanism_df()
    drugs = mech_df.index.values.tolist()
    full_list = {}
    with open('//data/total_disease_occurences.pkl', 'rb') as handle:
        dis_nums = pickle.load(handle)
    with open('//data/PKI-AE-occurrences.pkl', 'rb') as handle:
        PKIAE = pickle.load(handle)
    for PKI in PKIAE:
        pki_dict = calc_PRR(PKI[1], dis_nums)
        pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
        if PKI[0] in full_list.keys():
            full_list[PKI[0]] = full_list[PKI] + list(pki_dict.keys())
        else:
            full_list[PKI[0]] = list(pki_dict.keys())

    event_binaries = []
    for i in drugs:
        try:
            if event in full_list[i]:
                event_binaries.append(1)
            else:
                event_binaries.append(0)
        except:
            event_binaries.append(2)

    mech_df['event'] = event_binaries
    mech_df = mech_df[mech_df.event != 2]
    mech_df = mech_df.loc[:, (mech_df != 0).any(axis=0)]


    correlation_dict = {}
    for col in mech_df.columns:
        if 1 not in mech_df[col].tolist() :
            print(mech_df[col])
            continue
        rho = np.corrcoef(mech_df[col].tolist(), mech_df['event'].tolist())
        correlation_dict[col] = rho[0][1]

    print({k: v for k, v in sorted(correlation_dict.items(), key=lambda item: item[1])})

def drug_pearson_correlations_mechanisms():

    mech_df = get_mechanism_df()
    mech_df = mech_df.loc[:, (mech_df != 0).any(axis=0)]
    drugs = mech_df.index.to_list()
    correlation_dict = {}
    import itertools
    for d1, d2 in itertools.combinations(drugs, 2):
        d1_vals = mech_df.loc[d1].tolist()
        d2_vals = mech_df.loc[d2].tolist()
        if 1 not in d1_vals or 1 not in d2_vals:
            continue
        rho = np.corrcoef(d1_vals, d2_vals)[0][1]
        correlation_dict[d1+d2] = rho
        #print(d1, d2, rho)

    print({k: v for k, v in sorted(correlation_dict.items(), key=lambda item: item[1])})


def drug_prediction_similarity_correlation():

    mechanisms = extract_mechanisms()
    new_mechanisms = {}
    from drug_attributes import get_name_for_chemb_id
    for i in mechanisms.keys():
        new_mechanisms[get_name_for_chemb_id(i)] = mechanisms[i]


    drug_to_side_effects = {}
    with open('//data/total_disease_occurences.pkl', 'rb') as handle:
        dis_nums = pickle.load(handle)
    with open('//data/PKI-AE-occurrences.pkl', 'rb') as handle:
        PKIAE = pickle.load(handle)
    for PKI in PKIAE:
        pki_dict = calc_PRR(PKI[1], dis_nums)
        pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
        if PKI[0] in drug_to_side_effects.keys():
            drug_to_side_effects[PKI[0]] = drug_to_side_effects[PKI] + list(pki_dict.keys())
        else:
            drug_to_side_effects[PKI[0]] = list(pki_dict.keys())

    #average number of side effects, delete few AE entries
    from affinities import get_weighted_adjacency_matrix
    adjacency_matrix = get_weighted_adjacency_matrix()
    mech_df = get_mechanism_df()


    affinities = []
    commmon_adverse_events = []
    chembltargets = []
    diseasecorrs = []
    import itertools
    for a, b in itertools.combinations(drug_to_side_effects.keys(), 2):
        if len(drug_to_side_effects[a]) < 15 or len(drug_to_side_effects[b]) < 15:
            continue
        try:
            #affinities.append(adjacency_matrix.loc[a][b])
            chembltargets.append(len(list(set(new_mechanisms[a]).intersection(new_mechanisms[b]))))
            '''keyword = ''
            if a+'+'+b in disease_scores.keys():
                keyword = a+'+'+b
            else:
                keyword = b+'+'+a'''
            diseasecorrs.append(len(list(set(drug_to_side_effects[a]).intersection(drug_to_side_effects[b]))))
            #chembltargets.append(len(list(set(new_mechanisms[a]).intersection(new_mechanisms[b]))))
            #commmon_adverse_events.append(len(list(set(drug_to_side_effects[a]).intersection(drug_to_side_effects[b]))))
            #print('RHO: ', rho)
        except:
            continue


    print(np.mean(diseasecorrs),np.median(diseasecorrs))
    print(np.mean(chembltargets),np.median(chembltargets))
    from scipy.stats import spearmanr
    median_correlation = []
    aemean = np.mean(diseasecorrs)
    targetmean = np.mean(chembltargets)
    for i in range(len(diseasecorrs)):
        if diseasecorrs[i] > aemean and chembltargets[i] > targetmean:
            median_correlation.append(1)
        elif diseasecorrs[i] < aemean and chembltargets[i] < targetmean:
            median_correlation.append(1)
        else:
            median_correlation.append(0)

    from collections import Counter
    print(Counter(median_correlation))

    print(np.corrcoef(chembltargets, diseasecorrs))
    print(spearmanr(chembltargets, diseasecorrs))


def sankey_suni_erlo():
    #print(pearson_correlation_event_gene('PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME'))
    full_list = {}
    with open('//data/total_disease_occurences.pkl', 'rb') as handle:
        dis_nums = pickle.load(handle)
    with open('//data/PKI-AE-occurrences.pkl', 'rb') as handle:
        PKIAE = pickle.load(handle)
    for PKI in PKIAE:
        pki_dict = calc_PRR(PKI[1], dis_nums)
        pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
        if PKI[0] in full_list.keys():
            full_list[PKI[0]] = full_list[PKI] + list(pki_dict.keys())
        else:
            full_list[PKI[0]] = list(pki_dict.keys())

    sora = full_list['SORAFENIB']
    suni = full_list['SUNITINIB']
    vemu = full_list['VEMURAFENIB']

    sorasuni = list(set(sora).intersection(suni))
    soravemu = list(set(sora).intersection(vemu))
    sunivemu = list(set(suni).intersection(vemu))
    onlysora = [i for i in sora if i not in sorasuni and i not in soravemu]
    onlyvemu = [i for i in vemu if i not in soravemu and i not in sunivemu]
    onlysuni = [i for i in suni if i not in sorasuni and i not in sunivemu]

    print('SORAFENIB-SUNITINIB =',sorasuni)
    print('SORAFENIB-VEMURAFENIB =', soravemu)
    print('SUNITINIB-VEMURAFENIB =', sunivemu)
    print('SORAFENIB =', onlysora)
    print('VEMURAFENIB =', onlyvemu)
    print('SUNITINIB =', onlysuni)

    both = ['GINGIVAL BLEEDING', 'DECREASED APPETITE', 'BLISTER', 'ORAL PAIN', 'SKIN FISSURES', 'EATING DISORDER', 'BONE MARROW FAILURE', 'METASTASES TO LIVER', 'PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME', 'FEEDING DISORDER', 'MUCOSAL INFLAMMATION']
    onlysora = ['HEPATOCELLULAR CARCINOMA', 'METASTASES TO LUNG', 'ASCITES', 'METASTASES TO BONE', 'BLOOD BILIRUBIN INCREASED', 'HEPATIC FAILURE', 'JAUNDICE', 'THYROID CANCER', 'METASTASES TO CENTRAL NERVOUS SYSTEM', 'HEPATIC CANCER', 'HEPATIC CIRRHOSIS', 'HYPOPHAGIA', 'FAECES DISCOLOURED', 'HAEMATEMESIS', 'ABASIA', 'HEPATIC FUNCTION ABNORMAL', 'SKIN LESION', 'PAIN OF SKIN', 'DYSPHONIA', 'ACUTE MYELOID LEUKAEMIA']



    import pandas as pd

    # Create example dataset
    df = pd.DataFrame()
    df['source']= ['SORAFENIB', 'SUNITINIB', 'SORAFENIB', 'B-RAF']
    df['target']= ['OTHER GENES', 'OTHER GENES', 'B-RAF', 'AE']
    df['weight']= [10, 10, 3, 1]

    # Import library
    from d3blocks import D3Blocks

    # Initialize
    d3 = D3Blocks()

    d3.sankey(df, link={"color": "source-target"}, filepath='Sankey_demo_1.html')


'''lunggenes = ['GEFITINIB', 'ERLOTINIB', 'OSIMERTINIB', 'CRIZOTINIB', 'AFATINIB', 'ALECTINIB', 'Vandetanib'.upper(), 'Ceritinib'.upper(), 'Anlotinib'.upper(), 'Brigatinib'.upper()]
lunggenes = [get_chembl_id_for_name(i) for i in lunggenes]

sources = []
targets  = []

for x in lunggenes:
    sources = sources + [get_name_for_chemb_id(x)] * len(mechanisms[x])
    targets = targets + mechanisms[x]

import itertools
for a, b in itertools.combinations(mechanisms, 2):
    if a == 'CHEMBL1336' or b == 'CHEMBL1336':
        continue
    #if (a in lunggenes or b in lunggenes) and len(mechanisms[a]) > 2 and len(mechanisms[b]) > 2:
    if a == 'CHEMBL939' and mechanisms[a] == mechanisms[b]:
        sources.append(get_name_for_chemb_id(b))
        targets.append('CHEMBL203')
        print(get_name_for_chemb_id(a),get_name_for_chemb_id(b))

    if a in lunggenes and len(mechanisms[a]) > 1 and b not in lunggenes:
        if len(list(set(mechanisms[a]).intersection(mechanisms[b]))) > 1:
            sources = sources + [get_name_for_chemb_id(b)]*len(mechanisms[b])
            targets = targets + mechanisms[b]


    elif b in lunggenes and len(mechanisms[b]) > 1 and a not in lunggenes:
        if len(list(set(mechanisms[a]).intersection(mechanisms[b]))) > 1:
            sources = sources + [get_name_for_chemb_id(a)] * len(mechanisms[a])
            targets = targets + mechanisms[a]



print(sources)
print(targets)
print(len(sources), len(targets))'''

'''from drug_attributes import get_name_for_chemb_id

import pickle

sources = ['GEFITINIB', 'ERLOTINIB', 'OSIMERTINIB', 'OSIMERTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'AFATINIB', 'AFATINIB', 'AFATINIB', 'ALECTINIB', 'ALECTINIB', 'ALECTINIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'CERITINIB', 'CERITINIB', 'CERITINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'BRIGATINIB', 'BRIGATINIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'LAPATINIB', 'LAPATINIB', 'PELITINIB', 'ICOTINIB', 'EPITINIB', 'THELIATINIB', 'SIMOTINIB', 'ROCILETINIB', 'NAQUOTINIB', 'NAZARTINIB', 'OLMUTINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'TANDUTINIB', 'TANDUTINIB', 'TANDUTINIB', 'TANDUTINIB', 'TANDUTINIB', 'TANDUTINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DOVITINIB', 'DOVITINIB', 'DOVITINIB', 'DOVITINIB', 'DOVITINIB', 'QUIZARTINIB', 'QUIZARTINIB', 'QUIZARTINIB', 'QUIZARTINIB', 'QUIZARTINIB', 'LESTAURTINIB', 'LESTAURTINIB', 'LESTAURTINIB', 'LESTAURTINIB', 'LESTAURTINIB', 'LESTAURTINIB', 'LORLATINIB', 'LORLATINIB', 'ALLITINIB', 'ALLITINIB', 'VARLITINIB', 'VARLITINIB', 'SAPITINIB', 'SAPITINIB', 'SAPITINIB', 'PYROTINIB', 'PYROTINIB', 'TESEVATINIB', 'TESEVATINIB', 'TESEVATINIB', 'CANERTINIB', 'CANERTINIB', 'CANERTINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'LORLATINIB', 'LORLATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'AMUVATINIB', 'AMUVATINIB', 'AMUVATINIB', 'AMUVATINIB', 'LORLATINIB', 'LORLATINIB', 'PUQUITINIB', 'PUQUITINIB', 'PUQUITINIB', 'PUQUITINIB', 'PEXIDARTINIB', 'PEXIDARTINIB', 'PEXIDARTINIB', 'PEXIDARTINIB', 'PEXIDARTINIB', 'PEXIDARTINIB', 'GLESATINIB', 'GLESATINIB', 'GLESATINIB', 'GLESATINIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'PAZOPANIB', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'CEDIRANIB', 'CEDIRANIB', 'CEDIRANIB', 'SEMAXANIB', 'SEMAXANIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'VATALANIB', 'VATALANIB', 'VATALANIB', 'VATALANIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB']
targets = ['CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL3717', 'CHEMBL4247', 'CHEMBL3883330', 'CHEMBL2111387', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL4247', 'CHEMBL2041', 'CHEMBL3883330', 'CHEMBL267', 'CHEMBL4601', 'CHEMBL4128', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL2363043', 'CHEMBL2363049', 'CHEMBL4247', 'CHEMBL3883330', 'CHEMBL2111387', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL4247', 'CHEMBL203', 'CHEMBL1906', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL1974', 'CHEMBL5145', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL1844', 'CHEMBL1936', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL1844', 'CHEMBL1936', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2096618', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2096618', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL3009', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2068', 'CHEMBL2363074', 'CHEMBL2096618', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2068', 'CHEMBL2363074', 'CHEMBL2096618', 'CHEMBL2742', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL1844', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2041', 'CHEMBL1974', 'CHEMBL2971', 'CHEMBL2815', 'CHEMBL4898', 'CHEMBL5608', 'CHEMBL2041', 'CHEMBL4247', 'CHEMBL3883330', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL1824', 'CHEMBL5838', 'CHEMBL203', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL5147', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL1936', 'CHEMBL1868', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1936', 'CHEMBL1868', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL4247', 'CHEMBL3883330', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL3717', 'CHEMBL2007', 'CHEMBL4247', 'CHEMBL3883330', 'CHEMBL3559703', 'CHEMBL203', 'CHEMBL279', 'CHEMBL1913', 'CHEMBL1844', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1844', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL3717', 'CHEMBL2689', 'CHEMBL2095227', 'CHEMBL4128', 'CHEMBL2095227', 'CHEMBL2095227', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2093867', 'CHEMBL279', 'CHEMBL4722', 'CHEMBL2185', 'CHEMBL279', 'CHEMBL2742', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1844', 'CHEMBL1862', 'CHEMBL1906', 'CHEMBL1936', 'CHEMBL3650', 'CHEMBL3961', 'CHEMBL4142', 'CHEMBL4223', 'CHEMBL5122', 'CHEMBL5145', 'CHEMBL4128', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL2068', 'CHEMBL2815', 'CHEMBL2095227', 'CHEMBL2095227', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL1844', 'CHEMBL1936', 'CHEMBL258', 'CHEMBL2742', 'CHEMBL3650', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2959', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2093867', 'CHEMBL279', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL1936', 'CHEMBL2095227', 'CHEMBL1936', 'CHEMBL2041', 'CHEMBL2095227', 'CHEMBL2095189', 'CHEMBL4722', 'CHEMBL2185', 'CHEMBL279', 'CHEMBL2742', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1844', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL1844', 'CHEMBL1862', 'CHEMBL1906', 'CHEMBL1936', 'CHEMBL3650', 'CHEMBL3961', 'CHEMBL4142', 'CHEMBL4223', 'CHEMBL5122', 'CHEMBL5145', 'CHEMBL4128', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL2068', 'CHEMBL2815', 'CHEMBL1936', 'CHEMBL2041', 'CHEMBL2095227', 'CHEMBL2095189']
important_targets = ['CHEMBL1824', 'CHEMBL1906', 'CHEMBL267', 'CHEMBL2363049', 'CHEMBL4128', 'CHEMBL203', 'CHEMBL5145', 'CHEMBL3717', 'CHEMBL4601', 'CHEMBL3883330', 'CHEMBL2041', 'CHEMBL3009', 'CHEMBL2363043', 'CHEMBL1936', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1974', 'CHEMBL2111387', 'CHEMBL1913', 'CHEMBL2095227', 'CHEMBL4247']

drugs = ['GEFITINIB', 'ERLOTINIB', 'OSIMERTINIB', 'OSIMERTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'AFATINIB', 'AFATINIB', 'AFATINIB', 'ALECTINIB', 'ALECTINIB', 'ALECTINIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'CERITINIB', 'CERITINIB', 'CERITINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'BRIGATINIB', 'BRIGATINIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB']
drugs = list(set(drugs))

drug_to_side_effects = {}
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/total_disease_occurences.pkl', 'rb') as handle:
    dis_nums = pickle.load(handle)
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/PKI-AE-occurrences.pkl', 'rb') as handle:
    PKIAE = pickle.load(handle)
for PKI in PKIAE:
    pki_dict = calc_PRR(PKI[1], dis_nums)
    pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
    if PKI[0] in drug_to_side_effects.keys():
        drug_to_side_effects[PKI[0]] = drug_to_side_effects[PKI] + list(pki_dict.keys())
    else:
        drug_to_side_effects[PKI[0]] = list(pki_dict.keys())

for i in drugs:
    print(i, len(drug_to_side_effects[i]), drug_to_side_effects[i])'''



def pearson_correlation_gene_event(gene):
    mech_df = get_mechanism_df()
    mech_df = mech_df[[gene]]
    drugs = mech_df.index.values.tolist()
    full_list = {}
    with open('//data/total_disease_occurences.pkl', 'rb') as handle:
        dis_nums = pickle.load(handle)
    with open('//data/PKI-AE-occurrences.pkl', 'rb') as handle:
        PKIAE = pickle.load(handle)
    for PKI in PKIAE:
        pki_dict = calc_PRR(PKI[1], dis_nums)
        pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
        if PKI[0] in full_list.keys():
            full_list[PKI[0]] = full_list[PKI] + list(pki_dict.keys())
        else:
            full_list[PKI[0]] = list(pki_dict.keys())

    events = [item for sublist in list(full_list.values()) for item in sublist]
    events = [item for item in events if events.count(item) > 10]
    drugs = [i for i in drugs if i in list(full_list.keys())]
    mech_df = mech_df[mech_df.index.isin(drugs)]

    for event in events:
        event_binaries = []
        for i in drugs:
            if event in full_list[i]:
                event_binaries.append(1)
            else:
                event_binaries.append(0)

        mech_df[event] = event_binaries

    correlation_dict = {}
    for col in mech_df.columns:
        if 1 not in mech_df[col].tolist() :
            print(mech_df[col])
            continue
        rho = np.corrcoef(mech_df[col].tolist(), mech_df[gene].tolist())
        correlation_dict[col] = rho[0][1]

    side_effects = list({k for k, v in sorted(correlation_dict.items(), key=lambda item: item[1]) if v > 0.2 and v < 0.9})

    print(side_effects)
    return {k: v for k, v in sorted(correlation_dict.items(), key=lambda item: item[1])}


'''mech_df = get_mechanism_df()
gene_to_adverse_events_dict = {}
for col in mech_df.columns:
    gene_to_adverse_events_dict[col] = pearson_correlation_gene_event(col)
    

with open('data/gene_to_adverse_events_dict.pkl', 'wb') as f:
    pickle.dump(gene_to_adverse_events_dict, f)    
'''
'''full_list = {}
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/total_disease_occurences.pkl', 'rb') as handle:
    dis_nums = pickle.load(handle)
with open('/Users/dj_jnr_99/Downloads/master-thesis/data/PKI-AE-occurrences.pkl', 'rb') as handle:
    PKIAE = pickle.load(handle)
for PKI in PKIAE:
    pki_dict = calc_PRR(PKI[1], dis_nums)
    pki_dict = {key: val for key, val in pki_dict.items() if val > 3}
    if PKI[0] in full_list.keys():
        full_list[PKI[0]] = full_list[PKI] + list(pki_dict.keys())
    else:
        full_list[PKI[0]] = list(pki_dict.keys())

from drug_attributes import get_name_for_chemb_id'''

def precision_recall_simple_drug_prediction():
    with open('data/gene_to_adverse_events_dict.pkl', 'rb') as handle:
        gene_to_adverse_events_dict = pickle.load(handle)

    mechanisms = extract_mechanisms()
    tp = []
    fn = []
    fp = []
    for i in mechanisms:
        try:
            drug_aes = full_list[get_name_for_chemb_id(i)]
        except:
            continue
        if len(drug_aes) > 20 and len(mechanisms[i]) > 2:
            gene_aes = []
            for gene in mechanisms[i]:
                gene_aes.append(list({k for k, v in sorted(gene_to_adverse_events_dict[gene].items(), key=lambda item: item[1]) if v > 0.2 and v < 0.9}))
            predicted_aes = list(set([item for sublist in gene_aes for item in sublist]))

            truepositive = list(set(drug_aes) & set(predicted_aes))
            falsenegative = list(set(drug_aes) - set(predicted_aes))
            falsepositive = list(set(predicted_aes) - set(drug_aes))

            tp.append(len(truepositive))
            fn.append(len(falsenegative))
            fp.append(len(falsepositive))

            #print(i, predicted_aes, drug_aes)

    print(sum(tp), sum(fn), sum(fp))



def get_structural_similarity_for_nsclc():
    from affinities import get_weighted_adjacency_matrix
    from mol_similarities import load_tanimoto_similarities_matrix

    sim = load_tanimoto_similarities_matrix()

    same_target_203 = ['ERLOTINIB','GEFITINIB','PELITINIB','ICOTINIB','EPITINIB','THELIATINIB','SIMOTINIB','ROCILETINIB','NAQUOTINIB','NAZARTINIB','OLMUTINIB', 'OSIMERTINIB']

    import itertools
    for a, b in itertools.combinations(same_target_203, 2):
        for x in sim:
            if a==x[0] and b==x[1] or a==x[1] and b==x[0]:
                print(a, b, x[2])




def get_all_nonapproved_drugnames():
    import json
    f = open('data/PKIs')
    pkis = json.load(f)
    idlist = []
    for pki in pkis:
        chembl_id = pki.get('pref_name')
        first_approval = pki.get('first_approval')
        if chembl_id != None and first_approval == None:
            idlist.append(chembl_id)
    return idlist

def get_ll_indications():
    import json
    f = open('data/PKIs')
    pkis = json.load(f)
    indications = []
    for pki in pkis:
        drug_indications = pki.get('drug_indications')
        if drug_indications != None:
            indications = indications + drug_indications
    return list(set(indications))

def get_indications_for_drug(drug):
    import json
    f = open('data/PKIs')
    pkis = json.load(f)
    for pki in pkis:
        chembl_id = pki.get('pref_name')
        if drug == chembl_id:
            return pki.get('drug_indications')


from scipy.stats import spearmanr
def pearson_correlation_gene_indication():
    nonapproved = get_all_nonapproved_drugnames()
    mech_df = get_mechanism_df()
    genes = list(mech_df.columns)
    mech_df = mech_df[~mech_df.index.isin(nonapproved)]
    all_indications = get_ll_indications()
    for indication in all_indications:
        specific_ind_rows = []
        for i in mech_df.index:
            print('uuu')
            if indication in get_indications_for_drug(i):
                specific_ind_rows.append(1)
            else:
                specific_ind_rows.append(0)

        mech_df[indication] = specific_ind_rows

    gene_to_indications = {}
    for gene in genes:
        keep = [gene] + all_indications
        newmech_df = mech_df[keep].copy()
        correlation_dict = {}
        for col in newmech_df.columns:
            if 1 not in newmech_df[col].tolist():
                print(newmech_df[col])
                continue
            rho = spearmanr(newmech_df[col].tolist(), newmech_df[gene].tolist())[0]
            correlation_dict[col] = rho

        indicationzzz = list({k for k, v in sorted(correlation_dict.items(), key=lambda item: item[1]) if v > 0.2 and v < 0.9})
        print({k for k, v in sorted(correlation_dict.items(), key=lambda item: item[1]) if v > 0.2 and v < 0.9})
        print(gene, indicationzzz)
        gene_to_indications[gene] = indicationzzz
    print(gene_to_indications)
    #return {k: v for k, v in sorted(correlation_dict.items(), key=lambda item: item[1])}

#pearson_correlation_gene_indication()
from drug_attributes import get_chembl_id_for_name, get_poss_drug_indications_for_drug
gene_to_indications = {'CHEMBL5491': [], 'CHEMBL2096667': [], 'CHEMBL2527': [], 'CHEMBL258': [], 'CHEMBL3650': ['colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'cancer', 'cholangiocarcinoma', 'colorectal adenocarcinoma'], 'CHEMBL2363062': ['rheumatoid arthritis', 'immune system disease'], 'CHEMBL2095217': ['idiopathic pulmonary fibrosis', 'bladder tumor', 'urothelial carcinoma', 'urinary bladder cancer'], 'CHEMBL1862': ['childhood acute lymphoblastic leukemia', 'colorectal carcinoma', 'acute lymphoblastic leukemia', 'colorectal neoplasm', 'chronic myelogenous leukemia', 'colorectal cancer', 'lymphoblastic lymphoma', 'childhood cancer', 'colorectal adenocarcinoma'], 'CHEMBL4899': [], 'CHEMBL5568': ['non-small cell lung carcinoma'], 'CHEMBL5331': [], 'CHEMBL3553': [], 'CHEMBL3130': ['chronic lymphocytic leukemia', 'lymphoma', 'follicular lymphoma', 'non-Hodgkins lymphoma', 'neoplasm of mature B-cells'], 'CHEMBL2363043': ['medullary thyroid gland carcinoma', 'thyroid cancer', 'thyroid carcinoma', 'papillary thyroid carcinoma'], 'CHEMBL3055': [], 'CHEMBL2096618': ['neoplasm', 'childhood acute lymphoblastic leukemia', 'acute lymphoblastic leukemia', 'chronic myelogenous leukemia', 'lymphoblastic lymphoma', 'childhood cancer'], 'CHEMBL279': ['acute myeloid leukemia', 'mast-cell leukemia', 'systemic mastocytosis', 'Mastocytosis'], 'CHEMBL1844': [], 'CHEMBL5251': ['Waldenstrom macroglobulinemia', 'chronic lymphocytic leukemia', 'macroglobulinemia'], 'CHEMBL4630': [], 'CHEMBL2815': [], 'CHEMBL5285': [], 'CHEMBL2959': [], 'CHEMBL2095189': ['acute myeloid leukemia', 'neoplasm', 'systemic mastocytosis', 'colorectal carcinoma', 'colorectal neoplasm', 'mast-cell leukemia', 'colorectal cancer', 'idiopathic pulmonary fibrosis', 'Mastocytosis', 'colorectal adenocarcinoma'], 'CHEMBL4439': [], 'CHEMBL3430911': [], 'CHEMBL2007': ['Gastrointestinal stromal tumor'], 'CHEMBL3559684': [], 'CHEMBL2964': ['metastatic melanoma', 'melanoma', 'lung cancer', 'cutaneous melanoma'], 'CHEMBL3559691': [], 'CHEMBL2041': ['thyroid cancer', 'colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'thyroid neoplasm', 'medullary thyroid gland carcinoma', 'colorectal adenocarcinoma'], 'CHEMBL2111387': ['non-small cell lung carcinoma', 'diffuse large B-cell lymphoma'], 'CHEMBL3961': [], 'CHEMBL1936': ['neoplasm', 'systemic mastocytosis', 'colorectal carcinoma', 'childhood acute lymphoblastic leukemia', 'colorectal neoplasm', 'mast-cell leukemia', 'colorectal cancer', 'lymphoblastic lymphoma', 'Mastocytosis', 'childhood cancer', 'colorectal adenocarcinoma', 'Gastrointestinal stromal tumor'], 'CHEMBL1981': [], 'CHEMBL4601': ['medullary thyroid gland carcinoma', 'thyroid cancer', 'thyroid carcinoma', 'papillary thyroid carcinoma'], 'CHEMBL1824': ['brain neoplasm', 'HER2 Positive Breast Carcinoma'], 'CHEMBL2068': ['childhood acute lymphoblastic leukemia', 'colorectal carcinoma', 'acute lymphoblastic leukemia', 'colorectal neoplasm', 'chronic myelogenous leukemia', 'colorectal cancer', 'lymphoblastic lymphoma', 'childhood cancer', 'colorectal adenocarcinoma'], 'CHEMBL1868': [], 'CHEMBL2148': [], 'CHEMBL5024': [], 'CHEMBL3009': [], 'CHEMBL308': [], 'CHEMBL2842': [], 'CHEMBL3116': [], 'CHEMBL5147': [], 'CHEMBL267': ['thyroid cancer', 'chronic myelogenous leukemia', 'thyroid carcinoma', 'papillary thyroid carcinoma', 'thyroid neoplasm', 'medullary thyroid gland carcinoma'], 'CHEMBL2689': [], 'CHEMBL2363049': ['medullary thyroid gland carcinoma', 'thyroid cancer', 'thyroid carcinoma', 'papillary thyroid carcinoma'], 'CHEMBL2971': ['rheumatoid arthritis', 'immune system disease'], 'CHEMBL2363074': ['acute lymphoblastic leukemia', 'chronic myelogenous leukemia'], 'CHEMBL3024': [], 'CHEMBL1955': [], 'CHEMBL3717': ['diffuse large B-cell lymphoma'], 'CHEMBL3629': [], 'CHEMBL2111455': ['breast carcinoma', 'breast neoplasm', 'breast cancer'], 'CHEMBL2695': [], 'CHEMBL4142': ['colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'cholangiocarcinoma', 'colorectal adenocarcinoma'], 'CHEMBL2996': [], 'CHEMBL4179': [], 'CHEMBL2973': [], 'CHEMBL3267': ['follicular lymphoma', 'neoplasm of mature B-cells', 'lymphoma', 'chronic lymphocytic leukemia'], 'CHEMBL4128': ['thyroid cancer', 'colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'thyroid carcinoma', 'papillary thyroid carcinoma', 'thyroid neoplasm', 'medullary thyroid gland carcinoma', 'colorectal adenocarcinoma'], 'CHEMBL5443': [], 'CHEMBL3883330': ['non-small cell lung carcinoma', 'cancer', 'lymphoma', 'diffuse large B-cell lymphoma'], 'CHEMBL4937': [], 'CHEMBL4036': [], 'CHEMBL2185': [], 'CHEMBL4040': [], 'CHEMBL2508': ['breast neoplasm'], 'CHEMBL3610': [], 'CHEMBL3905': ['chronic myelogenous leukemia'], 'CHEMBL3883293': [], 'CHEMBL3778': [], 'CHEMBL3385': [], 'CHEMBL2599': [], 'CHEMBL5122': [], 'CHEMBL4722': [], 'CHEMBL1974': ['acute myeloid leukemia', 'neoplasm', 'systemic mastocytosis', 'mast-cell leukemia', 'Mastocytosis'], 'CHEMBL1906': ['renal cell carcinoma', 'colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'colorectal adenocarcinoma'], 'CHEMBL3559703': [], 'CHEMBL2835': ['rheumatoid arthritis', 'immune system disease'], 'CHEMBL4223': [], 'CHEMBL4895': [], 'CHEMBL5608': [], 'CHEMBL1902': ['subependymal giant cell astrocytoma', 'clear cell renal carcinoma', 'renal cell carcinoma', 'Kidney Angiomyolipoma', 'anaplastic astrocytoma', 'Tuberous sclerosis', 'angiomyolipoma', 'hamartoma', 'neuroendocrine neoplasm', 'astrocytoma', 'pancreatic neuroendocrine tumor', 'immune system disease', 'Adenoma sebaceum', 'pulmonary neuroendocrine tumor', 'eye disease', 'breast neoplasm'], 'CHEMBL2276': [], 'CHEMBL2111289': [], 'CHEMBL1913': ['lymphoblastic lymphoma', 'acute lymphoblastic leukemia', 'childhood cancer', 'childhood acute lymphoblastic leukemia'], 'CHEMBL5838': [], 'CHEMBL2095227': ['neoplasm', 'colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'thyroid carcinoma', 'papillary thyroid carcinoma', 'idiopathic pulmonary fibrosis', 'cancer', 'renal carcinoma', 'thyroid neoplasm', 'colorectal adenocarcinoma'], 'CHEMBL4898': [], 'CHEMBL3145': [], 'CHEMBL5145': ['neoplasm', 'colorectal carcinoma', 'colorectal neoplasm', 'colorectal cancer', 'cancer', 'melanoma', 'metastatic melanoma', 'colorectal adenocarcinoma', 'cutaneous melanoma'], 'CHEMBL1907601': ['breast carcinoma', 'breast neoplasm', 'breast cancer'], 'CHEMBL301': [], 'CHEMBL5469': [], 'CHEMBL3430904': ['non-small cell lung carcinoma', 'thyroid cancer', 'medullary thyroid gland carcinoma'], 'CHEMBL4247': ['neoplasm', 'lymphoma', 'diffuse large B-cell lymphoma', 'non-small cell lung carcinoma', 'cancer'], 'CHEMBL1957': [], 'CHEMBL4005': ['breast carcinoma', 'breast neoplasm', 'breast cancer'], 'CHEMBL2637': [], 'CHEMBL3587': ['metastatic melanoma', 'melanoma', 'lung cancer', 'cutaneous melanoma'], 'CHEMBL2093867': ['acute myeloid leukemia'], 'CHEMBL203': [], 'CHEMBL2111353': [], 'CHEMBL2742': ['cholangiocarcinoma'], 'CHEMBL2111459': ['glaucoma'], 'CHEMBL3430888': ['non-small cell lung carcinoma', 'thyroid cancer', 'medullary thyroid gland carcinoma'], 'CHEMBL331': ['breast neoplasm'], 'CHEMBL3234': ['chronic myelogenous leukemia'], 'CHEMBL260': [], 'CHEMBL3559685': []}

with open('data/mechanisms.pickle', 'rb') as handle:
    mechanisms = pickle.load(handle)

nonapproved = get_all_nonapproved_drugnames()
xd = []
for drug in nonapproved:
    chembl_id = get_chembl_id_for_name(drug)
    drug_targets = mechanisms[chembl_id]
    indics = []
    for target in drug_targets:
        indics = indics + gene_to_indications[target]
    if len(indics) != 0:
        poss_chembl_indications = get_poss_drug_indications_for_drug(chembl_id)
        if len(poss_chembl_indications) != 0:
            if not set(poss_chembl_indications).isdisjoint(indics):
                xd.append(1)
            else:
                xd.append(0)

print(sum(xd)/len(xd))