#extract and process affinities data, compute drug similarities

import pickle
from drug_attributes import get_all_chembl_ids
import os
import itertools
import csv
import numpy as np
from math import sqrt
import pandas as pd

def get_needed_sms_ids():
    # file from https://labsyspharm.shinyapps.io/smallmoleculesuite/?_inputs_&amp;tab=%22selectivity%22
    # returns affinity data for many molecules, this function takes this data and returns only the ids which
    # are desired (PKI's in our dataset). Therefor we match labsyspharm ids to chembl ids
    if os.path.exists('data/affinities_data/sms_ids.pkl'):
        with open('data/affinities_data/sms_ids.pkl', 'rb') as handle:
            smsids = pickle.load(handle)
        return smsids
    else:
        with open('data/affinities_data/smsid_to_chembl_id.pkl', 'rb') as f:
            all_ids = pickle.load(f)
        chembl_ids = get_all_chembl_ids()
        needed = []
        for i in all_ids:
            if i[2] in chembl_ids:
                needed.append(i[0])
        #some sms ids cover multiple chembl ids
        with open('data/affinities_data/sms_ids.pkl', 'wb') as f:
            pickle.dump(list(set(needed)), f)
    return list(set(needed))



with open('data/affinities_data/smsid_to_chembl_id.pkl', 'rb') as handle:
    xaffmatches = pickle.load(handle)
def match_ids(sms_id):
    chembl_ids = get_all_chembl_ids()
    for i in xaffmatches:
        if sms_id == i[0] and i[2] in chembl_ids:
            return i[2]
    return None

def create_affinities_list():
    smsids = get_needed_sms_ids()

    with open('data/affinities_data/lsp_biochem_agg.csv', newline='') as f:
        reader = csv.reader(f)
        affinity_data = list(reader)

    needed_data = []
    for i in affinity_data:
        if i[1] in smsids:
            chembl_id = match_ids(i[1])
            affinity_relation = i+[chembl_id]
            needed_data.append(affinity_relation)
    return needed_data
    '''
    with open('data/affinities_data/affinities.pkl', 'wb') as f:
        pickle.dump(needed_data, f)'''



def create_tas_list():
    smsids = get_needed_sms_ids()

    with open('data/affinities_data/lsp_tas.csv', newline='') as f:
        reader = csv.reader(f)
        affinity_data = list(reader)

    needed_data = []
    for i in affinity_data:
        if i[1] in smsids:
            chembl_id = match_ids(i[1])
            affinity_relation = i+[chembl_id]
            needed_data.append(affinity_relation)

    with open('data/affinities_data/TAS_values .pkl', 'wb') as f:
        pickle.dump(needed_data, f)

    return needed_data



def create_affinities_matrix(gene_subset=None):
    #creates 210 x 684 matrix of affinity values for each PKI
    with open('data/affinities_data/TAS_values .pkl', 'rb') as f:
        affinities = pickle.load(f)

    gene_ids = list(set([el[6] for el in affinities]))
    if gene_subset != None:
        gene_ids = list(set(gene_subset) & set(gene_ids))
    chembl_ids = list(set([el[7] for el in affinities]))

    #affinity=10 means no relation and we thus assume that NaN values mean no relation, we initialize matrix with 1s
    matrix = []
    for pki in chembl_ids:
        row = {'CHEMBL_ID': pki}
        for gene in gene_ids:
            row[gene] = 10
        matrix.append(row)

    for affinity in affinities:
        if gene_subset != None and affinity[6] not in gene_subset:
            continue
        for i in range(len(matrix)):
            if matrix[i]['CHEMBL_ID'] == affinity[7]:
                value = int(affinity[3])
                matrix[i][affinity[6]] = value
    return pd.DataFrame(matrix)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(x, y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def weighted_jaccard(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    q = np.concatenate([x, y], axis=1)
    q1 = (q[:, 0] < 4)
    q2 = (q[:, 1] < 4)
    q = q[q1 | q2] #we only take values where C1 or C2 has value 3 or less
    if len(q) >= 5:
        return np.sum(np.amin(q, axis=1)) / np.sum(np.amax(q, axis=1))
    else:
        return 0


def get_weighted_adjacency_matrix(inputmethod='jaccard',gene_subset=None):
    affinities = create_affinities_matrix(gene_subset).values.tolist()
    chids = [i[0] for i in affinities]
    weight_list = np.zeros((len(chids), len(chids))) #initialize weight list
    if inputmethod=='cosine':
        method = cosine_similarity
    elif inputmethod=='euclidian':
        method = euclidean_distance
    elif inputmethod=='jaccard':
        method = weighted_jaccard
    else:
        print('Only methods available are jaccard, cosine, and euclidian.')

    for a, b in itertools.combinations(affinities, 2):
        chindex1 = chids.index(a[0])
        chindex2 = chids.index(b[0])
        similarity = method(a[1:], b[1:])
        weight_list[chindex1][chindex2] = similarity
        weight_list[chindex2][chindex1] = similarity

    adj_matrix = pd.DataFrame.from_dict(dict(zip(chids, weight_list))) #convert to dataframe

    np.fill_diagonal(adj_matrix.values, 1.0) #scaling without 0s made 0s negatative, so replace negative values
    from drug_attributes import get_name_for_chemb_id

    drugnames = []
    for i in adj_matrix.columns:
        drugnames.append(get_name_for_chemb_id(i))

    adj_matrix = adj_matrix.set_axis(drugnames, axis=1, inplace=False)
    adj_matrix.index = adj_matrix.columns
    return adj_matrix

'''
with open('/Users/davidjackson/Downloads/lsp_biochem_agg.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

import numpy as np


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


affinities = []
for i in data:
    if i[1] == '96316':
        affinities.append(float(i[3]))
        print(i)

normalized = NormalizeData(affinities)'''


'''import collections
import matplotlib.pyplot as plt
jmatrix = get_weighted_adjacency_matrix('weighted_jaccard')

print(jmatrix)
print(jmatrix.loc[:,'CHEMBL3402762'])'''



'''
data = ee.values.tolist()
data = [round(item,1) for sublist in data for item in sublist]
print(data)
w = collections.Counter(data)
plt.bar(w.keys(), w.values(), width=0.05)
plt.show()
'''

'''df = create_affinities_matrix()
print(len(list(df)))
#keep = [c for c in list(df) if 1 in list(df[c]) or 2 in list(df[c]) or 3 in list(df[c])]
keep = [c for c in list(df) if 1 in list(df[c])]
df = df[keep]
print(len(keep))'''

#print(df.sum(axis=0).to_string())

#print(sorted(list(df.sum(axis=0))[1:]))
'''print(df[['CHEMBL_ID', 'Total']].to_string())
print(max(list(df['Total'])))
print(sorted(df['Total'].values.tolist()))'''