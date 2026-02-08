# extract tissue data from gtexplorer and link to AE's and drugs



import pickle

def get_kinase_tissue_matrix():
    gtex_link = 'https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct.gz'
    expression_data = pd.read_csv(gtex_link, sep='\t', index_col='gene_id', skiprows=2)

    kinases=[]
    with open('//data/all_proteine_kinases', 'r') as f:
        for line in f:
            line = line.strip()
            kinases.append(line)

    with open('data/affinities_data/TAS_values .pkl', 'rb') as f:
        affinities = pickle.load(f)

    gene_ids = list(set([el[6] for el in affinities]))
    df = expression_data.loc[expression_data['Description'].isin(gene_ids)] #remove all genes that aren't expressed in our PKI's
    df = df[df['Description'].isin(kinases)] #remove all rows that aren't kinases
    return df

def get_normalized_kinase_tissue_matrix(sort_values=None):

    df = get_kinase_tissue_matrix()
    df = df.sort_values('Description')
    df = df.set_index('Description')
    normalized_df = df.div(df.sum(axis=1), axis=0)
    #normalized_df['Specificity'] = df.sum(axis=1)
    if sort_values != None:
        normalized_df = normalized_df.sort_values(sort_values)
    return normalized_df

#print(get_normalized_kinase_tissue_matrix('Testis')['Testis'])


from affinities import create_affinities_matrix

def get_PKI_tissue_list(tissue):
    tissue_df = get_kinase_tissue_matrix()
    adj_df = create_affinities_matrix()
    adj_df = adj_df.set_index('CHEMBL_ID')
    adj_df = adj_df.sort_index(axis=1)
    pkis = tissue_df.index.values.tolist() #get all pkis from tissue matrix...
    adj_df = adj_df[adj_df.columns.intersection(pkis)]  #...and only keep those in the adjacency matrix
    values = tissue_df[tissue].tolist() #get values for tissue row
    adj_df[adj_df == 10], adj_df[adj_df == 1], adj_df[adj_df == 2], adj_df[adj_df == 3] = 0, 5, 2.5, 1 #change values
    adj_df = adj_df.mul(values, axis=1) #muliply each row with the tissue values
    adj_df['Sum'] = adj_df.sum(axis=1)
    adj_df = adj_df.sort_values('Sum')
    return adj_df


def get_PKI_tissue_matrix():
    tissue_df = get_normalized_kinase_tissue_matrix()
    #tissue_df = tissue_df.drop('Specificity', axis=1)
    adj_df = create_affinities_matrix()
    adj_df = adj_df.set_index('CHEMBL_ID')
    adj_df = adj_df.sort_index(axis=1)
    pkis = tissue_df.index.values.tolist()  # get all pkis from tissue matrix...
    adj_df = adj_df[adj_df.columns.intersection(pkis)]  # ...and only keep those in the adjacency matrix
    adj_df[adj_df == 10], adj_df[adj_df == 1], adj_df[adj_df == 2], adj_df[adj_df == 3] = 0, 5, 2.5, 1  # change values
    sum_dict = {}
    tissues = list(tissue_df.columns.values)
    for tissue in tissues:
        values = tissue_df[tissue].tolist()  # get values for tissue row
        new_df = adj_df.mul(values, axis=1)  # muliply each row with the tissue values
        sum_dict[tissue + ' sum'] = new_df.sum(axis=1).to_list()
    for sum in sum_dict:
        adj_df[sum] = sum_dict[sum]
    return adj_df.iloc[: , 405:]


from adverse_events import get_PKI_AE_table
import pandas as pd
def get_PKI_tissue_AE_matrix():
    adverse_events = get_PKI_AE_table(3)
    df = get_PKI_tissue_matrix()
    maxes = df.idxmax(axis=1).to_frame()
    PKI_tissue_AE_matrix = pd.merge(maxes, adverse_events, on='CHEMBL_ID', how='inner')
    print(PKI_tissue_AE_matrix.to_string())

#print(get_kinase_tissue_matrix())
'''df = get_kinase_tissue_matrix()
df = df.set_index('Description')
df['Specificity'] = df.sum(axis=1)
df.loc['Total'] = df.sum(numeric_only=True)
summen = df.sum(numeric_only=True)
#df = df.sort_values('Specificity')
print(summen)'''

'''df = get_kinase_tissue_matrix()
df['max'] = df.max(axis=1)
df.set_index('Description', inplace=True)
df['id max'] = df.idxmax(axis=1)
df = df.iloc[:,53:]
print(df.to_string())
print(sorted(df['max'].tolist()))'''
'''df['dif'] = df.max(axis=1) - df.min(axis=1)
df = df.iloc[:,-1:]
print(df.to_string())'''
#print(df.to_string())
#print(df)


