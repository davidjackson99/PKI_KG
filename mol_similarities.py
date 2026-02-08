# further file for affinity and similarity comparison



import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import itertools
import pickle
import os.path

def tanimoto_calc(smi1, smi2):
    #taken from: https://medium.com/data-professor/how-to-calculate-molecular-similarity-25d543ea7f40
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s



def load_tanimoto_similarities_matrix():
    if os.path.exists('data/PKI_mol_similarities_matrix.pkl'):
        with open('data/PKI_mol_similarities_matrix.pkl', 'rb') as handle:
            pki_similarities_matrix = pickle.load(handle)
        return pki_similarities_matrix

    #file doesnt exist yet
    f = open('data/PKIs')
    pkis = json.load(f)
    pki_smile = {}
    for i in pkis:
        name = i.get('pref_name')
        try:
            smile = i.get('molecule_structures').get('canonical_smiles')
        except:
            smile = None
        pki_smile[name] = smile
    pki_smile = {key:val for key, val in pki_smile.items() if val != None}
    comparisons = [[pki1,pki2, tanimoto_calc(pki_smile[pki1], pki_smile[pki2])] for (pki1, pki2) in itertools.product(pki_smile.keys(), pki_smile.keys())]
    with open('data/prev_data/PKI_mol_similarities.pkl', 'wb') as f:
        pickle.dump(comparisons, f)
    #this had to undergo some additional data cleaning, proper result in PKI_mol_similarities_matrix
    return comparisons



from drug_attributes import get_chembl_to_smiles_dict, get_name_for_chemb_id
def plot_tanimoto_vs_target_similarity(target_affinities, PKI):
    chembl_id_list = list(target_affinities.columns)
    aff_values = list(target_affinities.loc[:,PKI])
    tanimoto_values = []
    smiles = get_chembl_to_smiles_dict()
    for i in chembl_id_list:
        #print(PKI, i)
        tanimoto_values.append(tanimoto_calc(smiles[PKI], smiles[i]))


    import matplotlib.pyplot as plt

    plt.scatter(tanimoto_values, aff_values, s=[len(chembl_id_list)*0.04])
    plt.title(get_name_for_chemb_id(PKI))
    plt.xlabel('tanimoto similarity')
    plt.ylabel('target similarity')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    top_values = sorted(tanimoto_values+aff_values, reverse=True)[:6]
    for i, txt in enumerate(chembl_id_list):
        if tanimoto_values[i] in top_values or aff_values[i] in top_values:
            ax.annotate(get_name_for_chemb_id(txt), (tanimoto_values[i], aff_values[i]))
    plt.show()

