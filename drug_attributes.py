# get attributes for all drugs via chembl


from chembl_webresource_client.new_client import new_client
import json


def get_chembl_attributes(name, stage=4):
    #gets molecule name and returns all chembl information, including all diseases molecule has been approved to treat

    molecule = new_client.molecule
    drug_indication = new_client.drug_indication

    mols = molecule.filter(pref_name__iexact=name)
    if len(mols) == 0:
        return {'drug_indications':[]}
    moldic = list(mols)[0]
    chembl_id = moldic.get('molecule_chembl_id')
    drug_in = drug_indication.filter(molecule_chembl_id=chembl_id) #returns list of all diseases drug has been tested for

    diseases = []
    for ind in list(drug_in):
        if ind.get('max_phase_for_ind') == stage: #we are only interested in diseases where drug is in stage 4 (meaning approved)
            diseases.append(ind.get('efo_term'))

    moldic['drug_indications'] = diseases
    return moldic


def get_drug_indication_numbers():
    with open('data/protein_kinase_inhibitors.txt') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    all_pkis= []
    x=0
    for i in lines:
        print(x)
        x=x+1
        #print(i)
        attributes = get_chembl_attributes(i)
        all_pkis.append(attributes)

    import json
    with open('data/PKIs', 'w') as fout:
        json.dump(all_pkis, fout)


def get_all_pki_synonyms():
    #Returns a dict of all PKI's and their synonyms
    #As data is a little chaotic and there exist synonyms, the data is structured and multiple synonyms are removed

    f = open('data/PKIs')
    pkis = json.load(f)

    full = {}
    all_synonyms = []

    for pki in pkis:
        chembl_id = pki.get('molecule_chembl_id')
        pref_name = pki.get('pref_name')
        if pref_name == None:
            continue
        synlist = pki.get('molecule_synonyms')
        if synlist != None:
            synonyms = []
            for i in synlist:
                synonym_i = i.get('molecule_synonym').upper()
                if synonym_i not in all_synonyms:
                    synonyms.append(synonym_i)
                    all_synonyms.append(synonym_i)
            synonyms = list(dict.fromkeys(synonyms)) #remove duplicates
            if pref_name not in synonyms:
                synonyms.append(pref_name)
            else: #if preferred name of pki is in the list we put it to the back so we know which one it is
                synonyms.append(synonyms.pop(synonyms.index(pref_name)))
        full[chembl_id] = synonyms

    return full

def get_targets_for_molecule(chembl_id):
    mechanism = new_client.mechanism
    res = mechanism.filter(parent_molecule_chembl_id=chembl_id).filter()
    print(res)
    return res


def get_all_chembl_ids():
    f = open('data/PKIs')
    pkis = json.load(f)
    idlist = []
    for pki in pkis:
        chembl_id = pki.get('molecule_chembl_id')
        if chembl_id != None:
            idlist.append(chembl_id)
    return idlist



def get_chembl_id_for_name(name):
    f = open('data/PKIs')
    pkis = json.load(f)
    for pki in pkis:
        if name == pki.get('pref_name'):
            chembl_id = pki.get('molecule_chembl_id')
            return chembl_id
    return None


def get_name_for_chemb_id(name):
    f = open('data/PKIs')
    pkis = json.load(f)
    for pki in pkis:
        if name == pki.get('molecule_chembl_id'):
            chembl_id = pki.get('pref_name')
            return chembl_id
    return None


def get_chembl_to_smiles_dict():
    f = open('data/PKIs')
    pkis = json.load(f)
    pki_smile = {}
    for i in pkis:
        ID = i.get('molecule_chembl_id')
        print(ID)
        try:
            smile = i.get('molecule_structures').get('canonical_smiles')
        except:
            smile = None
        pki_smile[ID] = smile
    return pki_smile


def extract_mechanisms():
    import pickle
    with open('data/mechanisms.pickle', 'rb') as handle:
        mechanisms = pickle.load(handle)

    return mechanisms
    '''all_ids = get_all_chembl_ids()
    target = new_client.mechanism
    mechanism_dict = {}
    for id in all_ids:
        print(id)
        dinger = []
        res = target.filter(action_type='INHIBITOR', parent_molecule_chembl_id=id).only('mechanism_of_action', 'target_chembl_id')
        for i in res:
            print(i)
            dinger.append(i['target_chembl_id'])
        mechanism_dict[id] = dinger


    with open('data/mechanisms.pickle', 'wb') as handle:
        pickle.dump(mechanism_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)'''


def get_mechanism_df():
    import pandas as pd

    mechanisms = extract_mechanisms()
    all_targets = []
    for i in mechanisms.keys():
        all_targets = all_targets + mechanisms[i]
    all_targets = list(set(all_targets))
    matrix = []
    for i in mechanisms.keys():
        row = []
        for target in all_targets:
            if target in mechanisms[i]:
                row.append(1)
            else:
                row.append(0)
        row.append(i)
        matrix.append(row)
    all_targets.append('Drug')
    df = pd.DataFrame(matrix, columns = all_targets)
    df['Drug'] = df['Drug'].apply(get_name_for_chemb_id)
    df.set_index('Drug', inplace=True)
    return df


def get_poss_drug_indications_for_drug(id):
    result = []
    target = new_client.drug_indication
    res = target.filter(molecule_chembl_id=id).only('efo_term')
    for i in res:
        result.append(i['efo_term'])
    return result