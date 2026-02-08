# extract and arrange data from FAERS individual xml files

import xml.etree.ElementTree as ET
from drug_attributes import get_all_pki_synonyms
import csv
import os
import ast
import pickle
from collections import Counter

pki_synonyms = get_all_pki_synonyms()
pki_synonyms = [item for sublist in pki_synonyms.values() for item in sublist]

def extract_casereports(file, name):
    tree = ET.parse(file)
    root = tree.getroot()
    all = []

    for x in root.iter('safetyreport'):

        reportid = x.find('safetyreportid').text
        serious = x.find('serious').text
        drugs, events, drugcharacterizations, drugindic = [], [], [], []
        patientagegroup, patientsex, duplicate, duplicatenumb = None, None, None, None
        for element in list(x.iter()):
            if element.tag == "medicinalproduct":
                drugs.append(''.join(element.itertext()))
            elif element.tag == "reactionmeddrapt":
                events.append(''.join(element.itertext()))
            elif element.tag == "patientagegroup":
                patientagegroup = ''.join(element.itertext())
            elif element.tag == "patientsex":
                patientsex = ''.join(element.itertext())
            elif element.tag == "drugcharacterization":
                drugcharacterizations.append(''.join(element.itertext()))
            elif element.tag == 'duplicate':
                duplicate = ''.join(element.itertext())
            elif element.tag == 'duplicatenumb':
                duplicatenumb = ''.join(element.itertext())

        if not set(drugs).isdisjoint(pki_synonyms):
            events = [i.upper() for i in events]
            if bool(set(drugindic) & set(events)):
                updated_events = [x for x in events if x not in drugindic]
                events = updated_events
            if events != []:
                all.append({'reportid': reportid, 'drugs': drugs, 'events': list(set(events)), 'drugcharacterizations': drugcharacterizations,
                        'patientagegroup': patientagegroup, 'patientsex': patientsex, 'serious': serious, 'duplicate': duplicate, 'duplicatenumb': duplicatenumb})

    with open(name+'.csv', 'w', encoding='utf8', newline='') as output_file:
        fc = csv.DictWriter(output_file,
                            fieldnames=all[0].keys(),

                            )
        fc.writeheader()
        fc.writerows(all)
    return all



def extract_all_records():

    for year in ['2016','2017','2018','2019','2020','2021']:
        for Q in ['Q1','Q2','Q3','Q4']:
            dirname = "/Users/dj_jnr_99/Downloads/faers_xml_"+year+Q+"/xml" #directory where all the FAERS files are stored
            for filename in os.listdir(dirname):
                if not filename.endswith('.xml'): continue
                fullname = os.path.join(dirname, filename)
                new_filename = './FAERS/'+ fullname.rsplit('/', 1)[-1][:-4]
                print(fullname, new_filename)
                extract_casereports(fullname, new_filename)


def get_drug_condition_dict():
    synlist = list(get_all_pki_synonyms().values())
    drugs_conds = {}

    for filename in os.listdir('./data/FAERS/'):
        print(filename)
        with open('/Users/dj_jnr_99/Downloads/master-thesis/data/FAERS/'+filename, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            file = list(csv_reader)

        for entry in file[1:]:
            drugs = ast.literal_eval(entry[1])
            conditions = ast.literal_eval(entry[2])
            for syns in synlist:
                if not set(syns).isdisjoint(drugs):
                    if syns[-1] in drugs_conds.keys():
                        drugs_conds[syns[-1]] = drugs_conds[syns[-1]] + conditions
                    else:
                        drugs_conds[syns[-1]] = conditions

    newfile = []
    for key in drugs_conds:
        newfile.append([key, dict(Counter(drugs_conds[key]))])

    with open('data/PKI-AE-occurrences.pkl', 'wb') as f:
        pickle.dump(newfile, f)


def dis_counter_update(file, dict):
    tree = ET.parse(file)
    root = tree.getroot()

    for x in root.iter('safetyreport'):
        for element in list(x.iter()):
            if element.tag == "reactionmeddrapt" and ''.join(element.itertext()).upper() in dict.keys():
                dict[''.join(element.itertext()).upper()] = dict[''.join(element.itertext()).upper()] + 1
    return dict

def get_disease_numbers():
    with open('data/PKI-AE-occurrences.pkl', 'rb') as handle:
        aeoccur = pickle.load(handle)

    all_diseases = [i[1].keys() for i in aeoccur]
    all_diseases = list(set([item for sublist in all_diseases for item in sublist]))

    #initialize dataset
    disease_counter = {}
    for disease in all_diseases:
        disease_counter[disease] = 0
    for year in ['2016','2017','2018','2019','2020','2021']: #2022 added in later
        for Q in ['Q1','Q2','Q3','Q4']:
            dirname = "/Users/dj_jnr_99/Downloads/faers_xml_"+year+Q+"/xml"
            for filename in os.listdir(dirname):
                if not filename.endswith('.xml'): continue
                fullname = os.path.join(dirname, filename)
                print(fullname, disease_counter)
                disease_counter = dis_counter_update(fullname, disease_counter)

    with open('data/total_disease_occurences.pkl', 'wb') as f:
        pickle.dump(disease_counter, f)



'''           use of adverse events data for PRR calculation etc.             '''


def calc_PRR(PKI_AEs, dis_nums):
    PRRs = {}
    ab, cd = sum(PKI_AEs.values()), sum(dis_nums.values())
    for event in PKI_AEs:
        a, c = PKI_AEs[event], dis_nums[event]
        PRR = (a/ab)/(c/cd)
        #POR = (a*(cd-c))/(c*(ab-a))
        if c > 5000: # we do a cutoff for # total event occurrences, as events with very few total occurrences may have unreasonably high impact on PRR score of PKI
            PRRs[event] = PRR
    PRRs = dict(sorted(PRRs.items(), key=lambda item: item[1], reverse=True))
    return PRRs

def get_PKI_AE_table(top_x=None): #top_x only returns top x adverse events for each PKI
    import pandas as pd
    from drug_attributes import get_chembl_id_for_name

    pkis, aes, pki_ids = [], [], []
    with open('data/total_disease_occurences.pkl', 'rb') as handle:
        dis_nums = pickle.load(handle)
    with open('data/PKI-AE-occurrences.pkl', 'rb') as handle:
        PKIAE = pickle.load(handle)
    for PKI in PKIAE:
        pki_dict = calc_PRR(PKI[1], dis_nums)
        pki_dict = {key: val for key, val in pki_dict.items() if val > 1}
        if top_x != None:
            aes.append(list(pki_dict.keys())[:top_x])
        else:
            aes.append(list(pki_dict.keys()))
        pkis.append(PKI[0])
        pki_ids.append(get_chembl_id_for_name(PKI[0]))

    pki_ae_table = pd.DataFrame({'CHEMBL_ID': pki_ids, 'PKI': pkis, 'Adverse Events (sorted by relevance)': aes})
    return pki_ae_table

#print(get_PKI_AE_table(10))