# some of the D3 visualizations in the thesis


from d3blocks import D3Blocks
import pandas as pd

def nsclc_drug_to_gene_graph():

    drugs =['Erlotinib', 'Gefitinib', 'Osimertinib', 'Crizotinib', 'Crizotinib', 'Crizotinib', 'Crizotinib', 'Afatinib', 'Afatinib', 'Afatinib', 'Alectinib', 'Alectinib', 'Alectinib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Ceritinib', 'Ceritinib', 'Ceritinib', 'Anlotinib', 'Anlotinib', 'Anlotinib', 'Anlotinib', 'Brigatinib', 'Brigatinib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib']

    genes = ['erbB1', 'erbB1', 'erbB1', 'NPM/ALK', 'HGF', 'EML4-ALK', 'ALK', 'erbB-2', 'erbB-4', 'erbB1', 'ALK', 'EML4-ALK', 'RET', 'Ephrin_receptor', 'BRK', 'VEGFR', 'RET', 'TIE-2', 'SRC', 'erbB1', 'EML4-ALK', 'NPM/ALK', 'ALK', 'KIT', 'VEGFR2', 'PDGFR-beta', 'VEGFR3', 'erbB1', 'ALK', 'KIT', 'FLT3', 'RET', 'RAF', 'B-raf', 'VEGF', 'PDGFR-beta']

    d3 = D3Blocks()
    d = {"source": drugs, "target":genes, "weight": [1]*36}
    df = pd.DataFrame(d)


    d3.d3graph(df, showfig=True, charge=800)
    d3.D3graph.set_node_properties(color=None)

    for drug in list(set(drugs)):
        d3.D3graph.node_properties[drug]['color']='#000000'
        d3.D3graph.node_properties[drug]['size'] = 15

    for gene in list(set(genes)):
        if 'erb' in gene:
            d3.D3graph.node_properties[gene]['color'] = '#8141FF'
        elif 'ALK' in gene:
            d3.D3graph.node_properties[gene]['color'] = '#FF0000'
        elif 'VEGFR' in gene:
            d3.D3graph.node_properties[gene]['color'] = '#FAE958'
        else:
            d3.D3graph.node_properties[gene]['color'] = '#738276'

    d3.D3graph.set_edge_properties(directed=True, marker_end='arrow')
    d3.D3graph.show()


def graph_nsclc_discover():
    d3 = D3Blocks()
    lungdrugs =['Erlotinib', 'Gefitinib', 'Osimertinib', 'Crizotinib', 'Crizotinib', 'Crizotinib', 'Crizotinib', 'Afatinib', 'Afatinib', 'Afatinib', 'Alectinib', 'Alectinib', 'Alectinib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Vandetanib', 'Ceritinib', 'Ceritinib', 'Ceritinib', 'Anlotinib', 'Anlotinib', 'Anlotinib', 'Anlotinib', 'Brigatinib', 'Brigatinib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib', 'Sorafenib']
    lungdrugs = [i.upper() for i in lungdrugs]
    sources = ['GEFITINIB', 'ERLOTINIB', 'OSIMERTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'AFATINIB', 'AFATINIB', 'AFATINIB', 'ALECTINIB', 'ALECTINIB', 'ALECTINIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'CERITINIB', 'CERITINIB', 'CERITINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'BRIGATINIB', 'BRIGATINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'SUNITINIB', 'LAPATINIB', 'LAPATINIB', 'PELITINIB', 'ICOTINIB', 'EPITINIB', 'THELIATINIB', 'SIMOTINIB', 'ROCILETINIB', 'NAQUOTINIB', 'NAZARTINIB', 'OLMUTINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'IMATINIB', 'TANDUTINIB', 'TANDUTINIB', 'TANDUTINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'NERATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'DASATINIB', 'LORLATINIB', 'LORLATINIB', 'ALLITINIB', 'ALLITINIB', 'VARLITINIB', 'VARLITINIB', 'SAPITINIB', 'SAPITINIB', 'SAPITINIB', 'PYROTINIB', 'PYROTINIB', 'TESEVATINIB', 'TESEVATINIB', 'TESEVATINIB', 'CANERTINIB', 'CANERTINIB', 'CANERTINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'FAMITINIB', 'LORLATINIB', 'LORLATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'TELATINIB', 'LORLATINIB', 'LORLATINIB', 'PUQUITINIB', 'PUQUITINIB', 'PUQUITINIB', 'PUQUITINIB', 'GLESATINIB', 'GLESATINIB', 'GLESATINIB', 'GLESATINIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'TIVOZANIB', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'MIDOSTAURIN', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'ENMD-2076', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'REGORAFENIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB', 'MOTESANIB', "CERITINIB", "CRIZOTINIB", "ALECTINIB"]
    targets = ['CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL3717', 'CHEMBL4247', 'CHEMBL4247', 'CHEMBL2111387', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL4247', 'CHEMBL2041', 'CHEMBL4247', 'CHEMBL267', 'CHEMBL4601', 'CHEMBL4128', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL2363043', 'CHEMBL203', 'CHEMBL4247', 'CHEMBL4247', 'CHEMBL2111387', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL4247', 'CHEMBL203', 'CHEMBL1844', 'CHEMBL1936', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL203', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2096618', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL3009', 'CHEMBL1862', 'CHEMBL1913', 'CHEMBL1936', 'CHEMBL2068', 'CHEMBL2363074', 'CHEMBL2096618', 'CHEMBL4247', 'CHEMBL4247', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL1824', 'CHEMBL5838', 'CHEMBL203', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL5147', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL203', 'CHEMBL1824', 'CHEMBL3009', 'CHEMBL1936', 'CHEMBL1868', 'CHEMBL1974', 'CHEMBL2095189', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL4247', 'CHEMBL4247', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL4247', 'CHEMBL4247', 'CHEMBL3559703', 'CHEMBL203', 'CHEMBL279', 'CHEMBL1913', 'CHEMBL3717', 'CHEMBL2689', 'CHEMBL2095227', 'CHEMBL4128', 'CHEMBL2095227', 'CHEMBL2095227', 'CHEMBL1936', 'CHEMBL1913', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL2095189', 'CHEMBL2093867', 'CHEMBL279', 'CHEMBL4722', 'CHEMBL2185', 'CHEMBL279', 'CHEMBL2742', 'CHEMBL1974', 'CHEMBL1936', 'CHEMBL1844', 'CHEMBL1862', 'CHEMBL1906', 'CHEMBL1936', 'CHEMBL3650', 'CHEMBL3961', 'CHEMBL4142', 'CHEMBL4223', 'CHEMBL5122', 'CHEMBL5145', 'CHEMBL4128', 'CHEMBL2095189', 'CHEMBL2095227', 'CHEMBL2041', 'CHEMBL2068', 'CHEMBL2815', 'CHEMBL1936', 'CHEMBL2041', 'CHEMBL2095227', 'CHEMBL2095189', "EML4-ALK", "EML4-ALK",  "EML4-ALK"]
    print(sources.index("LORLATINIB"))
    del sources[sources.index("LORLATINIB")]
    del targets[sources.index("LORLATINIB")]
    del sources[sources.index("LORLATINIB")]
    del targets[sources.index("LORLATINIB")]
    del sources[sources.index("LORLATINIB")]
    del targets[sources.index("LORLATINIB")]
    del sources[sources.index("LORLATINIB")]
    del targets[sources.index("LORLATINIB")]
    important_targets = ["EML4-ALK", 'CHEMBL1824', 'CHEMBL267', 'CHEMBL2363049', 'CHEMBL4128', 'CHEMBL203', 'CHEMBL3717', 'CHEMBL4601', 'CHEMBL4247', 'CHEMBL2041', 'CHEMBL3009', 'CHEMBL2363043', 'CHEMBL1936', 'CHEMBL279', 'CHEMBL1955', 'CHEMBL2111387', 'CHEMBL1913', 'CHEMBL2095227', 'CHEMBL4247']
    weight = [1]*len(sources)
    d = {"source": sources, "target":targets, "weight": weight}
    df = pd.DataFrame(d)
    df.loc[df.weight != 1] = 1

    d3.d3graph(df, charge=500, filepath='drugdisc.html')
    d3.D3graph.set_node_properties(color=None)
    chembl_to_id_dict = {"EML4-ALK": "EML4-ALK", 'CHEMBL1824': 'erbB2', 'CHEMBL267': 'SRC', 'CHEMBL2363049': 'erbB1', 'CHEMBL4128': 'TIE-2', 'CHEMBL203': 'erbB1', 'CHEMBL3717': 'HGF', 'CHEMBL4601': 'BRK', 'CHEMBL3883330': 'ALK', 'CHEMBL2041': 'RET', 'CHEMBL3009': 'erbB4', 'CHEMBL2363043': 'Ephrin_receptor', 'CHEMBL1936': 'KIT', 'CHEMBL279': 'VEGFR2', 'CHEMBL1955': 'VEGFR3', 'CHEMBL2111387': 'NPM/ALK', 'CHEMBL1913': 'PDGFRbeta', 'CHEMBL2095227': 'VEGFR3', 'CHEMBL4247': 'ALK'}
    for ex in list(set(targets)):
        #print(d3.D3graph.node_properties[ex])
        if ex in important_targets:
            d3.D3graph.node_properties[ex]['color']='#ff4c33'
            d3.D3graph.node_properties[ex]['size'] = 14
            d3.D3graph.node_properties[ex]['name'] = chembl_to_id_dict[ex]
            d3.D3graph.node_properties[ex]['label'] = chembl_to_id_dict[ex]
        else:
            d3.D3graph.node_properties[ex]['color'] = '#ff6433'
            d3.D3graph.node_properties[ex]['size'] = 4
            d3.D3graph.node_properties[ex]['label'] = ''

    for ex in list(set(sources)):
        if ex in lungdrugs:
            d3.D3graph.node_properties[ex]['color']='#33ff6b'
            d3.D3graph.node_properties[ex]['size'] = 10
        else:
            d3.D3graph.node_properties[ex]['color'] = '#9f33ff'
            d3.D3graph.node_properties[ex]['size'] = 10
    d3.D3graph.set_edge_properties(directed=True, marker_end='arrow')

    d3.D3graph.show()


#graph_nsclc_discover()

import pickle
with open('/Users/davidjackson/Downloads/mechanisms.pickle', 'rb') as f:
   mechanisms = pickle.load(f)

import itertools

sources = []
targets = []
weights = []

for a, b in itertools.combinations(mechanisms.keys(), 2):
    if len(list(set(mechanisms[a]) & set(mechanisms[b]))) > 1:
        if len(list(set(mechanisms[a]))) >= len(list(set(mechanisms[b]))):
            if len(list(set(mechanisms[b])))/len(list(set(mechanisms[a]))) > 0.6:
                sources.append(a), targets.append(b), weights.append(len(list(set(mechanisms[b])))/len(list(set(mechanisms[a]))))
        else:
            if len(list(set(mechanisms[a]))) / len(list(set(mechanisms[b]))) > 0.6:
                sources.append(b), targets.append(a), weights.append(len(list(set(mechanisms[a]))) / len(list(set(mechanisms[b]))))

import json
def get_name_for_chemb_id(name):
    f = open('/Users/davidjackson/Downloads/PKIs')
    pkis = json.load(f)
    for pki in pkis:
        if name == pki.get('molecule_chembl_id'):
            chembl_id = pki.get('pref_name')
            return chembl_id
    return None

graph_nsclc_discover()


def sankey_graph():
    both = ['GINGIVAL BLEEDING', 'BLISTER', 'ORAL PAIN', 'SKIN FISSURES', 'EATING DISORDER', 'BONE MARROW FAILURE',
            'METASTASES TO LIVER', 'PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME', 'FEEDING DISORDER',
            'MUCOSAL INFLAMMATION']
    onlysora = ["PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME", 'HEPATOCELLULAR CARCINOMA', 'METASTASES TO LUNG',
                'ASCITES', 'METASTASES TO BONE', 'BLOOD BILIRUBIN INCREASED', 'HEPATIC FAILURE', 'JAUNDICE',
                'THYROID CANCER', 'METASTASES TO CENTRAL NERVOUS SYSTEM', 'HEPATIC CANCER', 'HEPATIC CIRRHOSIS',
                'HYPOPHAGIA', 'FAECES DISCOLOURED', 'HAEMATEMESIS', 'ABASIA', 'HEPATIC FUNCTION ABNORMAL',
                'SKIN LESION', 'PAIN OF SKIN', 'DYSPHONIA', 'ACUTE MYELOID LEUKAEMIA']

    SORAFENIB_SUNITINIB = ['STOMATITIS', 'ABASIA', 'MUCOSAL INFLAMMATION', 'ASCITES',
                           'PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME', 'METASTASES TO LUNG', 'BONE MARROW FAILURE',
                           'JAUNDICE', 'BLISTER', 'GINGIVAL BLEEDING', 'ORAL PAIN', 'FEEDING DISORDER', 'SKIN LESION',
                           'SKIN FISSURES', 'METASTASES TO LIVER', 'BLOOD LACTATE DEHYDROGENASE INCREASED']
    SORAFENIB_VEMURAFENIB = ['BLOOD BILIRUBIN INCREASED', 'SKIN EXFOLIATION', 'BLISTER',
                             'METASTASES TO CENTRAL NERVOUS SYSTEM', 'GAMMA-GLUTAMYLTRANSFERASE INCREASED',
                             'MUCOSAL INFLAMMATION', 'SKIN REACTION', 'SKIN LESION',
                             'BLOOD LACTATE DEHYDROGENASE INCREASED', 'PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME',
                             'RASH MACULO-PAPULAR']
    SUNITINIB_VEMURAFENIB = ['FACE OEDEMA', 'APHTHOUS ULCER', 'DISEASE PROGRESSION', 'BLISTER', 'SKIN DISORDER',
                             'MUCOSAL INFLAMMATION', 'DYSGEUSIA', 'SKIN LESION',
                             'BLOOD LACTATE DEHYDROGENASE INCREASED', 'PALMAR-PLANTAR ERYTHRODYSAESTHESIA SYNDROME']
    SORAFENIB = ['METASTASES TO BONE', 'HEPATIC FAILURE', 'HEPATIC CIRRHOSIS', 'HYPOPHAGIA', 'HAEMATEMESIS',
                 'HEPATIC FUNCTION ABNORMAL', 'PAIN OF SKIN', 'DYSPHONIA', 'ENCEPHALOPATHY',
                 'ASPARTATE AMINOTRANSFERASE INCREASED', 'MELAENA', 'BEDRIDDEN', 'ALANINE AMINOTRANSFERASE INCREASED']
    VEMURAFENIB = ['PHOTOSENSITIVITY REACTION', 'MASS', 'UVEITIS',
                   'DRUG REACTION WITH EOSINOPHILIA AND SYSTEMIC SYMPTOMS', 'RASH GENERALISED', 'DECREASED ACTIVITY',
                   'STEVENS-JOHNSON SYNDROME', 'RASH', 'EJECTION FRACTION DECREASED', 'CHOLESTASIS',
                   'BLOOD CREATINE PHOSPHOKINASE INCREASED', 'MALIGNANT MELANOMA', 'MYOCARDITIS', 'FACIAL PARALYSIS',
                   'HEPATOCELLULAR INJURY', 'ARTHRALGIA', 'ALOPECIA', 'MYALGIA', 'BLOOD ALKALINE PHOSPHATASE INCREASED',
                   'DYSPHAGIA', 'RASH ERYTHEMATOUS', 'HEPATOTOXICITY', 'PYREXIA', 'RASH PAPULAR', 'NO ADVERSE EVENT',
                   'DERMATITIS', 'HEPATIC ENZYME INCREASED', 'LYMPHOPENIA', 'OCULAR HYPERAEMIA']
    SUNITINIB = ['AGEUSIA', 'GLOSSODYNIA', 'ORAL DISCOMFORT', 'HAIR COLOUR CHANGES', 'HYPOTHYROIDISM', 'TASTE DISORDER',
                 'THYROID DISORDER', 'CHROMATURIA', 'BLOOD THYROID STIMULATING HORMONE INCREASED',
                 'OSTEONECROSIS OF JAW', 'MOUTH ULCERATION', 'DYSPEPSIA', 'SKIN DISCOLOURATION', 'HYPERTENSION',
                 'BLOOD UREA INCREASED', 'PLATELET COUNT DECREASED', 'FLATULENCE', 'TENDERNESS', 'EPISTAXIS',
                 'DRY MOUTH', 'TEMPERATURE INTOLERANCE', 'PROTEINURIA']

    df = pd.DataFrame()
    source = ['SORAFENIB', 'SUNITINIB', 'SORAFENIB', 'VEMURAFENIB'] + ['B-RAF'] * len(SORAFENIB_VEMURAFENIB) + [
        'OTHER GENES'] * len(
        SORAFENIB_SUNITINIB)  # + ['OTHER GENES'] * len(SUNITINIB) #+ ['OTHER GENES'] * len(both) + ['B-RAF'] * len(onlysora)
    target = ['OTHER GENES', 'OTHER GENES', 'B-RAF',
              'B-RAF'] + SORAFENIB_VEMURAFENIB + SORAFENIB_SUNITINIB  # + both + onlysora
    df['source'] = source
    df['target'] = target
    # df['weight']= [100, 110, 40] + [2] * (len(both) + len(onlysora))
    df['weight'] = [100, 110, 50, 50] + [9] * len(SORAFENIB_VEMURAFENIB) + [13] * len(SORAFENIB_SUNITINIB)
    # Import library
    from d3blocks import D3Blocks

    # Initialize
    d3 = D3Blocks()

    d3.sankey(df, link={"color": "source-target"}, filepath='Sankey_demo_1.html')

    # d3.node_properties['BLISTER']['align'] = 'center'