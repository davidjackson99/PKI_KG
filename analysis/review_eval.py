# evaluate and display the extracted reviews


import csv
import sys
import ast
import pickle
from collections import Counter
import pandas as pd
from itertools import islice
from math import log
import plotly.express as px

csv.field_size_limit(sys.maxsize)


def get_reviews():
    with open('//data/rels_with_rev.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        return list(csv_reader)

reviews = get_reviews()



def term_tree_map():
    ###### TREE MAP #######

    outcomes = []
    outcome_terms = []
    population_meshs = []
    intervention_meshs = []
    for i in reviews:
        papers = ast.literal_eval(i[4])
        for review in papers:
            for term in review['outcome_meshs']:
                outcome_terms.append(term['mesh_term'])
            for term in review['population_meshs']:
                population_meshs.append(term['mesh_term'])
            for term in review['intervention_meshs']:
                intervention_meshs.append(term['mesh_term'])
            outcomes.append(review['effect'])

    def take(n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))


    outcome_terms = list(filter(('Overall').__ne__, outcome_terms))
    population_meshs = list(filter(('secondary').__ne__, population_meshs))

    n = 50
    top_outcomes = take(n, Counter(outcome_terms).items())
    top_pops = take(n, Counter(population_meshs).items())
    top_inter = take(n, Counter(intervention_meshs).items())


    sizes = [i[1] for i in top_outcomes+top_pops+top_inter]
    sizes = [log(i)*log(i) for i in sizes]
    label= [i[0]for i in top_outcomes+top_pops+top_inter]
    values=['outcome']*n+['population']*n+['intervention']*n

    df = pd.DataFrame(
        {'sizes': sizes,
         'terms': label,
         'class': values
        })


    fig = px.treemap(df, path=['class', 'terms'],
                     values='sizes', width=1000, height=1000)

    fig.show()


with open('//data/clinicaltrials_results.pickle', 'rb') as f:
   clintrials = pickle.load(f)


def get_clin_trials_effect(pubmedid, pki):
    pki = pki.lower()
    if pubmedid not in list(clintrials.keys()):
        return None
    #print(pubmedid, pki)
    study = clintrials[pubmedid]
    for x in range(len(study['outcomes'].values())):
        measuretype = list(study['outcomes'].keys())[x]
        unit = list(study['outcomes'].values())[x][-1]
        groups = list(study['outcomes'].values())[x][0]
        values = list(study['outcomes'].values())[x][1]
        #remove NAN values
        nan_indices = [i for i, x in enumerate(values) if x == 'NA']
        groups = [ele for idx, ele in enumerate(groups) if idx not in nan_indices]
        values = [ele for idx, ele in enumerate(values) if idx not in nan_indices]

        if len(groups) != len(values):
            #SAVE AS NO EFFECT
            continue
        indices = [i for i, x in enumerate(groups) if pki in x.lower()]
        non_pki_values = [ele for idx, ele in enumerate(values) if idx not in indices]
        pki_values = [ele for idx, ele in enumerate(values) if idx in indices]
        if pki_values == [] or non_pki_values == []:
            continue

        if 'death' in measuretype.lower() and unit.lower() != 'months' and "Death Event Free" not in measuretype or unit.lower() == 'deaths':
            pki_values = [ast.literal_eval(i) for i in pki_values]
            non_pki_values = [ast.literal_eval(i) for i in non_pki_values]
            effect_diff = max(pki_values) - max(non_pki_values)
            if effect_diff < 0:
                return 1
            elif effect_diff > 0:
                return -1
            else:
                return 0

        else:
            pki_values = [ast.literal_eval(i) for i in pki_values]
            non_pki_values = [ast.literal_eval(i) for i in non_pki_values]
            effect_diff = max(pki_values) - max(non_pki_values)
            if effect_diff > 0:
                return 1
            elif effect_diff < 0:
                return -1
            else:
                return 0

def get_efffects():
    needed_reviews = []
    for x in reviews:
        if x[2] == 'C0007131' and x[1] == 'SORAFENIB':#, 'OSIMERTINIB', 'OSIMERTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'CRIZOTINIB', 'AFATINIB', 'AFATINIB', 'AFATINIB', 'ALECTINIB', 'ALECTINIB', 'ALECTINIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'VANDETANIB', 'CERITINIB', 'CERITINIB', 'CERITINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'ANLOTINIB', 'BRIGATINIB', 'BRIGATINIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB', 'SORAFENIB']:
            needed_reviews.append(x)

    effx = []
    for x in needed_reviews:
        real_reviews = ast.literal_eval(x[4])
        for i in range(len(real_reviews)):
            #print(real_reviews[i]['effect'], real_reviews[i]['punchline_text'], real_reviews[i]['HR'])
            clin = get_clin_trials_effect(ast.literal_eval(x[3])[i], x[1])
            if clin != None:
                effx.append(clin)
            elif real_reviews[i]['HR'] != None:
                if real_reviews[i]['HR'][0] < 1:
                    #print('POSITIVE', real_reviews[i]['HR'])
                    effx.append(1)
                    continue
                elif real_reviews[i]['HR'][0] > 1:
                    #print('NEGATIVE',  real_reviews[i]['HR'])
                    effx.append(-1)
                    continue
            else:

                #'well-tolerated' 'promising antitumor activity'

                if any(substring in real_reviews[i]['punchline_text'] for substring in ['no diff','no impact', 'no significant difference']) and real_reviews[i]['effect'] == '— no diff':
                    effx.append(0)
                elif real_reviews[i]['effect'] == '↓ sig. decrease' and any(substring in real_reviews[i]['punchline_text'] for substring in ['tumor size', 'tumor reduction', 'tumor shrinkage', 'tumor pain intensity', 'tumor blood volume']):
                    effx.append(1)
                elif real_reviews[i]['effect'] == '↑ sig. increase' and any(substring in real_reviews[i]['punchline_text'] for substring in ['antitumor', 'anti-tumor']):
                    effx.append(1)
                elif 'increase' in real_reviews[i]['effect'] and any(substring in real_reviews[i]['punchline_text'] for substring in [' OS', ' PFS', 'progression-free survival', 'overall survival']) and x[1].lower() in real_reviews[i]['punchline_text']:
                    effx.append(1)
                elif any(substring in real_reviews[i]['punchline_text'].lower() for substring in [' pfs','progression-free survival']) and real_reviews[i]['PFS'] != None:
                    effx.append('PFS')
                elif any(substring in real_reviews[i]['punchline_text'].lower() for substring in [' os ','overall survival']) and real_reviews[i]['OS'] != None:
                    effx.append('OS')
                elif any(substring in real_reviews[i]['punchline_text'] for substring in ['improved', 'improvement']):
                    effx.append(1)
                elif x[1].lower() in real_reviews[i]['punchline_text']:
                    print(real_reviews[i]['punchline_text'], real_reviews[i]['effect'])
                    effx.append('EE')
                elif any(substring in real_reviews[i]['punchline_text'] for substring in ['adverse event','related aes', 'toxicities', 'toxicity']):
                    effx.append('AE')
                else:
                    #print(real_reviews[i]['punchline_text'])
                    effx.append(None)

    return effx



effx = get_efffects()
print(Counter(effx))
print(effx.count(1)*1+effx.count(-1)*-1+effx.count(0)*0.1)
print((effx.count(1)*1+effx.count(-1)*-1+effx.count(0)*0.1)/(effx.count(1)+effx.count(-1)+effx.count(0)))
#no additional benefit
#did not affect
