#simple visualization scripts


import matplotlib.pyplot as plt
from zepid.graphics import EffectMeasurePlot
import plotly.graph_objects as go

def forest_plot(paper_ids, measure, low, high, type='PFS', title='Forest Plot'):
        labs = paper_ids + ['Overall']
        measure = measure + [sum(measure) / len(measure)]
        lower = low + [sum(measure) / len(measure)]
        upper = high + [sum(measure) / len(measure)]
        p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
        p.labels(effectmeasure=type)
        p.colors(pointshape="D")
        p.colors(pointcolor='b')
        ax=p.plot(figsize=(7,3), t_adjuster=0.09, max_value=max(high), min_value=min(low) )
        plt.title(title,loc="right",x=1, y=1.045)
        #plt.suptitle("Missing Data Imputation Method",x=0,y=0.98)
        #ax.set_xlabel("Favours Control      Favours ABEMACICLIB       ", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        plt.show()


def visualize_table(heads, vals):

    headerColor = 'grey'
    rowEvenColor = 'lightblue'
    rowOddColor = 'white'

    fig = go.Figure(data=[go.Table(
      header=dict(
        values=heads,
        line_color='darkslategray',
        fill_color=headerColor,
        align=['left','center'],
        font=dict(color='white', size=12)
      ),
      cells=dict(
        values=vals,
        line_color='darkslategray',
        # 2-D list of colors for alternating rows
        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])

    fig.show()

def standardize_PFS_OS(measures):
    #converts days/weeks/years into months, formats data correctly, and inserts 95% CI
    if measures[2] == '':
        CI = [float(measures[0]), float(measures[0])] #use the exact measurement as confidence interval if not given
    else:
        CI = [float(i) for i in measures[2].split('-')]

    return_val = [float(measures[0])]+CI

    if measures[1] == 'months':
        return return_val

    elif measures[1] == 'days':
        return_val = [round(i/30, 3) for i in return_val]
        return return_val

    elif measures[1] == 'weeks':
        return_val = [round(i*0.23, 3) for i in return_val] #1 week roughly equals 0.23 months
        return return_val

    elif measures[1] == 'years':
        return_val = [round(i * 12, 3) for i in return_val]
        print(return_val)
        return return_val

    else:
        print('Error, no time given.')



def visualize_relation(relation, plot_forest=True):
    dict_list = relation #this will change
    ids = []
    outcome_meshs = []
    population_terms = []
    key_sentences = []
    effects = []
    num_randomized = []
    for paper in dict_list:
        paper_id = paper['PAPER_ID']
        ids.append(paper_id)
        outcome_mesh_terms = [i['mesh_term'] for i in paper['outcome_meshs']]
        outcome_mesh_terms = ', '.join(outcome_mesh_terms)
        outcome_meshs.append(outcome_mesh_terms)
        population = [i['mesh_term'] for i in paper['population_meshs']]
        #population = paper['population']
        population = ', '.join(population)
        population_terms.append(population)
        key_sentence = paper['punchline_text']
        key_sentences.append(key_sentence)
        effect = paper['effect']
        effects.append(effect)
        participant_n = paper['num_randomized']
        num_randomized.append(participant_n)

    visualize_table(heads=['<b>Paper ID</b>','<b>Population</b>','<b>Outcome Mesh</b>','<b>n Participants</b>', '<b>Punchline</b>', '<b>Effect</b>'], vals=[ids, population_terms, outcome_meshs, num_randomized, key_sentences, effects])
    if plot_forest == True:
        types = ['HR', 'PFS', 'OS']
        for type in types:
            measures, lows, highs = [], [], []
            ids = []
            for i in dict_list:
                m = i[type]
                if m != None:
                    if type != 'HR':
                        m = standardize_PFS_OS(m)
                    measures.append(m[0])
                    lows.append(m[1])
                    highs.append(m[2])
                    ids.append(i['PAPER_ID'])

            if len(measures) < 3: #we want at least 3 measurements
                print('Not enough data for',type, 'score')
            else:
                forest_plot(paper_ids=ids, measure=measures, low=lows, high=highs, type=type, title=relation[0])


'''import ast
import csv
import sys
#visualize_relation(x, plot_forest=False)
csv.field_size_limit(sys.maxsize)
with open('data/rels_with_rev.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    rels = list(csv_reader)

d = ast.literal_eval(rels[5][4])
print(d[0]['effect'])
visualize_relation(d, plot_forest=True)'''

import csv
import sys
import ast

csv.field_size_limit(sys.maxsize)
with open('data/rels_with_rev.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    rels = list(csv_reader)

def get_number_of_papers_for_pki(name):
    count = 0
    for i in rels:
        if i[1] == name.upper():
            count = count + len(ast.literal_eval(i[3]))
    return count

#print(get_number_of_papers_for_pki('IMATINIB'))
import json
import pandas as pd
def group_pkis_by_max_phase():
    f = open('data/PKIs')
    pkis = json.load(f)
    maxphases = {0: [], 1: [], 2: [], 3:[], 4:[]}
    for i in pkis:
        try:
            maxphases[i['max_phase']] = maxphases[i['max_phase']] + [i['pref_name']]
        except:
            print('None')

    return maxphases

def plot_papers_to_phase_bardotplot():
    pkiphases = group_pkis_by_max_phase()
    data = []
    for phase in pkiphases:
        for pki in pkiphases[phase]:
            data.append([get_number_of_papers_for_pki(pki), phase])

    df = pd.DataFrame(data, columns = ['Papers', 'Phase'])
    print(df)
    import seaborn as sns, matplotlib.pyplot as plt
    sns.set(style="whitegrid")

    tips = df #sns.load_dataset("tips")
    print(tips)

    print(type(tips))
    g = sns.barplot(x="Phase", y="Papers", data=tips, capsize=.1, ci="sd")
    g = sns.swarmplot(x="Phase", y="Papers", data=tips, color="0", alpha=.35)
    g.set_yscale("log")
    plt.show()