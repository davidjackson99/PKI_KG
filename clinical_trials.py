# identify, extract, and match clinical trials from clinicaltrials.gov for our dataset


from pytrials.client import ClinicalTrials
from tools.scraper import get_NCT_number_from_PMID
import sys
import csv
from os import path
from drug_attributes import get_all_pki_synonyms

ct = ClinicalTrials()

synonyms = get_all_pki_synonyms().values()

#fields = ct.get_full_studies(search_expr="NCT02778685", max_studies=1)

def get_study_results(fields, nctid):
    results = {}
    results['nctid'] = nctid
    measure_results = {}
    try:
        participants = fields['FullStudiesResponse']['FullStudies'][0]['Study']['ProtocolSection']['DesignModule']['EnrollmentInfo']['EnrollmentCount']
    except:
        participants = None
    results['participants'] = participants

    interventions = []
    for i in fields['FullStudiesResponse']['FullStudies'][0]['Study']['ProtocolSection']['ArmsInterventionsModule']['InterventionList']['Intervention']:
        interventions.append(i['InterventionName'])

    results['interventions'] = interventions

    conditions = fields['FullStudiesResponse']['FullStudies'][0]['Study']['ProtocolSection']['ConditionsModule']['ConditionList'][
        'Condition']

    results['conditions'] = conditions

    try:
        results['phase'] = fields['FullStudiesResponse']['FullStudies'][0]['Study']['ProtocolSection']['DesignModule']['PhaseList']['Phase']
    except:
        results['phase'] = None

    for i in fields['FullStudiesResponse']['FullStudies']:
        if 'ResultsSection' in i['Study'] != None:
            try:
                OutcomeSect = i['Study']['ResultsSection']['OutcomeMeasuresModule']['OutcomeMeasureList']['OutcomeMeasure']
                if isinstance(OutcomeSect, dict):
                    OutcomeSect = [OutcomeSect]
                for hierarch in OutcomeSect:
                    if 'progression free survival' in hierarch['OutcomeMeasureTitle'].lower() or 'overall survival' in hierarch['OutcomeMeasureTitle'].lower() or 'progression-free survival' in hierarch['OutcomeMeasureTitle'].lower():
                        munit = hierarch['OutcomeMeasureUnitOfMeasure']
                        groups = [y['OutcomeGroupTitle'] for y in hierarch['OutcomeGroupList']['OutcomeGroup']]

                        for i in hierarch['OutcomeClassList']['OutcomeClass']:
                            for x in i['OutcomeCategoryList']['OutcomeCategory']:
                                measurements = [y['OutcomeMeasurementValue'] for y in x['OutcomeMeasurementList']['OutcomeMeasurement']]
                        measure_results[hierarch['OutcomeMeasureTitle']] = [groups, measurements, munit]

            except:
                continue
    results['outcomes'] = measure_results
    return results

def get_pmid_to_nctidlist():
    import pickle
    if path.exists('data/pmid_to_nctid.pkl'):
        return pickle.load(open("data/pmid_to_nctid.pkl", "rb"))
    else:
        csv.field_size_limit(sys.maxsize)
        with open('data/prev_data/papers_db_updated_copy.csv', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            papers = list(csv_reader)

        paper_ids = [i[-1] for i in papers]
        import pickle
        pm_to_nct = []
        for id in paper_ids:
            nct_id = get_NCT_number_from_PMID(id)
            pm_to_nct.append([id, nct_id])
        with open('data/pmid_to_nctid.pkl', 'wb') as f:
            pickle.dump(pm_to_nct, f)


def get_nctfrompmid(pmid):
    pmctlist = get_pmid_to_nctidlist()
    for i in pmctlist:
        if i[0] == pmid:
            return i[1]

import pickle
def load_all_clinical_trials_to_dict():


    paperids = pickle.load(open("data/pmid_to_nctid.pkl", "rb"))
    clinicaltrials_dict = {}
    for i in paperids:
        pubmedid = i[0]
        nctid = i[1]
        if nctid != None:
            try:
                fields = ct.get_full_studies(search_expr=nctid, max_studies=1)
                nct_results = get_study_results(fields, nctid)
                if pubmedid in clinicaltrials_dict.keys():
                    print(pubmedid)
                clinicaltrials_dict[pubmedid] = nct_results
            except:
                print(pubmedid, nctid)

    with open('data/clinicaltrials_results.pickle', 'wb') as handle:
        pickle.dump(clinicaltrials_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('data/clinicaltrials_results.pickle', 'rb') as handle:
    b = pickle.load(handle)

