# extract and structure the needed data from pubmed and PMC articles


import xml.etree.ElementTree as ET
import requests
import re

def get_PM_info(string, ids):
    #still some issues here with the abstracts and titles
    #convert string to xml
    tree = ET.ElementTree(ET.fromstring(string))
    root = tree.getroot()
    titles = []
    years = []
    abst = []
    for title in root.iter('ArticleTitle'):
        titles.append(title.text)
    for abstract in root.iter('Abstract'):
        for element in list(abstract.iter()):
            if element.tag == 'Abstract':
                abst = [''.join(element.itertext())]
            '''print(element.tag)
            print(''.join(element.itertext()))
            if element.tag == 'AbstractText':
                abstracts.append(''.join(element.itertext()))'''
    for year in root.iter('DateRevised'):
        for x in year:
            if x.tag == 'Year':
                years.append(int(x.text))

    ptype = []
    for types in root.iter('PublicationTypeList'):
        typelist = []
        for type in types.findall('PublicationType'):
            typelist.append(type.text)
        ptype.append(typelist)

    return list(zip(*[titles, abst, years, ptype, ids]))


def get_PMC_info(string):
    #Article type missing, unsure where in xml file it is.
    #Also a lot of articles have no full text in the xml file which I don't understand yet.
    string = string.encode("ascii", errors="ignore").decode()
    tree = ET.ElementTree(ET.fromstring(string))
    root = tree.getroot()
    articles = []
    for article in root.iter('article'):
        title = ''
        abstract = []
        body = ''
        year = None
        found=False
        for element in list(article.iter()):
            if element.tag == 'article-id' and found == False:
                found = True #first article id given in file is the proper id
                art_id = ''.join(element.itertext())
            if element.tag == 'title-group':
                title = ''.join(element.itertext())
            elif element.tag == 'abstract':
                abstract.append(''.join(element.itertext()))
            elif element.tag == 'body':
                body = ''.join(element.itertext())
            elif element.tag == 'pub-date' and element.attrib == {'pub-type': 'epub'}: #several dates are given we're interested in the epub date
                year = list(element.iter())[-1].text #no exact date needed, just year
        if abstract != []: #some articles have no abstracts
            abstract = abstract[0] #some have additional information but this is not needed
        else: #we need the abstracts
            abstract = ['NO ABSTRACT FOUND']

        articles.append([title, abstract, body, year, art_id])

    return articles


def get_citations_for_ID(ID):

    #will probably have to insert error statement for 0 citation articles
    url = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pmc_refs&id="+ID)
    txt = url.text
    Ids = re.findall(rf'<Id>(.*?)</Id>', txt)
    return Ids

