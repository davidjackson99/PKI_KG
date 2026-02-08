# Extract articles from PubMed and PMC

#Taken in part from Jason Salazar's code and adjusted

import gzip
import requests
from io import BytesIO
from datetime import date
import logging
import xml.etree.ElementTree as ET


CONST_EUTILS_MAX_ARTICLES = 10000
CONST_EUTILS_DEFAULT_MINDATE = "1900"
CONST_EUTILS_DEFAULT_MAXDATE = date.today().strftime("%Y/%m/%d")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

EFETCH_UTILITY = "efetch.fcgi"
ESEARCH_UTILITY = "esearch.fcgi"

log = logging.getLogger(__name__)


def extractArticlesData(database, ids):

    ARGS = {"db": database, "retmode": "xml", "id": ids}

    try:
        response = requests.get(f"{BASE_URL}/{EFETCH_UTILITY}", params=ARGS)
    except Exception as e:
        print('Bro')
        log.error(f"ERROR: API call failed: {e}")
        try:
            response = requests.get(f"{BASE_URL}/{EFETCH_UTILITY}", params=ARGS)
        except Exception as e:
            print('Brooo')
            log.error(f"ERROR: API call failed: {e}")
            try:
                response = requests.get(f"{BASE_URL}/{EFETCH_UTILITY}", params=ARGS)
            except Exception as e:
                log.error(f"ERROR: API call failed: {e}")

    xml_file = response.text

    return xml_file


def getBaselineFileData(year, fileNumber):

    try:
        response = requests.get(
            f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed{year}n{fileNumber}.xml.gz"
        )
    except Exception as e:
        print("ERROR: API call failed: ", e)

    xmlFile = gzip.open(BytesIO(response.content)).read()

    return xmlFile


def getLatestArticleIds(database, lastNDays=7, searchModified=False):

    ARGS = {
        "db": database,
        "retmode": "xml",
        "retmax": str(CONST_EUTILS_MAX_ARTICLES),
        "reldate": str(lastNDays),
    }

    ARGS["datetype"] = "mdat" if searchModified else "edat"

    try:
        response = requests.get(f"{BASE_URL}/{ESEARCH_UTILITY}", params=ARGS)
    except Exception as e:
        print("ERROR: API call failed: ", e)

    ids = getIdsFromXML(response.text)

    return ids


def getArticleIdsForKeyword(database, searchTerm, *args, searchModified=False):
    if args:
        mindate = args[0]
        maxdate = args[1]
    else:
        mindate = CONST_EUTILS_DEFAULT_MINDATE
        maxdate = CONST_EUTILS_DEFAULT_MAXDATE

    ARGS = {
        "db": database,
        "term": searchTerm,
        "mindate": mindate,
        "maxdate": maxdate,
        "retmax": str(CONST_EUTILS_MAX_ARTICLES),
        "retmode": "xml",
    }
    ARGS["datetype"] = "mdat" if searchModified else "edat"

    try:
        response = requests.get(f"{BASE_URL}/{ESEARCH_UTILITY}", params=ARGS)

    except Exception as e:
        log.error(f"ERROR: API call failed: {e}")
        try:
            response = requests.get(f"{BASE_URL}/{ESEARCH_UTILITY}", params=ARGS)

        except Exception as e:
            log.error(f"ERROR: API call failed: {e}")
            try:
                response = requests.get(f"{BASE_URL}/{ESEARCH_UTILITY}", params=ARGS)

            except Exception as e:
                log.error(f"ERROR: API call failed: {e}")
    txt = response.text

    ids = getIdsFromXML(response.text)

    return ids


def getIdsFromXML(xmlObject):
    tree = ET.fromstring(xmlObject)
    x_ids = tree.findall(".//Id")
    ids = [i.text for i in x_ids]
    found_articles = tree.find(".//Count").text

    #print(f"Found {found_articles} articles.\n")

    if int(found_articles) > CONST_EUTILS_MAX_ARTICLES:
        print(
            f"Payload to huge couldn't fetch all articles (max. {CONST_EUTILS_MAX_ARTICLES})"
        )

    return ids

###########


def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

def getArticleFrequencyForKeyword(database, searchTerms, *args, searchModified=False):
    if args:
        mindate = args[0]
        maxdate = args[1]
    else:
        mindate = CONST_EUTILS_DEFAULT_MINDATE
        maxdate = CONST_EUTILS_DEFAULT_MAXDATE

    frequencies = []
    from tqdm import tqdm
    problem = []

    for searchTerm in tqdm(searchTerms):
        ARGS = {
            "db": database,
            "term": searchTerm,
            "mindate": mindate,
            "maxdate": maxdate,
            "retmax": str(CONST_EUTILS_MAX_ARTICLES),
            "retmode": "xml",
        }
        ARGS["datetype"] = "mdat" if searchModified else "edat"

        try:
            response = requests.get(f"{BASE_URL}/{ESEARCH_UTILITY}", params=ARGS)

        except Exception as e:
            frequencies.append(0)
            log.error(f"ERROR: API call failed: {e}")
            continue


        try:
            tree = ET.fromstring(response.text)
            found_articles = int(tree.find(".//Count").text)
            frequencies.append(found_articles)
            if found_articles > 40:
                print(searchTerm, ': ', found_articles)
        except:
            frequencies.append(0)
            print('oops')

    new_dict = dict(zip(searchTerms, frequencies))
    return dict(sorted(new_dict.items(), key=lambda item: item[1]))


def get_NCT_number_from_PMID(PMID):
    string = extractArticlesData('PubMed', PMID)
    string = string.encode("ascii", errors="ignore").decode()
    try:
        tree = ET.ElementTree(ET.fromstring(string))
    except:
        print("AEFFEWAEFAEFAEFAFE", PMID)
    for element in list(tree.iter()):
        if element.tag == 'AccessionNumber':
            nct_number = ''.join(element.itertext())
            if nct_number.startswith('NCT'):
                return nct_number
    return None

#print(get_NCT_number_from_PMID('28273028'))