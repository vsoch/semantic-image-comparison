from Bio import Entrez
from cognitiveatlas.api import get_concept
import pickle
import json
import time
import os
import sys

base = sys.argv[1]
pubmed_folder = "%s/pubmed" %base
if not os.path.exists(pubmed_folder):
    os.mkdir(pubmed_folder)

def search_pubmed(term,retstart=0,retmax=100000):
    '''search_pubmed returns a record for a search term using Entrez.esearch
    :param term: the term to search for
    :param retstart: where to start retrieving results, default is 0
    :param retmax: the max number of results to return, default is 100K
    '''
    handle = Entrez.esearch(db='pubmed',term=str(term),retstart=retstart,retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record

def search_articles(email,term,retmax=100000):
    '''search_articles returns a list of articles associated with a term and count
    :param email: an email address to associated with the query for Entrez
    :param term: the search term for pubmed, "+" will replace spaces
    :param retmax: maximum number of results to return, default 100K. If more exist, will be obtained.
    '''
    Entrez.email = email
    # Replace spaces in term with +
    term = term.replace(" ","+")
    record = search_pubmed(term)
    if "IdList" in record:
        number_matches = int(record["Count"])
        allrecords = record["IdList"]
        start = 100000
        # Tell the user this term is going to take longer
        if number_matches >= 100000:
            print "Term %s has %s associated pmids" %(term,number_matches)
        while start <= number_matches:
            record = search_pubmed(term,retstart=start)
            allrecords = allrecords + record["IdList"]
            start = start + 100000
            time.sleep(0.5)
        return allrecords,number_matches
    else:
        return [],0    

# Save a list of just the Cognitive Atlas terms we will search for
concepts = get_concepts().json
concepts = concepts.pandas.drop(["concept_class",
                                 "alias",
                                 "def_event_stamp",
                                 "relationships",
                                 "event_stamp",
                                 "id_concept_class",
                                 "id_user"],axis=1)
concepts.to_csv("%s/concepts_metadata_799.tsv" %pubmed_folder,sep="\t",encoding="utf-8")
email = "vsochat@stanford.edu"

# We will save a dictionary with pmids for each term, and counts
lookup = dict()
counts = dict()

for concept in concepts["name"]:
    terms,number_matches = search_articles(email,concept,retmax=100000)    
    if len(terms) != number_matches:
        print "ERROR parsing term %s, indicated %s matches but list has %s" %(concept,number_matches,len(terms))
    lookup[concept] = terms
    counts[concept] = number_matches
    time.sleep(0.5)

# Save data structure, notes, documentation
note = "Counts were derived from record['Count'] and should be equivalent to len() of each corresponding term list of pmid."
code = 'https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/6.pubmed_cogat.py' #This is so meta!
result = {"count":counts,"pmids":lookup,"code":code,"note":note} 
pickle.dump(result,open("%s/cogat_pmid_dict.pkl" %pubmed_folder,"wb"))
