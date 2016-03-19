import xmltodict # https://github.com/martinblech/xmltodict
from cognitiveatlas.api import get_concept
from Bio import Entrez
from glob import glob
import pickle
import numpy
import pandas
import urllib2
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


#################################################################################################
# Filter Based on Mesh Terms
#################################################################################################

# @russpoldrack, semantic web ninjoid!
# from http://docs.python-guide.org/en/latest/scenarios/xml/

# Function to get treecodes from a mesh dictionary
def get_tree_codes(doc,treecodes):
    mesh_term_matches=[]
    mesh_tree_matches=[]
    for i in range(len(doc['DescriptorRecordSet']['DescriptorRecord'])):
        descriptor = doc['DescriptorRecordSet']['DescriptorRecord'][i]
        if not 'TreeNumberList' in descriptor.keys():
            continue
        for treenumber in descriptor['TreeNumberList']['TreeNumber']:
            for treecode in treecodes:
                if treenumber.find(treecode)==0:
                    print descriptor['DescriptorName']['String']
                    mesh_term_matches.append(descriptor['DescriptorName']['String'])
                    mesh_tree_matches.append(treecode)
    return mesh_term_matches,mesh_tree_matches


concepts = pandas.read_csv("%s/concepts_metadata_799.tsv" %pubmed_folder,sep="\t",encoding="utf-8",index_col=0)
mesh_file = "ftp://nlmpubs.nlm.nih.gov/online/mesh/.xmlmesh/desc2016.xml"
response = urllib2.urlopen(mesh_file)
mesh = response.read()
doc = xmltodict.parse(mesh)
treecodes=['G11.561','F','H01.158.610.030']
mesh_term_matches,mesh_tree_matches = get_tree_codes(doc,treecodes)

mesh_df = pandas.DataFrame()
mesh_df["term_matches"] = mesh_term_matches
mesh_df["tree_matches"] = mesh_tree_matches
# There are repeats!
mesh_df = mesh_df.drop_duplicates()
# Save result to file
mesh_df.to_csv("%s/cognitive_mesh_1119.tsv" %pubmed_folder,sep="\t")


# Function to filter a pubmed xml by mesh term
def filter_mesh(article_xml,mesh_filter):
    filtered = []
    for article in article_xml:
        if 'MedlineCitation' in article:
            if 'MeshHeadingList' in article['MedlineCitation']:
                found = False
                for mesh in article['MedlineCitation'][u'MeshHeadingList']:
                    if found == True:
                        continue
                    if str(mesh["DescriptorName"]) in mesh_filter:
                        filtered.append(str(article["MedlineCitation"]["PMID"]))
                        found = True
    return filtered


# Function to fetch info on articles using efetch
def search_mesh(email,pmids,mesh_filter,retmax=100000):
    '''filter_mesh returns a list of articles associated with a term and count
    :param email: an email address to associated with the query for Entrez
    :param term: the search term for pubmed, "+" will replace spaces
    :param retmax: maximum number of results to return, default 100K. If more exist, will be obtained.
    '''
    Entrez.email = email
    iters = int(numpy.ceil(len(pmids)/float(retmax)))
    filtered_pmids = []
    for i in range(iters):
        start = i*retmax
        if (start + retmax) < len(pmids):
            end = start + retmax
        else:
            end = len(pmids)
        batch = pmids[start:end]
        handle = Entrez.efetch(db='pubmed', id=batch, retmax=retmax,retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        mesh_pmids = filter_mesh(record,mesh_filter) 
        filtered_pmids = filtered_pmids + mesh_pmids
        time.sleep(0.5)
    print "Found %s out of %s pmids" %(len(filtered_pmids),len(pmids))
    return filtered_pmids

# Function to save output
def save_output_pkl(output,output_file):
    if not os.path.exists(output_file):
        pickle.dump(output,open(output_file,"wb"))

# Prepare list of mesh terms
mesh_list = mesh_df["term_matches"].tolist()

# Save counts
counts = dict()

# We will save each list to a file as we go
for c in range(len(concepts["name"])):
    concept = concepts["name"][c]
    pmids,number_matches = search_articles(email,concept,retmax=100000)    
    if len(pmids) != number_matches:
        print "ERROR parsing term %s, indicated %s matches but list has %s" %(concept,number_matches,len(pmids))
    counts[concept] = number_matches
    output_file = "%s/cogat_%s.pkl" %(pubmed_folder,concept.replace(" ","_"))
    save_output_pkl(pmids,output_file)
    mesh_pmids = search_mesh(email,pmids,mesh_list)
    output_file = "%s/cogat_%s_filtered.pkl" %(pubmed_folder,concept.replace(" ","_"))
    save_output_pkl(mesh_pmids,output_file)

# Finally, save counts
pickle.dump(counts,open("%s/cogat_COUNTS_dict.pkl" %pubmed_folder,"wb"))
