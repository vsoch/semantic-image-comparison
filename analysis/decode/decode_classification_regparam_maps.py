#!/usr/bin/python

from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.mr.transformation import *
from cognitiveatlas.api import get_concept
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import *
from glob import glob
import nltk.data
import pickle
import nltk
import pandas
import nibabel
import sys
import os

base = sys.argv[1]
data = "%s/data" %base
node_folder = "%s/groups" %data
results = "%s/results" %base  # any kind of tsv/result file
decode_folder = "%s/decode" %base

if not os.path.exists(decode_folder):
    os.mkdir(decode_folder)

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
images = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
output_folder = "%s/classification_final" %results

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# Load the regression params data frame
result = pickle.load(open("%s/regression_params_dfs.pkl" %output_folder,"rb"))

all_concepts = get_concept().json
concepts = dict()
for concept in all_concepts:
    concepts[concept["id"]] = str(concept["name"])

# You will need to copy abstracts.txt into this folder from the repo
abstracts = pandas.read_csv("%s/abstracts.txt" %decode_folder,sep="\t",index_col=0,header=None)
abstracts.columns = ["text"]

# Functions to parse text
def remove_nonenglish_chars(text):
    return re.sub("[^a-zA-Z]", " ", text)
    
def text2sentences(text,remove_non_english_chars=True):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    
    if remove_non_english_chars:
        text = remove_nonenglish_chars(text)
    for s in tokenizer.tokenize(text):
        yield s

def processText(text):
    '''combines text2sentences and sentence2words'''
    vector = []
    for line in text2sentences(text):            
        words = sentence2words(line)
        vector = vector + words
    return vector

def sentence2words(sentence,remove_stop_words=True,lower=True):
    if isinstance(sentence,list): sentence = sentence[0]
    re_white_space = re.compile("\s+")
    stop_words = set(stopwords.words("english"))
    if lower: sentence = sentence.lower()
    words = re_white_space.split(sentence.strip())
    if remove_stop_words:
        words = [w for w in words if w not in stop_words]
    return words

def do_stem(words,return_unique=False,remove_non_english_words=True):
    '''do_stem
    Parameters
    ==========    
    words: str/list
        one or more words to be stemmed
    return_unique: boolean (default True)
        return unique terms
    '''
    stemmer = PorterStemmer()
    if isinstance(words,str):
        words = [words]
    stems = []
    for word in words:
        if remove_non_english_words:
            word = re.sub("[^a-zA-Z]", " ", word)
        stems.append(stemmer.stem(word))
    if return_unique:
        return numpy.unique([s.lower() for s in stems]).tolist()
    else:
        return stems


# Prepare feature data frame -
featuresraw = pandas.DataFrame(columns=concepts.values())

count = 1
for row in abstracts.iterrows():
    pmid = row[0]
    text = row[1].text
    if str(text) != "nan":
        words = processText(text)
        text = " ".join(words)
        print "Parsing pmid %s, %s of %s" %(pmid,count,len(abstracts))
        # search for each cognitive atlas term, take a count
        for concept in features.columns:
            processed_concept = " ".join(processText(str(concept)))
            featuresraw.loc[pmid,concept] = len(re.findall(processed_concept,text))
        print "Found %s total occurrences for %s" %(features.loc[pmid].sum(),pmid)
        count +=1

featuresraw.to_csv("%s/concepts_800_nostem_neurosynth_counts.tsv" %decode_folder,sep="\t")


# Look at overall counts
counts = featuresraw.sum().copy()
counts.sort_values(inplace=True,ascending=False)
counts.to_csv("%s/concepts_800_nostem_counts.tsv" %decode_folder,sep="\t")

# We only want to keep those with counts >=50
nonzeros = counts[counts>=50].index.tolist()

# Each subsequent column is a feature, with the value in the header being used as the feature name. The weights can be whatever we want; in practice, it's not going to matter much because neurosynth (Tal) uses a binary threshold for inclusion. 
normalized_features = pandas.DataFrame(columns=nonzeros)

counter = 1
for row in abstracts.iterrows():
    print "Parsing row %s of %s" %(counter,abstracts.shape[0])
    pmid = row[0]
    if pmid in features.index:
        count = row[1]["count"]
        # We will take the normalized term frequency (i.e., number of occurrences in abstract, divided by number of words in abstract).
        normalized_features.loc[pmid,nonzeros] = featuresraw.loc[pmid,nonzeros] / count
        # Then pick a cut-off like 0.001 or something (basically it amounts to minimum one occurrence, so it's not even clear the normalization matters).
        normalized_features.loc[pmid,nonzeros][normalized_features.loc[pmid,nonzeros] < 0.001] = 0    
        counter +=1

# We want to drop activation
normalized_features = normalized_features.drop(["activation"],axis=1)
normalized_features_file = "%s/concepts_211_cognitive_atlas_normalized.tsv" %decode_folder
normalized_features.to_csv(normalized_features_file,sep="\t")

# Now we want to save as a features.txt file to do decoding 

# DECODING - METHOD 1 uses the neurosynth python API

print "Starting neurosynth decoding..."

from neurosynth import Dataset
from neurosynth.analysis import meta
from neurosynth.analysis import decode

neurosynth_data = "%s/neurosynth-data" %base

# you will need to open the features file in vim and add pmid as the first column name to appear before "action." 
dataset = Dataset('%s/database.txt' %neurosynth_data, normalized_features_file)

# Create decoder to decode our images

from nilearn.image import resample_img
decoder = decode.Decoder(dataset) # select all 211 features from our set
pickle.dump(decoder,open("%s/decoder.pkl" %output_folder,"wb"))
concept_maps = glob("%s/*.nii.gz" %output_folder)

# Generate maps for the features
neurosynth_feature_maps = "%s/feature_maps" %neurosynth_data
if not os.path.exists(neurosynth_feature_maps):
    os.mkdir(neurosynth_feature_maps)

meta.analyze_features(dataset, output_dir=neurosynth_feature_maps, prefix='cog_atlas')

#...and boom, you should have a full set of images. (-Tal Yarkoni) :)

# Decoder needs 2mm images
brain_2mm = get_standard_mask(2)
concept_maps_2mm = []
for concept_map in concept_maps:
    twomm = concept_map.replace("regparam_z","regparam_z_2mm")
    nii = resample_img(concept_map,target_affine=brain_2mm.get_affine())
    nibabel.save(nii,twomm)
    concept_maps_2mm.append(twomm)

decode_result_file = "%s/concept_regparam_decoding.csv" %results
decode_result = decoder.decode(concept_maps_2mm, save=decode_result_file)
df = pandas.DataFrame(decode_result)
df.columns = [x.replace("%s/classification_final/" %results,"").replace("_regparam_z_2mm.nii.gz","") for x in df.columns.tolist()]

# Look up concept names
concept_names = []
for concept_id in df.columns:
    concept = get_concept(id=concept_id).json
    concept_names.append(concept[0]["name"])

df = df.transpose()
df.index = concept_names
df.to_csv(decode_result_file)

# Finally, just look at top ten scores per concept
top_tens = pandas.DataFrame(index=df.index,columns=range(0,10))
for concept_id in df.index.tolist():
    top_ten = df.loc[concept_id,:]
    top_ten.sort_values(ascending=False,inplace=True)
    top_tens.loc[concept_id,:] = top_ten.index[0:10]

top_tens.to_csv("%s/concept_regparam_decoding_named_topten.tsv" %results,sep="\t")

# Look at top tens (absolute value)
top_tens_abs = pandas.DataFrame(index=df.index,columns=range(0,10))
for concept_id in df.index.tolist():
    top_ten = df.loc[concept_id,:]
    top_ten = top_ten.abs()
    top_ten.sort_values(ascending=False,inplace=True)
    top_tens_abs.loc[concept_id,:] = top_ten.index[0:10]

top_tens_abs.to_csv("%s/concept_regparam_decoding_named_topten_abs.tsv" %results,sep="\t")


# METHOD 2 uses the neurosynth web/REST API - we cannot use our own features
# URL-based decoding (possible since all images are in NeuroVault) - much faster!
# Note this was not used for the analysis, as the current neurosynth database doesn't
# include cognitive atlas terms. When it does, this method will be more useful

from pyneurovault.api import get_images
from cognitiveatlas.api import get_concept
from urllib2 import Request, urlopen, HTTPError
import pandas
import json

def get_url(url):
    request = Request(url)
    response = urlopen(request)
    return response.read()

# First, just get all the terms
ns = json.loads(get_url("http://neurosynth.org/decode/data/?neurovault=308"))
terms = []
for term in  ns["data"]:
    terms.append(term["analysis"])

# In case we want it, decode each of our original images
unique_images = images.image_id.unique().tolist()
decode = pandas.DataFrame(index=unique_images,columns=terms)

for unique_image in unique_images:
    print "Decoding original image %s" %(unique_image)
    ns = json.loads(get_url("http://neurosynth.org/decode/data/?neurovault=%s" %unique_image))
    for term in  ns["data"]:
        decode.loc[unique_image,term["analysis"]] = term["r"]

decode_result_file = "%s/original_images_decoding.tsv" %results
decode.to_csv(decode_result_file,sep="\t")

# Now we will decode our collection of images!
rp_images = get_images(collection_pks=[neurovault_collection])
unique_rp_images = rp_images.image_id.unique().tolist()
rp_decode = pandas.DataFrame(index=unique_rp_images,columns=terms)

for unique_image in unique_rp_images:
    print "Decoding regression parameter image %s" %(unique_image)
    ns = json.loads(get_url("http://neurosynth.org/decode/data/?neurovault=%s" %unique_image))
    for term in ns["data"]:
        rp_decode.loc[unique_image,term["analysis"]] = term["r"]

# Now let's add some cognitive atlas meta data, so we don't have to look up later
image_urls = rp_images.file[rp_images.image_id.isin(unique_rp_images)]
rp_decode["0image_urls"] = image_urls.tolist()
concept_ids = [str(x.split("/")[-1].replace(".nii.gz","").replace("_regparam_z","")) for x in image_urls]
rp_decode["0cognitive_atlas_concept_id"] = concept_ids

concept_names = []
for concept_id in concept_ids:
    concept = get_concept(id=concept_id).json
    concept_names.append(concept[0]["name"])

rp_decode["0cognitive_atlas_concept_name"] = concept_names

decode_rp_file = "%s/concept_regparam_decoding.tsv" %results
rp_decode.to_csv(decode_rp_file,sep="\t")
