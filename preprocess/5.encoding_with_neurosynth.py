# Encoding to NeuroSynth

#!/usr/bin/python
from pybraincompare.compare.mrutils import get_images_df
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

# Get all cognitive atlas concepts
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
        for concept in featuresraw.columns:
            processed_concept = " ".join(processText(str(concept)))
            featuresraw.loc[pmid,concept] = len(re.findall(processed_concept,text))
        print "Found %s total occurrences for %s" %(featuresraw.loc[pmid].sum(),pmid)
        count +=1
    else:
        featuresraw.loc[pmid] = numpy.zeros(len(concepts))

#PMID 9728909
#17
# Is NAN in the dataframe, does not have an abstract

featuresraw.to_csv("%s/concepts_799_nostem_neurosynth_counts.tsv" %decode_folder,sep="\t")
#featuresraw = pandas.read_csv("%s/concepts_800_nostem_neurosynth_counts.tsv" %decode_folder,sep="\t",index_col=0)

# Look at overall counts
counts = featuresraw.sum().copy()
counts.sort_values(inplace=True,ascending=False)
counts.to_csv("%s/concepts_800_nostem_counts.tsv" %decode_folder,sep="\t")

# We only want to keep those with counts >=50
nonzeros = counts[counts>0].index.tolist()

features = featuresraw[nonzeros]

# We want to drop activation
features_file = "%s/concepts_614_cognitive_atlas.tsv" %decode_folder
features.to_csv(features_file,sep="\t")

from neurosynth import Dataset
from neurosynth.analysis import meta
from neurosynth.analysis import decode

neurosynth_data = "%s/neurosynth-data" %base

# you will need to open the features file in vim and add pmid as the first column name to appear before "action." 
dataset = Dataset('%s/database.txt' %neurosynth_data,features_file)

# Generate maps for the features
neurosynth_feature_maps = "%s/feature_maps" %neurosynth_data
if not os.path.exists(neurosynth_feature_maps):
    os.mkdir(neurosynth_feature_maps)

meta.analyze_features(dataset, output_dir=neurosynth_feature_maps, prefix='cog_atlas')

#...and boom, you should have a full set of images. (-Tal Yarkoni) :)

# we need to put image data into 2mm mask to resize to 4mm
df = pandas.DataFrame(dataset.image_table.data.toarray())
df.columns = features.index

# Make 4mm images
brain_4mm = get_standard_mask(4)
for pmid in df.columns:
    pmid_mr = df[pmid].tolist()
    empty_nii = numpy.zeros(dataset.masker.volume.shape)
    empty_nii[dataset.masker.volume.get_data()!=0] = pmid_mr
    empty_nii = nibabel.Nifti1Image(empty_nii,affine=dataset.masker.volume.get_affine())
    tmpnii = "%s/tmp.nii.gz" %(neurosynth_feature_maps)
    nibabel.save(empty_nii,tmpnii)
    # ***Interpolation must be nearest as neurosynth data is binary!
    nii = resample_img(tmpnii,target_affine=brain_4mm.get_affine(),interpolation="nearest")
    nibabel.save(nii,"%s/%s.nii.gz" %(neurosynth_feature_maps,pmid))

# Load into image data frame
os.remove("%s/tmp.nii.gz"%(neurosynth_feature_maps))
concept_maps_4mm = glob("%s/*.nii.gz"%(neurosynth_feature_maps))
X = get_images_df(file_paths=concept_maps_4mm,mask=brain_4mm)

Xindex = [int(x.replace(".nii.gz","").replace(neurosynth_feature_maps,"").replace("/","")) for x in concept_maps_4mm]
X.index = Xindex

### ENCODING MODEL
## This is our "features" data frame
# X=load_neurosynth_term_mappings() # size nterms X npapers -  for each paper, a binary encoding of the presence/absence of each cog atlas term in the abstract
# mapping=numpy.zeros(nvoxels,nterms)

# neurosynth_map=load_data() # data from all voxels, size novels X npapers, binary encoding of activation presence/absence

# Get rid of entry with all zeros (PMID does not have abstract)
features=features.drop(9728909,axis=0)
X=X.drop(9728909,axis=0)

# Save data structures so we can run en masse
pickle.dump(X,open("%s/neurosynth_X.pkl" %decode_folder,"wb"))
pickle.dump(features,open("%s/neurosynth_Y.pkl" %decode_folder,"wb"))

mapping = pandas.DataFrame(0,index=X.columns,columns=features.columns)

from sklearn import linear_model
# for v in all_voxels:
# y=neurosynth_map[v,:] # i.e. activation presence/absence in voxel v for all papers
# mapping[v,:]=regularized_logistic_regression(y,X)

# First build model with all data, to look at images
for voxel in range(X.shape[1]):
    print "Parsing voxel %s of %s" %(voxel,X.shape[1])
    # only build model with coefficients if there are nonzero voxels
    if len(X.loc[features.index,voxel].unique())>1:
        y = X.loc[features.index,voxel].tolist()
        # Use regularized logistic regression
        clf = linear_model.LogisticRegression()
        #clf = linear_model.SGDClassifier(loss="log",penalty="elasticnet")
        clf.fit(features,y)
        mapping.loc[voxel,:] = clf.coef_[0]

pickle.dump(mapping,open("%s/neurosynth_regression_params_df.pkl" %decode_folder,"wb"))

# Generate maps for the regparams
regparam_maps = "%s/regparam_maps" %neurosynth_data
if not os.path.exists(regparam_maps):
    os.mkdir(regparam_maps)

for feat in mapping.columns:
    feat_name = feat.replace(" ","_")
    brain_map = mapping[feat].tolist()
    empty_nii = numpy.zeros(brain_4mm.shape)
    empty_nii[brain_4mm.get_data()!=0] = brain_map
    empty_nii = nibabel.Nifti1Image(empty_nii,affine=brain_4mm.get_affine())
    nibabel.save(empty_nii,"%s/%s_regparam.nii.gz" %(regparam_maps,feat_name))


###################################################################################
# COMPARE REGRESSION PARAM MAPS FROM NEUROSYNTH WITH NEUROVAULT MAPS
###################################################################################

# @russpold it would be nice to have an analysis that compares the basis images from the encoding model for the terms that overlap between the neurovault and neurosynth analyses.  

neurosynth_maps = glob("%s/decode/regparam_maps/*.nii.gz" %base)
len(neurosynth_maps)
# 399

neurovault_maps = glob("%s/results/classification_final/*regparam_z.nii.gz" %base)
len(neurovault_maps)
#132

from pybraincompare.compare.mrutils import make_binary_deletion_mask, apply_threshold
from pybraincompare.compare.maths import calculate_correlation
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.compare.mrutils import get_images_df
from cognitiveatlas.api import get_concept

standard_mask = get_standard_mask(4)
decoding_comparison = pandas.DataFrame(columns=["concept_id","concept_name","pearson_brainmask",
                                                "pearson_cca","nvoxels_cca","pearson_cca_thresh1",
                                                "nvoxels_cca_thresh1"])

def make_brainmap(vector,standard_mask):
    zeros = numpy.zeros(standard_mask.shape)
    zeros[standard_mask.get_data()!=0] = vector
    return nibabel.Nifti1Image(zeros,affine=standard_mask.get_affine())
   

count=0
for neurosynth_map in neurosynth_maps:
    concept_name = os.path.basename(neurosynth_map).replace("_regparams.nii.gz","")
    concept = get_concept(name=concept_name).json[0]
    neurovault_map = "%s/results/classification_final/%s_regparam_z.nii.gz" %(base,concept["id"])
    if neurovault_map in neurovault_maps:
        print "Found match for %s" %(concept_name)
        nsmap = nibabel.load(neurosynth_map)
        nvmap = nibabel.load(neurovault_map)
        score = calculate_correlation([nsmap,nvmap],mask=standard_mask)
        # Let's also calculate just for overlapping voxels
        cca_mask = make_binary_deletion_mask([nsmap,nvmap])
        nvoxels = len(cca_mask[cca_mask!=0])
        cca_mask = nibabel.Nifti1Image(cca_mask,affine=standard_mask.get_affine())
        cca_score = calculate_correlation([nsmap,nvmap],mask=cca_mask)        
        # And finally, since we see consistent size of cca mask (meaning not a lot of zeros) let's
        # try thresholding at +/- 1. The nsmap needs to be converted to z score
        image_df = get_images_df([nsmap,nvmap],mask=standard_mask)
        image_df.loc[0] = (image_df.loc[0] - image_df.loc[0].mean()) / image_df.loc[0].std()
        nsmap_thresh = make_brainmap(image_df.loc[0],standard_mask)
        nsmap_thresh = apply_threshold(nsmap_thresh,1.0)
        nvmap_thresh = make_brainmap(image_df.loc[1],standard_mask)
        nvmap_thresh = apply_threshold(nvmap_thresh,1.0)
        cca_mask_thresh = make_binary_deletion_mask([nsmap_thresh,nvmap_thresh])
        nvoxels_thresh = len(cca_mask_thresh[cca_mask_thresh!=0])
        cca_mask_thresh = nibabel.Nifti1Image(cca_mask_thresh,affine=standard_mask.get_affine())                
        cca_score_thresh = calculate_correlation([nsmap_thresh,nvmap_thresh],mask=cca_mask_thresh)        
        decoding_comparison.loc[count] = [concept["id"],concept_name,score,cca_score,nvoxels,cca_score_thresh,nvoxels_thresh]
        count+=1

# Sort by value
decoding_comparison = decoding_comparison.sort_values(by=["pearson_cca_thresh"],ascending=False)
decoding_comparison.to_csv("%s/decode/neurovault_vs_neurosynth_concept_decoding_comparison_df.tsv" %base,sep="\t")
