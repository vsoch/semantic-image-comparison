#!/usr/bin/env python2

# Use neurovault, cognitive atlas APIs, and pybraincompare to calculate similarity (complete case analysis with pearson score) for images in NeuroVault tagged with a cognitive atlas contrast (N=93)

import os
import shutil
import nibabel
import pandas
import sys
import numpy
from glob import glob
from pyneurovault import api
from cognitiveatlas.api import get_task, get_concept
from nilearn.image import resample_img
from pybraincompare.compare.maths import TtoZ

## STEP 1: DOWNLOAD OF NV IMAGES ######################################################

# Set up work folders for data
# For the VM: these paths will be environmental variables
base = sys.argv[1]

results = "%s/results" %base  # any kind of tsv/result file
data = "%s/data" %base        # mostly images

if not os.path.exists(base):
    os.mkdirs(base)

if not os.path.exists(results):
    os.mkdir(results)

if not os.path.exists(data):
    os.mkdir(data)

# Get all collections
collections = api.get_collections()

# Filter images to those that have a DOI
collections = collections[collections.DOI.isnull()==False]
# We use this file to create custom web interface
collections.to_csv("%s/collections_with_dois.tsv" %results,encoding="utf-8",sep="\t")

# Get image meta data for collections (N=1023)
images = api.get_images(collection_pks=collections.collection_id.tolist())

# Filter images to those with contrasts defined (N=98)
images = images[images.cognitive_contrast_cogatlas_id.isnull()==False]
# Not needed for future analyses, for documentation only
images.to_csv("%s/contrast_defined_images.tsv" %results,encoding="utf-8",sep="\t")

# Get rid of any not in MNI
images = images[images.not_mni == False]

images = images[images.analysis_level!='single-subject']

# Get rid of thresholded images
images = images[images.is_thresholded == False]

# We can't use Rest or other/none
images = images[images.cognitive_paradigm_cogatlas_id.isnull()==False]
images = images[images.cognitive_paradigm_cogatlas.isin(["None / Other","rest eyes closed","rest eyes open"])==False]

# Limit to Z and T maps (all are Z and T)
z = images[images.map_type == "Z map"]
t = images[images.map_type == "T map"]

# Remove tmaps that do not have # subjects defined
t_collections = collections[collections.collection_id.isin([int(x) for x in t.collection_id])]
to_keep = t_collections.collection_id[t_collections.number_of_subjects.isnull()==False]
t = t[t.collection_id.isin(to_keep)]
images = z.append(t)

# Download images
standard = os.path.abspath("mr/MNI152_T1_2mm_brain.nii.gz")
api.download_images(dest_dir = data,images_df=images,target=standard)

# For T images, convert to Z
tmaps = [ "%s/resampled/%06d.nii.gz" %(data,x) for x in t.image_id.tolist()]

# Look up number of subjects, and calculate dofs
dofs = []
for row in t.iterrows():
    dof = collections.number_of_subjects[collections.collection_id == int(row[1].collection_id)].tolist()[0] -2
    dofs.append(dof)

outfolder_z = "%s/resampled_z" %(data)
if not os.path.exists(outfolder_z):
    os.mkdir(outfolder_z)

for tt in range(0,len(tmaps)):
    tmap = tmaps[tt]
    dof = dofs[tt]
    zmap_new = "%s/%s" %(outfolder_z,os.path.split(tmap)[-1])
    TtoZ(tmap,output_nii=zmap_new,dof=dof)

# Copy all (already) Z maps to the folder
zmaps = [ "%s/resampled/%06d.nii.gz" %(data,x) for x in z.image_id.tolist()]
for zmap in zmaps:
    zmap_new = "%s/%s" %(outfolder_z,os.path.split(zmap)[-1])
    shutil.copyfile(zmap,zmap_new)

if len(glob("%s/*.nii.gz" %(outfolder_z))) != images.shape[0]:
    raise ValueError("ERROR: not all images were found in final folder %s" %(outfolder_z))

# NEEDED for future analyses, 
# moved to https://github.com/vsoch/semantic-image-comparison/tree/master/analysis/wang/data
images.to_csv("%s/contrast_defined_images_filtered.tsv" %results,encoding="utf-8",sep="\t")

## STEP 2: IMAGE SIMILARITY
######################################################

from pybraincompare.compare.mrutils import make_binary_deletion_mask
from pybraincompare.compare.maths import calculate_correlation
""" Usage:
calculate_correlation(images,mask=None,atlas=None,summary=False,corr_type="pearson"):
make_binary_deletion_mask(images)
"""
standard = nibabel.load(standard)

# Function to pad ID with appropriate number of zeros
def pad_zeros(the_id,total_length=6):
    return "%s%s" %((total_length - len(str(the_id))) * "0",the_id)

# Calculate image similarity with pearson correlation
# Feasible to run in serial for small number of images
print "Calculating spatial image similarity with pearson score, complete case analysis (set of overlapping voxels) for pairwise images..."
image_ids = images.image_id.tolist()
simmatrix = pandas.DataFrame(columns=image_ids,index=image_ids)
for id1 in image_ids:
    print "Processing %s..." %id1
    mr1_id = pad_zeros(id1)
    mr1_path = "%s/resampled_z/%s.nii.gz" %(data,mr1_id)
    mr1 = nibabel.load(mr1_path)
    for id2 in image_ids:
        mr2_id = pad_zeros(id2)
        mr2_path = "%s/resampled_z/%s.nii.gz" %(data,mr2_id)
        mr2 = nibabel.load(mr2_path)
        # Make a pairwise deletion / complete case analysis mask
        pdmask = make_binary_deletion_mask([mr1,mr2])
        pdmask = nibabel.Nifti1Image(pdmask,affine=standard.get_affine())
        score = calculate_correlation([mr1,mr2],mask=pdmask)
        simmatrix.loc[id1,id2] = score
        simmatrix.loc[id2,id1] = score

simmatrix.to_csv("%s/contrast_defined_images_pearsonpd_similarity.tsv" %results,sep="\t")

# Finally, resample images to 4mm voxel for classification analysis
outfolder_z4mm = "%s/resampled_z_4mm" %(data)
if not os.path.exists(outfolder_z4mm):
    os.mkdir(outfolder_z4mm)

maps = glob("%s/*.nii.gz" %outfolder_z)
for mr in maps:
    image_name = os.path.basename(mr)
    print "Resampling %s to 4mm..." %(image_name)
    nii = nibabel.load(mr)
    nii_resamp = resample_img(nii,target_affine=numpy.diag([4,4,4]))
    nibabel.save(nii_resamp,"%s/%s" %(outfolder_z4mm,image_name))
