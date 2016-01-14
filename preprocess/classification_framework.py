# Developing new classification framework
# 1/11/2016

from pybraincompare.compare.mrutils import get_images_df
from pybraincompare.mr.datasets import get_standard_mask
import matplotlib.pyplot as plt
from sklearn import linear_model
from glob import glob
import pickle
import pandas
import nibabel
import re
import os

# Get example group to work with
base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison"
data = "%s/data" %base        # mostly images
likelihood_pickles = glob("%s/likelihood/*.pkl" %(data))
results = "%s/results" %base  # any kind of tsv/result file
web = "%s/web" %base

# Get standard mask
standard_mask=get_standard_mask()

# We will use image meta data
images = pandas.read_csv("%s/contrast_defined_images_filtered.tsv" %results,sep="\t")
collections = pandas.read_csv("%s/collections_with_dois.tsv" %results,sep="\t")

### Step 1: Load meta data sources
unique_contrasts = images.cognitive_contrast_cogatlas_id.unique().tolist()

# Images that do not match the correct identifier will not be used (eg, "Other")
expression = re.compile("cnt_*")

# Load group data to test a concept node
group = likelihood_pickles[15]
group = pickle.load(open(group,"rb"))

concepts = [os.path.basename(x).split("group_")[-1].replace(".pkl","") for x in likelihood_pickles]

# We can go through each voxel (column) in a data frame of image data
mr = get_images_df(file_paths=group["in"] + group["out"],mask=standard_mask)
image_paths = group["in"] + group["out"]
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
image_ids = image_ids_in + image_ids_out
mr.index = image_ids

# This is the model that we want to build for each voxel
# Y (n images X 1) = X (n images X n concepts) * beta (n concepts X 1)
# Y is the value of the voxel (we are predicting)
# X is images by concepts, with values corresponding to concept labels
# beta are the coefficients we get from building the model
X = pandas.DataFrame(0,index=image_ids,columns=concepts)
for group_pkl in likelihood_pickles:
    group = pickle.load(open(group_pkl,"rb"))
    concept_id = os.path.basename(group_pkl).split("group_")[-1].replace(".pkl","")
    print "Parsing concept %s" %(concept_id)
    image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
    image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
    X.loc[image_ids_in,concept_id] = 1 
   
# hold out an image
holdout = mr.index[0]
holdoutY = X.loc[holdout,:]
#holdout_nii = [x for x in image_paths if re.search(str(holdout),x)][0]
#nii_df = get_images_df(holdout_nii,standard_mask)
predicted_image = []
for voxel in mr.columns:
    train = [x for x in X.index if x!=holdout and x in mr.index]
    Y = mr.loc[train,voxel].tolist()
    Xtrain = X.loc[train,:] 
    # Use regularized regression
    clf = linear_model.ElasticNet(alpha=0.1)
    clf.fit(Xtrain,Y)
    predicted_value = clf.predict(holdoutY.tolist())
    predicted_image.append(predicted_value)    

# Save to new nifti image
image = numpy.zeros(standard_mask.shape)
image[standard_mask.get_data()!=0] = predicted_image
nii = nibabel.Nifti1Image(image,affine=standard_mask.get_affine())
nibabel.save(nii,"%s/%s_predicted.nii.gz" %(base,holdout))

# what you can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

# Ok - instead of using predict each time, I should save the regression parameters. duh.
regression_params = pandas.DataFrame(0,index=mr.columns,columns=concepts)

for voxel in mr.columns:
    print "Training voxel %s" %(voxel)
    train = [x for x in X.index if x!=holdout and x in mr.index]
    Y = mr.loc[train,voxel].tolist()
    Xtrain = X.loc[train,:] 
    # Use regularized regression
    clf = linear_model.ElasticNet(alpha=0.1)
    clf.fit(Xtrain,Y)
    regression_params.loc[voxel,:] = clf.coef_.tolist()
    

# EXTRA: Here is trying LOOCV for one concept, using whole brain for model
predictions = pandas.DataFrame(index=image_ids,columns=["lasso_prediction"])
unique_images = group["in"] + group["out"]

for image_path in unique_images:
    # Set "image_id" ids to 1 in indicator
    image_id = int(os.path.basename(image_path).replace(".nii.gz",""))
    "Predicting image %s" %(image_id)
    images_in = [x for x in group["in"] if x != image_path]
    images_out = [x for x in group["out"] if x != image_path]
    image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in images_in]
    image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in images_out]
    mr = get_images_df(file_paths=images_in + images_out,mask=standard_mask)
    mr.index = image_ids_in + image_ids_out
    indicator = pandas.DataFrame(0,columns=["concept"],index=image_ids_in + image_ids_out)
    indicator = indicator.loc[mr.index]
    indicator.loc[image_ids_in,"concept"] = 1
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(mr,indicator["concept"].tolist())
    # Save regression parameters for voxel

    # Load the new image
    query = get_images_df(image_path,mask=standard_mask)
    Z = clf.predict(query[0])
    predictions.loc[image_id,"lasso_prediction"] = Z

actual = pandas.DataFrame(0,columns=["concept"],index=image_ids)
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
actual.loc[image_ids_in,"concept"] = 1
actual = actual.loc[predictions.index]
