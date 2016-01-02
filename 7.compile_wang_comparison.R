# Concatenate scores into one matrix

image_file = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/results/contrast_defined_images_filtered.tsv"

images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)

output_folder = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/wang_scores"
input_files = list.files(output_folder,pattern="*.Rda",full.names=TRUE)

# We will put our results in a data frame
similarities = matrix(nrow=nrow(images),ncol=nrow(images))
rownames(similarities) = images$image_id
colnames(similarities) = images$image_id

for (file in input_files) {
  load(file)
  if (is.na(similarities[result$row1,result$row2]) || (is.na(similarities[result$row2,result$row1]))) { 
  similarities[result$row1,result$row2] = result$score
  similarities[result$row2,result$row1] = result$score
  }
};

# Find missing similarities - the Cognitive Atlas API is not terrible reliable,
# and we will calculate these locally before analysis
missing = which(is.na(similarities),arr.ind=TRUE)
save(missing,file="/home/vsochat/SCRIPT/R/brainmeta/missing.Rda")


output_file = cat(output_folder,"/contrast_defined_images_wang.tsv",sep="")
write.csv(similarities,filename=output_file,sep="\t")
