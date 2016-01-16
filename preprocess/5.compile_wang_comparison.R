# Concatenate scores into one matrix

args = commandArgs(TRUE)
base = args[1]

image_file = paste(base,"/results/contrast_defined_images_filtered.tsv",sep="")

images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)

output_folder = paste(base,"/data/","wang_scores",sep="")
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
# and we will calculate these again (see end of 7.run_wang_graph_comparison_sherlock.R) before analysis
missing = which(is.na(similarities),arr.ind=TRUE)
save(missing,file="missing.Rda")


# We will copy this into the "analysis/wang" folder on our local system
output_file = paste(output_folder,"/contrast_defined_images_wang.tsv",sep="")
write.csv(similarities,file=output_file)
