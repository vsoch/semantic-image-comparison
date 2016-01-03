options(java.parameters = "-Xmx4g") # This ensures we don't run out of memory
library(CogatSimilar) # https://github.com/CognitiveAtlas/cogat-similaR

# Read in table with images
image_file = "/home/vanessa/Documents/Work/BRAINMETA/reverse_inference/contrast_defined_images.tsv"
images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)

# We will put our results in a data frame
similarities = matrix(nrow=nrow(images),ncol=nrow(images))
rownames(similarities) = images$image_id
colnames(similarities) = images$image_id

# For each image, calculate a score!
for (i in 1:nrow(images)){
  cat("Processing",i,"of",nrow(images),"\n")
  mr1 = images[i,]
  CAID1 = mr1$cognitive_contrast_cogatlas_id
  for (j in 1:nrow(images)){
    if (i<=j){
      if (is.na(similarities[i,j])) { # ensures we can just re-run loop
          mr2 = images[j,]            # with spotty internet connection
          CAID2 = mr2$cognitive_contrast_cogatlas_id
          score = CogatSimilar(CAID1,CAID2)
          similarities[i,j] = score
          similarities[j,i] = score
      }
    }    
  }
}


# Export to file
output_file = "/home/vanessa/Documents/Work/BRAINMETA/reverse_inference/contrast_defined_images_wang.tsv"
write.csv(similarities,filename=output_file,sep="\t")