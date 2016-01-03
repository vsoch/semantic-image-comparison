options(java.parameters = "-Xmx4g") # This ensures we don't run out of memory
library(CogatSimilar) # https://github.com/CognitiveAtlas/cogat-similaR

args <- commandArgs(TRUE)
i = as.numeric(args[1])
j = as.numeric(args[2])
output_file = args[3]
image_file = args[4]

# Read in table with images
images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)

cat("Processing",i,"of",nrow(images),"\n")
mr1 = images[i,]
CAID1 = mr1$cognitive_contrast_cogatlas_id
mr2 = images[j,]
CAID2 = mr2$cognitive_contrast_cogatlas_id
score = CogatSimilar(CAID1,CAID2)

result = list()
result$score = score
result$row1 = i
result$row2 = j

# Export to file
save(result,file=output_file)