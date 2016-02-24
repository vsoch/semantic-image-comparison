df=read.csv("data/classification_confusion_binary_4mm_norm.tsv",sep="\t",row.names=1)
images = read.csv("../../doc/contrast_defined_images_filtered.tsv",sep="\t")
colnames(df) = gsub("X","",colnames(df)) # Stupid R adds an X...

colnames(df) = as.character(images$cognitive_contrast_cogatlas[images$image_id %in% colnames(df)])

library(pheatmap)
pdf("concept_confusion_norm.pdf",width=20,height=20)
pheatmap(df)
pheatmap(df,cluster_rows=FALSE,cluster_cols=FALSE)
dev.off()
