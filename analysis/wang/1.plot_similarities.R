library(pheatmap)

# Read in spatial and wang (graph-based) similarities
spatial = read.csv("data/contrast_defined_images_pearsonpd_similarity.tsv",sep="\t",head=TRUE,stringsAsFactors=FALSE,row.names=1)
graph = read.csv("data/contrast_defined_images_wang.tsv",sep=",",head=TRUE,stringsAsFactors=FALSE,row.names=1)
images = read.csv("data/contrast_defined_images_filtered.tsv",sep="\t",head=TRUE,stringsAsFactors=FALSE,row.names=1)

# These should be exactly the same, but we will separate to be extra careful
spatial_contrasts = c()
for (image_id in rownames(spatial)){
  con = images$cognitive_contrast_cogatlas[images$image_id==image_id]
  spatial_contrasts=c(spatial_contrasts,con)
}
colnames(spatial) = spatial_contrasts

graph_contrasts = c()
for (image_id in rownames(graph)){
  con = images$cognitive_contrast_cogatlas[images$image_id==image_id]
  graph_contrasts=c(graph_contrasts,con)
}
colnames(graph) = graph_contrasts

# Generate a scatterplot of scores

# We just want to compare images by images
library(ggplot2)
library(plyr)
library(reshape2)
spatial = read.csv("data/contrast_defined_images_pearsonpd_similarity.tsv",sep="\t",head=TRUE,stringsAsFactors=FALSE,row.names=1)
graph = read.csv("data/contrast_defined_images_wang.tsv",sep=",",head=TRUE,stringsAsFactors=FALSE,row.names=1)
colnames(spatial) = gsub("X","",colnames(spatial))
colnames(graph) = gsub("X","",colnames(graph))
gflat = melt(as.matrix(graph))
sflat = melt(as.matrix(spatial))
both = as.data.frame(cbind(gflat$value,sflat$value,image1=gflat$Var1,image2=gflat$Var2),stringsAsFactors=FALSE)

# Remove images compared to themselves
both = both[-which(both$image1==both$image2),]
colnames(both) = c("graph_score","spatial_score","image1","image2")

# Add task names (not enough overlap contrasts to be meaningful)
image1_task = c()
image2_task = c()
for (rownum in 0:nrow(both)){
  task1 = images$cognitive_paradigm_cogatlas[images$image_id==both$image1[rownum]]
  task2 = images$cognitive_paradigm_cogatlas[images$image_id==both$image2[rownum]]
  image1_task=c(image1_task,task1)
  image2_task=c(image2_task,task2)
}
both$image1_task = image1_task
both$image2_task = image2_task
color_vector = array(dim=nrow(both))
color_vector[which(both$image1_task==both$image2_task)] = both$image1_task[which(both$image1_task==both$image2_task)]
color_vector[is.na(color_vector)] = "tasks not equal"

both$color = color_vector
png("img/scatter_spatial_semantic_bytask.png", width = 16, height = 12, units = 'in', res = 300)
ggplot(both, aes(x=graph_score,y=spatial_score,color=color,group=color)) + 
  geom_point() +
  xlab("Wang graph metric score") +
  ylab("Spatial comparison score")
dev.off()  

png("img/scatter_spatial_semantic.png", width = 16, height = 12, units = 'in', res = 300)
ggplot(both, aes(x=graph_score,y=spatial_score)) + 
  geom_point() +
  xlab("Wang graph metric score") +
  ylab("Spatial comparison score")
dev.off()  

png("img/scatter_spatial_semantic_image1_task.png", width = 16, height = 12, units = 'in', res = 300)
ggplot(both, aes(x=graph_score,y=spatial_score,color=image1_task)) + 
  geom_point() +
  xlab("Wang graph metric score") +
  ylab("Spatial comparison score")
dev.off()  

png("img/scatter_spatial_semantic_image2_task.png", width = 16, height = 12, units = 'in', res = 300)
ggplot(both, aes(x=graph_score,y=spatial_score,color=image2_task)) + 
  geom_point() +
  xlab("Wang graph metric score") +
  ylab("Spatial comparison score")
dev.off()  

# Just plot as is
pdf("img/graph_vs_spatial_sim.pdf",width=12,height=12)
pheatmap(graph,cluster_rows=FALSE,cluster_cols=FALSE,title="Wang Graph Similarity")
pheatmap(spatial,cluster_rows=FALSE,cluster_cols=FALSE,title="Spatial Similarity, Pearson")
dev.off()

# With clustering
pdf("img/graph_vs_spatial_sim_clustering.pdf",width=12,height=12)
pheatmap(graph,title="Wang Graph Similarity")
pheatmap(spatial,title="Spatial Similarity, Pearson")
dev.off()

# Now read in RSA score matrix
rsa = read.csv("data/spatial_semantic_rsa_df_abs.tsv",row.names=1,sep="\t")

# Plot rsa scores
tmp = rsa[with(rsa, order(-RSA)), ]
rownames(tmp) = seq(1,nrow(tmp))
tmp$sort = as.numeric(rownames(tmp))
ggplot(tmp, aes(x=sort,y=RSA,fill=RSA)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("Cognitive Atlas Task or Concept") +
  ylab(paste("RSA Score")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$name) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")

# Now for file
png("img/rsa_cognitiveatlas.png", width = 12, height = 12, units = 'in', res = 300)
ggplot(tmp, aes(x=sort,y=RSA,fill=RSA)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("Cognitive Atlas Task or Concept") +
  ylab(paste("RSA Score")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$name) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")
dev.off()