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
rsa = read.csv("data/spatial_semantic_rsa_df.tsv",row.names=1,sep="\t")

# Plot rsa scores
library(ggplot2)
tmp = rsa[with(rsa, order(-RSA)), ]
rownames(tmp) = seq(1,nrow(tmp))
tmp$sort = as.numeric(rownames(tmp))
ggplot(tmp, aes(x=sort,y=RSA,fill=RSA)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("Cognitive Atlas Task or Concept") +
  ylab(paste("RSA Score")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$name) +
  theme(axis.teuxt.x = element_text(angle = 90, hjust = 1),legend.position="none")

# Now for file
png("img/rsa_cognitiveatlas.png", width = 12, height = 12, units = 'in', res = 300)
ggplot(tmp, aes(x=sort,y=RSA,fill=RSA)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("Cognitive Atlas Task or Concept") +
  ylab(paste("RSA Score")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$name) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")
dev.off()