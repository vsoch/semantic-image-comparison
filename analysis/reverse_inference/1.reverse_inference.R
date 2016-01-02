library(plyr)
library(dplyr)
library(ggplot2)

setwd("/home/vanessa/Documents/Dropbox/Code/papers/semantic-comparison/semantic-image-comparison-analysis/analysis/ontological_comparison")

# Reading in the result data - this is the file produced by 4.compile_reverse_inference_results.py
ri_score = read.csv("data/reverse_inference_scores.tsv",sep="\t",stringsAsFactors=FALSE,row.names=1) # reverse inference scores

# Let's look at the overall posterior scores
png("img/ri_hist.png", width = 8, height = 5, units = 'in', res = 300)
hist(as.matrix(ri_score$ri_distance),main="Reverse Inference Scores",col="orange",xlab="posterior probability")
dev.off()

# Read in all groups
groups = read.csv("groups/all_groups.tsv",sep="\t",stringsAsFactors=FALSE)
image_ids = c()
for (image in groups$image){
  image = strsplit(image,"/")[[1]]
  image = as.numeric(strsplit(image[length(image)],"[.]")[[1]][1])
  image_ids = c(image_ids,image)
}
groups$image_ids = image_ids

# Make a lookup table for the node name
nodes = unique(groups$group)
node_lookup = c()
for (node in nodes){
  node_name = unique(groups$name[groups$group==node])
  node_lookup = c(node_lookup,node_name)
}
length(node_lookup) == length(nodes)
names(node_lookup) = nodes

# For each group, calculate an accuracy across thresholds
df = c()

# Thresholds are the values in the column names
#thresholds = as.numeric(gsub("ri_binary_","",colnames(ri_score[7:ncol(ri_score)])))

# Calculate accuracy for each node group
# Count evidence for (meaning bayes_in > bayes_out or against (bayes_out > bayes_in)) each concept
# for each of ranges and bin data
for (threshold in seq(0,1,by=0.01)){
  cat("Parsing",threshold,"\n")  
  for (node in nodes){
    cat("Parsing",node,"\n")  
    accuracies = c()
    # Find in group
    group = groups[groups$group==node,]
    in_group = group$image_ids[which(group$direction=="in")]
    out_group = group$image_ids[which(group$direction=="out")]
    # Get reverse inference scores
    if (node %in% ri_score$node){
      node_scores = ri_score[ri_score$node==node,]
      scores = node_scores[,which(colnames(ri_score)=="ri_distance")]
      names(scores) = node_scores$image_id # image_ids
      # This case happens when N=1 for the node in question, since we removed the image from the group. The score should be 1.
      scores[is.na(scores)] = 1
      # Image index will have 1 for belonging to class, 0 otherwise
      real = array(0,dim=length(unique(image_ids)))
      predicted = array(0,dim=length(unique(image_ids)))
      names(real) = unique(image_ids)
      names(predicted) = unique(image_ids)
      predicted[names(which(scores>=threshold))] = 1
      real[as.character(in_group)] = 1
      # Calculate metrics
      # c("TP","FP","TN","FN","accuracy","in_count","out_count")
      TP = sum(real*predicted)  
      TN = length(intersect(names(which(real==0)),names(which(predicted==0))))
      FP = length(intersect(names(which(real==0)),names(which(predicted==1))))
      FN = length(intersect(names(which(real==1)),names(which(predicted==0))))
      if (TP+FP==0){
          sens = 0
      } else {
          sens = TP / (TP + FN)
      }
      if (TN+FN==0){
         spec = 0        
      } else {
         spec = TN / (TN + FP)
      }
      accuracy = (TP + TN)/ (TP + TN + FP + FN)
      accuracies = rbind(accuracies,c(node,TP,FP,TN,FN,sens,spec,accuracy,length(in_group),length(out_group),threshold))
    }
    df = rbind(df,accuracies)
  }
}

# Now look at accuracies for each threshold!
rownames(df) = seq(1,nrow(df))
colnames(df) = c("nid","TP","FP","TN","FN","sensitivity","specificity","accuracy","in_count","out_count","threshold")
df = as.data.frame(df,stringsAsFactors=FALSE)
save(df,file="result/accuracies_df_nodes.rda")

# Plot a basic ROC for each class
pdf("result/roc_gr4.pdf")
nodes = unique(df$nid)
for (node in nodes){
  subset = df[df$nid==node,]
  N = unique(subset$in_count)
  if (as.numeric(N)>4){
    title = paste("ROC Curve ",as.character(node_lookup[node])," N=(",N,")",sep="")
    plot(1-as.numeric(subset$specificity),as.numeric(subset$sensitivity),
         xlab="1-specificity",ylab="sensitivity",main=title,
         xlim=c(0,1),ylim=c(0,1),type="n")
    lines(1-as.numeric(subset$specificity),as.numeric(subset$sensitivity),col="blue",lwd=2,xlim=c(0,1),ylim=c(0,1))
    lines(seq(0,1,0.05),seq(0,1,0.05),col="red",lwd=2,xlim=c(0,1),ylim=c(0,1))
  }
}

dev.off()

# Now lets generate a single vector of AUC scores - one for each concept
library(ROCR)

# First let's build our "actual label" matrix, Z
Z = array(0,dim=c(length(unique(image_ids)),length(nodes)))
rownames(Z) = unique(image_ids)
colnames(Z) = nodes
# 1 means labeled == YES, -1 means NO
for (node in nodes){
  # Find in group
  group = groups[groups$group==node,]
  in_group = group$image_ids[which(group$direction=="in")]
  Z[which(rownames(Z)%in% in_group),node] = 1
}
write.csv(Z,file="result/node_concepts_binary_df.csv")

# As a reminder, our matrix Z has actual labels in it
# Let's calculate an AUC for each node (concept/term)
aucs =c()
nodes_defined = c()
for (node in nodes){
  if (node %in% ri_score$node){
    # Find in group
    group = groups[groups$group==node,]
    in_group = group$image_ids[which(group$direction=="in")]
    predictions = ri_score[ri_score$node==node,]
    actual = as.numeric(Z[which(rownames(Z)%in%predictions$image_id),node])
    predictions = predictions[,which(colnames(ri_score)=="ri_distance")]
    predictions[is.na(predictions)]=1
    pred = prediction(predictions, actual)
    perf = performance(pred,"auc")
    aucs = c(aucs,perf@y.values[[1]])
    nodes_defined = c(nodes_defined,node)
  }
}

# Now let's plot the AUCs
library(dplyr)
library(reshape2)
names(aucs) = node_lookup[nodes_defined]
save(aucs,file="result/aucs_concepts_132.Rda")
aucdf = as.data.frame(aucs)
aucdf$concept = rownames(aucdf)
tmp = melt(aucdf,id.vars=c("concept"))

# Let's sort! MAKE IT BLUE.
library(ggplot2)
tmp = tmp[with(tmp, order(-value)), ]
rownames(tmp) = seq(1,nrow(tmp))
tmp$sort = as.numeric(rownames(tmp))
ggplot(tmp, aes(x=sort,y=value,fill=value)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("concept") +
  ylab(paste("AUC")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$concept) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")

# Now for file
png("img/aucs_concepts_132.png", width = 12, height = 8, units = 'in', res = 300)
ggplot(tmp, aes(x=sort,y=value,fill=value)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("concept") +
  ylab(paste("AUC")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$concept) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")
dev.off()