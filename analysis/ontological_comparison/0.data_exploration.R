library(plyr)
library(dplyr)
library(ggplot2)

setwd("/home/vanessa/Documents/Dropbox/Code/Python/brainmeta/ontological_comparison/cluster/classification-framework/analysis")

# Reading in the result data

ri_score = read.csv("data/reverse_inference_scores.tsv",sep="\t",stringsAsFactors=FALSE,row.names=1) # reverse inference scores
#counts_in = read.csv("data/reverse_inference_counts_in.tsv",sep="\t",stringsAsFactors=FALSE,row.names=1) # bayes for query images, ranges in
#counts_out = read.csv("data/reverse_inference_counts_out.tsv",sep="\t",stringsAsFactors=FALSE,row.names=1) # bayes for query images, ranges in

# Let's look at the overall posterior scores
hist(as.matrix(ri_score$ri_distance),main="Reverse Inference Scores",col="orange",xlab="posterior probability")

# Read in all groups
groups = read.csv("data/groups/all_groups.tsv",sep="\t",stringsAsFactors=FALSE)
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

# Calculate accuracy for each node group
# Count evidence for (meaning bayes_in > bayes_out or against (bayes_out > bayes_in)) each concept
# for each of ranges and bin data
for (threshold in seq(0,1,by=0.05)){
  cat("Parsing",threshold,"\n")  
  accuracies = c()
  for (node in nodes){
    # Find in group
    group = groups[groups$group==node,]
    in_group = group$image_ids[which(group$direction=="in")]
    out_group = group$image_ids[which(group$direction=="out")]
    # Get reverse inference scores
    if (node %in% colnames(ri_score)){
      scores = ri_score[,node]
      names(scores) = rownames(ri_score)
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
  
  }
  df = rbind(df,accuracies)
}

# Now look at accuracies for each threshold!
rownames(df) = seq(1,nrow(df))
colnames(df) = c("nid","TP","FP","TN","FN","sensitivity","specificity","accuracy","in_count","out_count","threshold")
df = as.data.frame(df,stringsAsFactors=FALSE)
save(df,file="accuracies_df_all.rda")

# Plot a basic ROC for each class
pdf("roc_all.pdf")
nodes = unique(df$nid)
pdf("roc_gr30.pdf")
for (node in nodes){
  subset = df[df$nid==node,]
  N = unique(subset$in_count)
  if (as.numeric(N)>30){
    title = paste("ROC Curve ",as.character(node_lookup[node])," N=(",N,")",sep="")
    plot(1-as.numeric(subset$specificity),as.numeric(subset$sensitivity),
         xlab="1-specificity",ylab="sensitivity",main=title,
         xlim=c(0,1),ylim=c(0,1),type="n")
    lines(1-as.numeric(subset$specificity),as.numeric(subset$sensitivity),col="blue",lwd=2,xlim=c(0,1),ylim=c(0,1))
    lines(seq(0,1,0.05),seq(0,1,0.05),col="red",lwd=2,xlim=c(0,1),ylim=c(0,1))
  }
}
dev.off()



# Now we want to assess a multilabel confusion matrix for each threshold.

# Here is a function to assess a multilabel confusion matrix
# From Sanmi Koyejo
# computes the confusion between labels Zt and predictions Ze.
# Assumes that Zt is coded as 0/1 0r -1/+1
# Assumes that the threshold has already been applied to Ze, so sign(Ze) corresponds to a decision
# Includes optional normalization wrt the rows

multilabel_confusion = function(Zt, Ze, normalize=TRUE) {
  L = dim(Zt)[2]
  M = array(0,dim=c(L, L))
  
  for (ix in seq(1,L)) {
    t = Zt[,ix]>0
    for (jx in seq(1,L)){
      p = Ze[,jx]>0
      M[ix, jx] = length(which((p & t)==TRUE))
    }
  }
  
  # To normalize, we divide by number of images tagged with the concept
  if (normalize==TRUE){
    # The images are in rows, concepts in columns
    # so getting a sum for each column --> total number of images tagged
    msums = as.numeric(colSums(Zt))
        for (ix in seq(1,nrow(M))) {
          for (iy in seq(1,ncol(M))){
            M[ix,iy] = M[ix,iy]/msums[ix]
          }
        }
  }
  return(M)
}

unique_images = unique(image_ids)

# Our inputs are Zt, the labels, and Ze, the predictions.
# Each is an N by M matrix of N images and M contrasts
# A 1 at index Ze[N,M] indicates image N is predicted to be concept M
# A 1 at index Zt[N,M] means that this is actually the case 

# First let's build our "actual label" matrix, Zt
Zt = array(0,dim=c(length(unique_images),length(nodes)))
rownames(Zt) = unique_images
colnames(Zt) = nodes
# 1 means labeled == YES, -1 means NO
for (node in nodes){
  # Find in group
  group = groups[groups$group==node,]
  in_group = group$image_ids[which(group$direction=="in")]
  Zt[which(rownames(Zt)%in% in_group),node] = 1
}

# Now let's build a matrix to compare, for each threshold
# We will save a list of score matrices.
pdf("multilabel_confusions.pdf")
for (threshold in seq(0,1,by=0.05)){
  cat("Parsing",threshold,"\n")  
  Ze = array(0,dim=c(length(unique_images),length(nodes)))
  rownames(Ze) = unique_images
  colnames(Ze) = nodes
  for (node in nodes){
    # Find in group
    group = groups[groups$group==node,]
    in_group = group$image_ids[which(group$direction=="in")]
    out_group = group$image_ids[which(group$direction=="out")]
    # Get reverse inference scores
    if (node %in% colnames(ri_score)){
      scores = ri_score[,node]
      names(scores) = rownames(ri_score)
      # This case happens when N=1 for the node in question, since we removed the image from the group. The score should be 1.
      scores[is.na(scores)] = 1
      # Image index will have 1 for belonging to class, 0 otherwise
      real = array(0,dim=length(unique(image_ids)))
      predicted = array(0,dim=length(unique(image_ids)))
      names(real) = unique(image_ids)
      names(predicted) = unique(image_ids)
      correct_predictions = names(which(scores>=threshold))
      Ze[which(rownames(Ze) %in% correct_predictions),node] = 1 
    }
  }
  # Calculate multilabel confusion score
  mat = multilabel_confusion(Zt,Ze,TRUE)
  rownames(mat) = node_lookup[nodes]
  colnames(mat) = node_lookup[nodes]
  pheatmap(mat,cluster_rows=FALSE,cluster_cols=FALSE,main=paste("Multi-label confusion for threshold",threshold),fontsize=4)
}
dev.off()

# Now lets generate a single vector of AUC scores - one for each concept
library(ROCR)

# As a reminder, our matrix Zt has actual labels in it
# Let's calculate an AUC for each node
aucs =c()
nodes_defined = c()
for (node in nodes){
  if (node %in% colnames(ri_score)){
    actual = as.numeric(Zt[which(rownames(Zt)%in%rownames(ri_score)),node])
    # Find in group
    group = groups[groups$group==node,]
    in_group = group$image_ids[which(group$direction=="in")]
    predictions = ri_score[,node]
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
save(aucs,file="data/aucs_139.Rda")
aucdf = as.data.frame(aucs)
aucdf$concept = rownames(aucdf)
tmp = melt(aucdf,id.vars=c("concept"))

# Let's sort! MAKE IT PINK.
tmp = tmp[with(tmp, order(-value)), ]
rownames(tmp) = seq(1,nrow(tmp))
tmp$sort = as.numeric(rownames(tmp))
ggplot(tmp, aes(x=sort,y=value,fill=value)) + 
  geom_bar(stat="identity",ylim=c(0,1)) + 
  xlab("concept") +
  ylab(paste("AUC")) +
  scale_x_discrete(limits=tmp$sort,labels=tmp$concept) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),legend.position="none")



# Pairwise AUC computation pesudocode:
# Algorithm by Sanmi Koyejo, Poldracklab
# Goal: Measure fraction of times when concept i would be confused for concept j with some fixed threshold
# Only evaluate on data subset containing only concept i or only concept j.

# Some definitions:
# dataset = Includes ground truth concepts and probability predictions
# concepts = list of concepts
# auc_matrix = matrix of size #concepts x # concepts: Main output
overlaps = array(0,dim=c(length(nodes),length(nodes)))
colnames(overlaps) = nodes
rownames(overlaps) = nodes

# First let's make a matrix of counts for the number of images shared between 
# pairwise concepts.
for (node1 in nodes){
  for (node2 in nodes){
    if (node1 != node2){
      # Find the datasets shared between the concepts   
      group1 = groups[groups$group==node1,]
      group2 = groups[groups$group==node2,]
      group1_in = group1$image_ids[which(group1$direction=="in")]
      group2_in = group2$image_ids[which(group2$direction=="in")]
      mrset = intersect(group1_in,group2_in)
      overlaps[node1,node2] = length(mrset)
     }
  }
}
# Overlap is sparse, however there is a small set that we can work with, I will move forward with analysis.
pheatmap(overlaps)

nodes = nodes[which(nodes%in%colnames(ri_score))]
auc_matrix = array(0,dim=c(length(nodes),length(nodes)))
colnames(auc_matrix) = nodes
rownames(auc_matrix) = nodes

library(pROC)
# First let's make a matrix of counts for the number of images shared between 
# pairwise concepts.
for (node1 in nodes){
    for (node2 in nodes){
      if (node1 != node2){
        # Find the datasets shared between the concepts   
        group1 = groups[groups$group==node1,]
        group2 = groups[groups$group==node2,]
        group1_in = group1$image_ids[which(group1$direction=="in")]
        group2_in = group2$image_ids[which(group2$direction=="in")]

        # Our base set of values is the entire image set between group1 and group 2
        baseset = union(group1_in,group2_in)

        # Here we will create labels, 1 will indicate belonging to both sets
        labels = array(-1,dim=c(length(baseset)))    
        names(labels) = baseset
        
        # If there are values shared between the sets, these are labels we set to 1
        # because they would be confused
        mrintersect = intersect(group1_in,group2_in)
        if (length(mrintersect) !=0){
          labels[which(names(labels) %in% mrintersect)] = 1
          # Get corresponding ri_scores for each
          ri1 = ri_score[which(rownames(ri_score)%in% names(labels)),node1]
          ri2 = ri_score[which(rownames(ri_score)%in% names(labels)),node2]
          ri1[is.na(ri1)] = 1
          ri2[is.na(ri2)] = 1
          score_vector = ri1 / ri2

          # if there is only one value, this means that bayes factor is 1, AUC is 0.5
          # This means that the scores are EXACTLY the same
          if (length(unique(score_vector))!=1){
            pred = roc(score_vector, labels)
            auc_matrix[node1, node2] = as.numeric(pred$auc)
          } else {
            auc_matrix[node1, node2] = 1
          }          
        # No overlap means that the AUC value is 1 because we always get it right
        } else {
          auc_matrix[node1, node2] = 1
        }
      }
    }
}
colnames(auc_matrix) = node_lookup[colnames(auc_matrix)]
rownames(auc_matrix) = node_lookup[rownames(auc_matrix)]
pdf("auc_matricesv3_pt5base.pdf")
pheatmap(auc_matrix,main="AUC Matrix for Pairwise Concepts",fontsize = 4)
pheatmap(auc_matrix,main="AUC Matrix for Pairwise Concepts",cluster_rows=FALSE,cluster_cols=FALSE,fontsize=4)
dev.off()

# Notes:
#  - you can compute avg(auc_matrix) as a summary
#  - Will not be symmetric, so you need to run all off-diagonal pairs

# Function to calculate confidence intervals
get_ci = function(dat,direction="upper"){
  error = qnorm(0.975)*sd(dat)/sqrt(length(dat))
  if (direction=="upper"){
    return(mean(dat)+error)
  } else {
    return(mean(dat)-error)    
  }
}


# Finally, let's explore the relationship between the size of the "in" set and the scores
library(dplyr)
library(plyr)
size_results = list.files("data/size_results/",full.names=TRUE)
pdf("explore_concept_set_sizes.pdf")
for (result in size_results){
  res = read.csv(result,sep="\t",row.names=1,stringsAsFactors=FALSE)
  if (nrow(res)>0){
    ressum = ddply(res,"in_count",summarise,mean_score=mean(ri_score),ci_up=get_ci(ri_score,"upper"),ci_down=get_ci(ri_score,"lower"))
    node_name = as.character(node_lookup[unique(res$node)])
    # First let's look at how the score changes with size
    p = ggplot(ressum, aes(x=in_count,y=mean_score,ymax=ci_up,ymin=ci_down)) + 
      geom_line(size=1.5) +
      ylim(0,1) +
      geom_ribbon(alpha=0.15,linetype=0) +
      xlab(paste("size of image set for",node_name)) +
      ylab("Mean RI Score") 
      print(p)
  }
}
dev.off()