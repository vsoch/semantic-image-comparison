# Finally, let's explore the relationship between the size of the "in" set and the scores
library(plyr)
library(dplyr)

# Here is a function to get confidence intervals
get_ci = function(dat,direction="upper"){
  error = qnorm(0.975)*sd(dat)/sqrt(length(dat))
  if (direction=="upper"){
    return(mean(dat)+error)
  } else {
    return(mean(dat)-error)    
  }
}

size_results = list.files("data/size_results",full.names=TRUE)
pdf("img/explore_concept_set_sizes.pdf")
for (result in size_results){
  res = read.csv(result,sep="\t",row.names=1,stringsAsFactors=FALSE)
  if (nrow(res)>0){
    ressum = ddply(res,"in_count",summarise,mean_score=mean(ri_score),ci_up=get_ci(ri_score,"upper"),ci_down=get_ci(ri_score,"lower"))
    node_name = as.character(node_lookup[unique(res$node)])
    # First let's look at how the score changes with size
    if (length(unique(ressum$in_count))>1){
        p = ggplot(ressum, aes(x=in_count,y=mean_score,ymax=ci_up,ymin=ci_down)) + 
        geom_line(size=1.5) +
        ylim(0,1) +
        geom_ribbon(alpha=0.15,linetype=0) +
        xlab(paste("size of image set for",node_name)) +
        ylab("Mean RI Score") 
      print(p)
    }
  }
}
dev.off()