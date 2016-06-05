# Read in table with images

args = commandArgs(TRUE)
base = args[1]

image_file = paste(base,"/results/contrast_defined_images_filtered.tsv",sep="")

images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)

# Define output folders
output_folder = paste(base,"/data/","wang_scores",sep="")

for (i in 1:nrow(images)){ # i is the row number
  for (j in 1:nrow(images)){ # i is the row number
      if (i<=j){
        outfile = paste(output_folder,"/",i,"_",j,"_wangsim.Rda",sep="")
        if (!file.exists(outfile)) {
          jobby = paste(i,"_",j,".job",sep="")
          sink(paste(".job/",jobby,sep=""))
          cat("#!/bin/bash\n")
          cat("#SBATCH --job-name=",jobby,"\n",sep="")  
          cat("#SBATCH --output=.out/",jobby,".out\n",sep="")  
          cat("#SBATCH --error=.out/",jobby,".err\n",sep="")  
          cat("#SBATCH --time=1-00:00\n",sep="")
          cat("#SBATCH --mem=12000\n",sep="")
          cat("Rscript 5.wang_graph_comparison_sherlock.R",i,j,outfile,image_file,"\n")
          sink()
      
          # SUBMIT R SCRIPT TO RUN ON CLUSTER  
          system(paste("sbatch -p russpold --qos russpold ",paste(".job/",jobby,sep="")))
       }
    }
  }
}

## PIPELINE TO RUN FOR MISSING ROWS/COLS LOADED FROM FILE
# see 7.compile_wang_comparison.R for where this file comes from
# If we have missing, load
load("missing.Rda")

for (m in 1:nrow(missing)){
  i = missing[m,1]
  j = missing[m,2]
  outfile = paste(output_folder,"/",i,"_",j,"_wangsim.Rda",sep="")
  if (!file.exists(outfile)) {
      jobby = paste(i,"_",j,".job",sep="")
      cat(jobby,"\n")
      sink(paste(".job/",jobby,sep=""))
      cat("#!/bin/bash\n")
      cat("#SBATCH --job-name=",jobby,"\n",sep="")  
      cat("#SBATCH --output=.out/",jobby,".out\n",sep="")  
      cat("#SBATCH --error=.out/",jobby,".err\n",sep="")  
      cat("#SBATCH --time=1-00:00\n",sep="")
      cat("#SBATCH --mem=12000\n",sep="")
      cat("Rscript 5.wang_graph_comparison_sherlock.R",i,j,outfile,image_file,"\n")
      sink()
        
      # SUBMIT R SCRIPT TO RUN ON CLUSTER  
      system(paste("sbatch -p russpold --qos russpold ",paste(".jobs/",jobby,sep="")))
  }
}

