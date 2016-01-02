# Read in table with images
image_file = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/results/contrast_defined_images_filtered.tsv"
images = read.csv(image_file,sep="\t",head=TRUE,stringsAsFactors=FALSE)


# Define output folders
output_folder = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/wang_scores"

for (i in 1:nrow(images)){ # i is the row number
  for (j in 1:nrow(images)){ # i is the row number
      if (i<=j){
        outfile = paste(output_folder,"/",i,"_",j,"_wangsim.Rda",sep="")
        if (!file.exists(outfile)) {
          jobby = paste(i,"_",j,".job",sep="")
          sink(paste(".jobs/",jobby,sep=""))
          cat("#!/bin/bash\n")
          cat("#SBATCH --job-name=",jobby,"\n",sep="")  
          cat("#SBATCH --output=.out/",jobby,".out\n",sep="")  
          cat("#SBATCH --error=.out/",jobby,".err\n",sep="")  
          cat("#SBATCH --time=1-00:00\n",sep="")
          cat("#SBATCH --mem=12000\n",sep="")
          cat("Rscript /home/vsochat/SCRIPT/R/brainmeta/1.wang_graph_comparison_sherlock.R",i,j,outfile,image_file,"\n")
          sink()
      
          # SUBMIT R SCRIPT TO RUN ON CLUSTER  
          system(paste("sbatch -p russpold ",paste(".jobs/",jobby,sep="")))
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
      sink(paste(".jobs/",jobby,sep=""))
      cat("#!/bin/bash\n")
      cat("#SBATCH --job-name=",jobby,"\n",sep="")  
      cat("#SBATCH --output=.out/",jobby,".out\n",sep="")  
      cat("#SBATCH --error=.out/",jobby,".err\n",sep="")  
      cat("#SBATCH --time=1-00:00\n",sep="")
      cat("#SBATCH --mem=12000\n",sep="")
      cat("Rscript /home/vsochat/SCRIPT/R/brainmeta/1.wang_graph_comparison_sherlock.R",i,j,outfile,image_file,"\n")
      sink()
        
      # SUBMIT R SCRIPT TO RUN ON CLUSTER  
      system(paste("sbatch -p russpold ",paste(".jobs/",jobby,sep="")))
  }
}

