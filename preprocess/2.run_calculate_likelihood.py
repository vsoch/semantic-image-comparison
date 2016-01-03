#!/usr/bin/python
from glob import glob
import sys
import os

base = sys.argv[1]
data = "%s/data" %base        # mostly images
likelihood_pickles = glob("%s/likelihood/*.pkl" %(data))
tables_folder = "%s/likelihood/tables" %(data) # output folder for likelihood tables

# This first section will make likelihood tables for entire image set. We need these
# tables for the "threshold" and "binary" versions of the reverse inference analysis.

if not os.path.exists(tables_folder):
    os.mkdir(tables_folder)

for p in range(0,len(likelihood_pickles)):
    pkl = likelihood_pickles[p]
    contrast_id = os.path.split(pkl)[-1].split("_")[-1].replace(".pkl","")
    filey = ".jobs/revinf_%s.job" %(contrast_id)
    filey = open(filey,"w")
    filey.writelines("#!/bin/bash\n")
    filey.writelines("#SBATCH --job-name=%s\n" %(contrast_id))
    filey.writelines("#SBATCH --output=.out/%s.out\n" %(contrast_id))
    filey.writelines("#SBATCH --error=.out/%s.err\n" %(contrast_id))
    filey.writelines("#SBATCH --time=2-00:00\n")
    filey.writelines("#SBATCH --mem=64000\n")
    filey.writelines("python 2.calculate_likelihood.py %s %s" %(pkl, tables_folder))
    filey.close()
    os.system("sbatch -p russpold " + ".jobs/revinf_%s.job" %(contrast_id))
