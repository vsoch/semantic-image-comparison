#!/usr/bin/pythjon

# Functions for VM

import os
import shutil

"""
make_analysis_web_folder

copies a web report to an output folder in the web directory

html_snippet: the html file to copy into the folder
folder_path: the folder to put the file, will be created if does not exist
data_files: additional data files (full paths) to be moved into folder
file_name: the name to give the main web report [default is index.html]

"""
def make_analysis_web_folder(html_snippet,folder_path,data_files=None,file_name="index.html"):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    output_file = "%s/%s" %(folder_path,file_name) 
    filey = open(output_file,"wb")
    filey.writelines(html_snippet)
    filey.close()
    if data_files:
        for data_file in data_files:
            shutil.copyfile(data_file,folder_path)


