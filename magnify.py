#! usr/bin/python
# magnify.py is a simple command-line wrapper for ISR

# Import modules 
import os
import sys
import argparse 
import numpy as np
from PIL import Image
from ISR.models import RDN
from ISR.models import RRDN

# Parse command line args 
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputFile", help="Input file", action = "store", required = True)
parser.add_argument("-m", "--model", type=int, default=1, choices=range(1,5), help="Model number (1) GAN 4x, (2) PSNR-large 2x, (3) PSNR-small 2x, (4) noise-cancel 2x; Default = 1")
args = parser.parse_args()

# Handle specified input file 
inputfile = str(args.inputFile)
file_root, file_ext = os.path.splitext(inputfile)
print("Input file is {}".format(inputfile))
img = Image.open(inputfile)
lr_img = np.array(img) 

# Handle specified model 
if args.model == 1:
	model = RRDN(weights='gans')
	tagout = '_gan'
elif args.model == 2:
	model = RDN(weights='psnr-large')
	tagout = '_psnr-large'
elif args.model == 3:
	model = RDN(weights='psnr-small')
	tagout = '_psnr-small'
elif args.model == 4:
	model = RDN(weights='noise-cancel')
	tagout = '_denoise'
else:
	print("Couldn't resolve model choice to valid value")
	
# Enlarge the image using ISR 
print("Enlarging {} using {} now\n".format(args.inputFile, tagout))

# 'by_patch_of_size' option plays nicely with consumer-grade GPUs
sr_img = model.predict(lr_img, by_patch_of_size=50)

# Output to an appropriately tagged file 
outfile = Image.fromarray(sr_img) 
outputname = str(file_root + tagout + '.png')
outfile.save(outputname, 'png', option='optimize')
outfile.close()
