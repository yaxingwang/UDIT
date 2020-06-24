# -*- coding: utf-8 -*-

# python scripts/traintest.py sun rain sun2rain

import sys
import random
from os import listdir, makedirs
from os.path import isfile, join
import shutil

DATA = "dataset"
DOMAIN_FOLDER = "../domains"
DOMAINS = ['sun', 'rain']
DATASET = ['sun2rain']
TEST_SIZE = 150

def createFolder(base, *name):
    try:
        makedirs(join(base, '/'.join(name)))
        print("%s"%(join(base, '/'.join(name))))
    except OSError:
        print("Folder %s already exists, skipping folder creation!"%(join(base, '/'.join(name))))

def copyImages(dom, files, valset, dset, folder):
    try:
        print("Copying from %s to %s"%(join(DOMAIN_FOLDER, dom), join(DATA, dset, folder)))
        [shutil.copy(join(DOMAIN_FOLDER, dom, files[x]), join(DATA, dset, folder)) for x in valset]
    except IOError as e:
        print("Unable to copy file %s" % e)
    except:
        print("Unexpected error!")

def main():
    print("Starting splitting images into train - test folders....\n")
    print("Checking arguments....\n")
    args = sys.argv

    if len(args) == 1:
        print("Make sure to add arguments: python traintest.py FIRST_DOMAIN SECOND_DOMAIN DATASET_NAME\n")
        exit(-1)

    domA = args[1]
    domB = args[2]
    dset = args[3]

    if domA not in DOMAINS or domB not in DOMAINS:
        print("Wrong domain names, currently only allowed:\n")
        print(' '.join(DOMAINS))
        exit(-1)

    if dset not in DATASET:
        print("Wrong dataset name, currently only allowed:\n")
        print(' '.join(DATASET))
        exit(-1)

    print("Reading images....\n")
    folderA = join(DOMAIN_FOLDER, domA)
    folderB = join(DOMAIN_FOLDER, domB)

    filesA =  [f for f in listdir(folderA) if isfile(join(folderA, f))]
    filesB =  [f for f in listdir(folderB) if isfile(join(folderB, f))]

    print("Creating folder structures....\n")
    createFolder(DATA, dset)
    createFolder(DATA, dset, "trainA")
    createFolder(DATA, dset, "trainB")
    createFolder(DATA, dset, "testB")
    createFolder(DATA, dset, "testA")
    print("")

    print("Total images in domain %s: %d"%(domA, len(filesA)))
    print("Total images in domain %s: %d\n" % (domB, len(filesB)))
    print("Selecting %d random images from both domains for testing...."%TEST_SIZE)

    # Randomly select ~TEST_SIZE images from train set and make ist the val set
    valsetA = random.sample(xrange(0, len(filesA)), TEST_SIZE)
    valsetB = random.sample(xrange(0, len(filesB)), TEST_SIZE)

    trainsetA = [x for x in xrange(0, len(filesA)) if x not in valsetA]
    trainsetB = [x for x in xrange(0, len(filesB)) if x not in valsetB]

    print("Copying images to final folder structure....\n")
    copyImages(domA, filesA, valsetA, dset, "testA")
    copyImages(domA, filesA, trainsetA, dset, "trainA")
    copyImages(domB, filesB, valsetB, dset, "testB")
    copyImages(domB, filesB, trainsetB, dset, "trainB")

    print("Successfully processed images....\n")

if __name__ == '__main__':
    main()