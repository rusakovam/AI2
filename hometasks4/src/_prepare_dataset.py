import requests
import tarfile
import os


# mkdir
datasetSrcDir = "../dataset_src/"
if not os.path.exists(datasetSrcDir):
    os.makedirs(datasetSrcDir)


# download
url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
fileName = datasetSrcDir + "101_ObjectCategories.tar.gz"

if not os.path.exists(fileName):
    f = requests.get(url, allow_redirects=True)
    open(fileName, 'wb').write(f.content)


# extract
extractedDir = fileName.rstrip(".tar.gz") + "/"
if not os.path.exists(extractedDir):
    if fileName.endswith("tar.gz"):
        tar = tarfile.open(fileName, "r:gz")
        tar.extractall(path=datasetSrcDir)
        tar.close()


# elif (fileName.endswith("tar")):
#     tar = tarfile.open(fileName, "r:")
#     tar.extractall(path=datasetSrcDir)
#     tar.close()

# move and shuffle
datasetDir = "../dataset/"
if not os.path.exists(datasetDir):
    os.mkdir(datasetDir)

categories = os.listdir(extractedDir)
for cat in categories:
    files = os.listdir(extractedDir + cat)
    for f in files:
        os.rename(extractedDir + cat + "/" + f, datasetDir + cat + "_" + f)
    os.rmdir(extractedDir + cat)

os.rmdir(extractedDir)
