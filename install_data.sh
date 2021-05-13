#!/bin/bash
# A Shell Script to Download VQA Data
# Bryan Tor - 12/May/2021

dir_path=`pwd`

mkdir processed_data

mkdir data
cd data
data_path=`pwd`

# download annotations data
mkdir annotations
cd annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
cd $data_path

# download questions data
mkdir questions
cd questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_*
cd $data_path

# download image data
mkdir images
cd images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip
unzip train2014.zip
unzip val2014.zip
unzip test2015.zip

cd $dir_path
