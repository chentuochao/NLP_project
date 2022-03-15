wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
mkdir data
cd data
mkdir COCO
cd COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip annotations_trainval2014.zip
unzip train2014.zip
unzip val2014.zip
cd ../
Download flickr30k images. (Need to fill out the form.)
tar -xvzf flickr30k-images.tar.gz
cd ../
