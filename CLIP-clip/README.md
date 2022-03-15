# CLIP-clip
# Step 0: Install the packages
`pip install requirement.txt`

# Step 1: Download data
`sh download.sh`

# Step 2: Run CLIP-clip
- COCO: `python clip_pred_karpathy.py --name coco`
- Flickr: `python clip_pred_karpathy.py --name flickr`

# Step 3: Evaluate Recall and Gender Bias
- COCO-1K: `python clip_eval.py --name coco --num_clip 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400 --topK 1,5,10 --dataset_type coco1k --plot --print`
  - add `--print` if you want to print the quantitative numbers (Tab. 2 in the report)
  - add `--plot` if you want to plot the Fig. 3
- COCO-5K: `python clip_eval.py --name coco --num_clip 0,20 --topK 1,5,10 --dataset_type coco5k --print`
- FLICKR30K: `python clip_eval.py --name flickr30k --num_clip 0,20 --topK 1,5,10 --dataset_type flickr30k --print`

# (Optional) Evaluate the result of Activation Maximization on MS-COCO
- `python clip_CAM.py`
