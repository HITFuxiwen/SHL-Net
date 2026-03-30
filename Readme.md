# SHL-Net
This project provides the code and results for 'SHL-Net: Semantics-Enhanced Network for Localizing Harmonized Image Splicing'

# Network Architecture
   <div align=center>
   <img src="https://github.com/HITFuxiwen/SHL-Net/images/architecture.png">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.9.0
# Training
Download [pvt_v2_b2.pth] https://drive.google.com/file/d/11j4KBAaNDnBUSDis2mCdhc9vK3F-56Nv/view?usp=drive_link , and put it in './model/'. 

Download [sam_vit_l_0b3195.pth] https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth, and put it in './model/'. 

Modify paths of datasets, then run train.py.

Note: Our main model is under './model/SHLNet_models.py' 

# Pre-trained model and testing
1. Download the pre-trained models (link: https://pan.baidu.com/s/1wGZq20Uff8n7Gfi5nYItGQ Extraction Code: k8vy) , and put them in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test.py.

# Evaluation Tool
Modify paths of your predictions and corresponding groundtruth in evalnew.py, then run evalnew.py
