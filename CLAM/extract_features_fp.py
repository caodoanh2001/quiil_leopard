import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
import sys
from datasets_clam.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda')
from torchvision import transforms
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
import glob
# from huggingface_hub import login

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
	
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
	
# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
	
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def uni():
	import sys
	sys.path.append('/workspace/CLAM/timm_versions/0.9.16/')
	import timm
	dict_uni = dict()
	for chunk in glob.glob('/workspace/pretrained_weights/uni/*'): dict_uni.update(torch.load(chunk))
	model = timm.create_model(
		"vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
	)
	# model.load_state_dict(torch.load(dict_uni, map_location="cpu"), strict=True)
	model.load_state_dict(dict_uni, strict=True)
	return model.cuda()

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, custom_transforms=None):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, custom_transforms=custom_transforms)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--features_type', type=str, default='resnet50')
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	slide_dir = args.data_slide_dir
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint', args.features_type)

	if args.features_type == 'resnet50':
		model = resnet50_baseline(pretrained=True)
		model = model.to(device)
		pipeline_transforms = None

	elif args.features_type == 'ctranspath':
		from sometools import ctranspath
		model = ctranspath()
		model.head = nn.Identity()
		model.load_state_dict(torch.load('/workspace/pretrained_weights/ctranspath.pth')['model'], strict=True)
		model = model.to(device)
		pipeline_transforms = trnsfrms_val

	elif args.features_type == 'uni':
		model = uni()
		model = model.to(device)
		pipeline_transforms = trnsfrms_val

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_file_path = bags_dataset[bag_candidate_idx]
		slide_file_path = os.path.join(slide_dir, slide_file_path)
		slide_id = '/'.join(bags_dataset[bag_candidate_idx].split(args.slide_ext)[0].split('/')[-2:]) # For BRCA
		bag_name = slide_id.split('/')[-1] + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		
		if not os.path.exists(output_path):
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			basename = os.path.splitext(os.path.basename(slide_file_path))[0]
			if os.path.exists(h5_file_path):
					pipeline_transform = transform
					output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
					model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
					custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, custom_transforms=pipeline_transform)

					time_elapsed = time.time() - time_start
					print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
					
					file = h5py.File(output_file_path, "r")

					features = file['features'][:]
					print('features size: ', features.shape)
					print('coordinates size: ', file['coords'].shape)
					features = torch.from_numpy(features)
					bag_base, _ = os.path.splitext(bag_name)
					torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
		else:
			print("Already processed!")