import os
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms

class VITONDataset(data.Dataset):
    def __init__(self, dataset_dir="",          # /home/jovyan/users/almas/VTON/datasets/VTON-HD/
                       dataset_mode="train", dataset_list="train_pairs.txt",
                       load_height=512, load_width=384):
        super(VITONDataset, self).__init__()
        self.load_height = load_height
        self.load_width = load_width
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.data_path = osp.join(dataset_dir, dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_names = [] 
        c_names = []
        with open(osp.join(self.dataset_dir, self.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names    
        self.c_names = dict()
        # self.c_names['paired'] = c_names
        ###img跟cloth名称相同，在不同文件夹下
        self.c_names['paired'] = img_names

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.load_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)
            cm[key].unsqueeze_(0)


        img = Image.open(osp.join(self.data_path, 'image', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        # img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        agnostic_mask = Image.open(osp.join(self.data_path, 'agnostic-mask', img_name.replace('.jpg', '_mask.png')))
        agnostic_mask = transforms.Resize(self.load_width, interpolation=2)(agnostic_mask)
        img = self.transform(img)
        agnostic_mask = self.mask_transform(agnostic_mask)  # [-1,1]
        
        result = {
            'img_name': img_name,
            # 'c_name': c_name,
            'img': img,
            'agnostic_mask': agnostic_mask,
            # 'parse_agnostic': new_parse_agnostic_map,
            # 'pose': pose_rgb,
            'cloth': c["paired"],
            'cloth_mask': cm["paired"],
        }
        return result

    def __len__(self):
        return len(self.img_names)
    
    
class VITONDataLoader:
    def __init__(self, dataset, batch_size, shuffle, workers):
        super(VITONDataLoader, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        
        if self.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=(train_sampler is None),
                num_workers=self.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.data_iter = self.data_loader.__iter__()
        

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch

class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
    
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
        self.data = self.load_data()
    
    def load_data(self):
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class VITONHDTestDataset(InferenceDataset):
    def load_data(self):
        assert os.path.exists(pair_txt:=os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
        output_dir = os.path.join(self.args.output_dir, "vitonhd", 'unpaired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")
            if os.path.exists(os.path.join(output_dir, person_img)):
                continue
            if self.args.eval_pair:
                cloth_img = person_img
            data.append({
                'person_name': person_img,
                'person': os.path.join(self.args.data_root_path, 'image', person_img),
                'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', person_img.replace('.jpg', '_mask.png')),
            })
        return data