from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import os
import json
from mmdet3d.datasets.pipelines import Compose
import numpy as np
import copy

@DATASETS.register_module()
class NuscDepthDataset(Dataset):

    CLASSES=2
    def __init__(self, data_path='data/nuscenes/depth_maps/train/detpth_map', 
                pipeline=None, training=True, **kwargs):
        self.depth_root = os.path.join(data_path, 'depth_data')
        self.meta_path = os.path.join(data_path, 'meta.json')
        self.training = training
        #from IPython import embed
        #embed()
        if training:
            #self.data_infos = self.load_annotations()[:-20000]
            self.data_infos = self.load_annotations()[:-20000]
        else:
            #self.data_infos = self.load_annotations()[-20000:]
            self.data_infos = self.load_annotations()[-20000:]
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        # not important
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self):

        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        data_infos = []

        for data_info in self.meta:
            img_path = data_info['img_path']
            depth_path = data_info['depth_path']+'.npy' # depth_path+'.npy'
            
            tmp = {'img_info':{'filename':img_path}, 
                'npy_info': {'filename': depth_path}}

            data_infos.append(tmp)
        
        return data_infos

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img_prefix'] = None
        data_info['flip'] = None
        data_info['flip_direction'] = None
        return data_info
    
    def __len__(self,):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.get_data_info(idx)

        data = self.pipeline(data_info)

        return data

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None):
        # [(loss.item(), twenty_acc.item(), ten_acc.item(), five_acc.item(), one_acc.item())]
        # print(results)

        
        abs_diff = [res[0] for res in results]
        abs_rel = [res[1] for res in results]
        sq_rel = [res[2] for res in results]
        rmse = [res[3] for res in results]
        rmse_log = [res[4] for res in results]
        losses = [res[5] for res in results]

        num_batch = len(losses)

        abs_diff = sum(abs_diff) / num_batch
        abs_rel = sum(abs_rel) / num_batch
        sq_rel = sum(sq_rel) / num_batch
        rmse = sum(rmse) / num_batch
        rmse_log = sum(rmse_log) / num_batch
        loss = sum(losses) / num_batch
        


        #print(results, loss)
        ret_dict = {'loss': loss, 
                    'abs_diff': abs_diff, 'abs_rel': abs_rel, 
                    'sq_rel': sq_rel, 'rmse': rmse, 
                    'rmse_log': rmse_log
                     }

        return ret_dict
        

