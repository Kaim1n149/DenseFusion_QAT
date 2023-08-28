import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import pathlib
import statistics
import time
import argparse
import cv2
import kornia

# from models.DenseNet_cat import DenseNet_half as Model
from models.DenseNet_add import q_DenseNet, q_DenseNet_half

class Fuse:
    """
    fuse with infrared folder and visible folder
    """

    def __init__(self, model_path: str):
        """
        :param model_path: path of pre-trained parameters
        """

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # original
        '''
        params = torch.load(model_path, map_location="cpu")
        self.net = q_DenseNet()
        self.net.load_state_dict(params["net"])
        self.net.to(device)
        self.net.eval()
        '''
        
        #ptsq
        '''
        params = torch.load(model_path)
        self.net = q_DenseNet_half()
        #self.net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        self.net.qconfig = torch.quantization.qconfig.QConfig(activation=torch.ao.quantization.observer.MinMaxObserver.with_args(dtype=torch.quint8),
                                                              weight=torch.ao.quantization.observer.default_observer.with_args(dtype=torch.qint8))
        self.pre_net = torch.quantization.prepare(self.net)
        self.ptsq_net = torch.quantization.convert(self.pre_net)
        self.ptsq_net.load_state_dict(params)
        self.ptsq_net.eval()
        '''
        
        #qat
        
        params = torch.load(model_path, map_location="cpu")
        self.net = q_DenseNet()
        torch.backends.quantized.engine = 'fbgemm'
        qcfg = torch.quantization.get_default_qat_qconfig('fbgemm')
        qact = torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,
                                                     quant_min=0, quant_max=255, dtype=torch.quint8,
                                                     qscheme=torch.per_tensor_affine, reduce_range=False)
        qcfg = torch.quantization.QConfig(activation=qact, weight=qcfg.weight)
        self.net.qconfig = qcfg
        self.net = torch.quantization.prepare_qat(self.net, inplace=False)
        self.net = torch.quantization.convert(self.net, inplace=False)
        self.net.load_state_dict(params["net"])
        self.net.eval()
        
        
    def __call__(self, i1_folder: str, i2_folder: str, dst: str, fuse_type = None):
        """
        fuse with i1 folder and vi folder and save fusion image into dst
        :param i1_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))

        # image list
        i1_folder = pathlib.Path(i1_folder)
        i2_folder = pathlib.Path(i2_folder)
        i1_list = sorted([x for x in sorted(i1_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])
        i2_list = sorted([x for x in sorted(i2_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(i1_list, i2_list))

        for i1_path, i2_path in rge:
            start = time.time()

            # check image name
            i1_name = i1_path.stem
            i2_name = i2_path.stem
            rge.set_description(f'fusing {i1_name}')
            # assert i1_name == vi_name

            # read image
            i1, i2 = self._imread(str(i1_path), str(i2_path), fuse_type = fuse_type)
            i1 = i1.unsqueeze(0)#.to(self.device)
            i2 = i2.unsqueeze(0)#.to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(i1, i2)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


            # save fusion tensor
            fu_path = pathlib.Path(dst, i1_path.name)
            self._imsave(fu_path, fu)

            end = time.time()
            fuse_time.append(end - start)
        
        # time analysis
        if len(fuse_time) > 2:
            mean = statistics.mean(fuse_time[1:])
            print('fps (equivalence): {:.2f}'.format(1. / mean))

        else:
            print(f'fuse avg time: {fuse_time[0]:.2f}')


    @torch.no_grad()
    def _forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        fusion = self.net(i1, i2)
        return fusion

    @staticmethod
    def _imread(i1_path: str, i2_path: str, flags=cv2.IMREAD_GRAYSCALE, fuse_type = None) -> torch.Tensor:
        i1_cv = cv2.imread(i1_path, flags).astype('float32')
        
        i2_cv = cv2.imread(i2_path, flags).astype('float32')
        height, width = i1_cv.shape[:2]
        # if fuse_type == 'black_ir':
        #     i1_cv[True] = 0
        # elif fuse_type == 'black_vi':
        #     i2_cv[True] = 0
        # if fuse_type == 'white_ir':
        #     i1_cv[True] = 255
        # if fuse_type == 'white_vi':
        #     i2_cv[True] = 255
            
        i1_ts = kornia.utils.image_to_tensor(i1_cv / 255.0).type(torch.FloatTensor)
        i2_ts = kornia.utils.image_to_tensor(i2_cv / 255.0).type(torch.FloatTensor)
        return i1_ts, i2_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(path), im_cv)



if __name__ == '__main__':
    model = 'densenet_add'
    # original
    #f = Fuse(f"./cache/{model}/best.pth")
    # qat
    f = Fuse(f"./cache/{model}/qat.pth")

    parser = argparse.ArgumentParser()
    #LLVIP
    #parser.add_argument("--i1", default='../datasets/LLVIP/infrared/test', help="ir path")
    #parser.add_argument("--i2", default='../datasets/LLVIP/visible/test', help="vi path")
    #NCHU_0007
    #parser.add_argument("--i1", default='../datasets/NCHU/dongshi_0007/ir', help="ir path")
    #parser.add_argument("--i2", default='../datasets/NCHU/dongshi_0007/vi', help="vi path")
    #For_PPT
    parser.add_argument("--i1", default='../datasets/LLVIP/for_ppt/ir', help="ir path")
    parser.add_argument("--i2", default='../datasets/LLVIP/for_ppt/vi', help="vi path")
    #TNO Nato
    #parser.add_argument("--i1", default='../datasets/TNO/TNO_Duine_Nato_Tree/Nato_sequence/thermal', help="ir path")
    #parser.add_argument("--i2", default='../datasets/TNO/TNO_Duine_Nato_Tree/Nato_sequence/visual', help="vi path")
    args = parser.parse_args()
    
    # f(args.i1, args.i2, f'runs/test/{model}/black_ir', 'black_ir')
    # f(args.i1, args.i2, f'runs/test/{model}/black_vi', 'black_vi')
    # f(args.i1, args.i2, f'runs/test/{model}/white_ir', 'white_ir')
    # f(args.i1, args.i2, f'runs/test/{model}/white_vi', 'white_vi')
    
    #f(args.i1, args.i2, f'runs/test/{model}/fuse')
    #f(args.i1, args.i2, f'runs/nchu/{model}/fuse')
    f(args.i1, args.i2, f'runs/for_ppt/{model}/fuse')
    #f(args.i1, args.i2, f'runs/tno/nato/{model}/fuse')
