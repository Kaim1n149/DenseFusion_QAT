import pathlib
import statistics
import time
import argparse
import cv2
import kornia
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# from models.DenseNet import DenseNet as Model
from models.DenseNet_add import DenseNet_half as Model

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
        # model parameters
        params = torch.load(model_path, map_location='cpu')

        self.net = Model()

        self.net.load_state_dict(params['net'])

        self.net.to(device)
        
        self.net.eval()

    def __call__(self, ir_folder: str, vi_folder: str, dst: str):
        """
        fuse with ir folder and vi folder and save fusion image into dst
        :param ir_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))

        # image list
        ir_folder = pathlib.Path(ir_folder)
        vi_folder = pathlib.Path(vi_folder)
        ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg']]
        vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg']]

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(ir_list, vi_list))

        for ir_path, vi_path in rge:
            start = time.time()

            # check image name
            ir_name = ir_path.stem
            vi_name = vi_path.stem
            rge.set_description(f'fusing {ir_name}')
            # assert ir_name == vi_name

            # read image
            ir, vi, cr, cb = self._imread(str(ir_path), str(vi_path))
            ir = ir.unsqueeze(0)
            vi = vi.unsqueeze(0)
            ir = ir.to(self.device)
            vi = vi.to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(ir, vi)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # save fusion tensor
            fu_path = pathlib.Path(dst, ir_path.name)
            self._imsave(fu_path, fu, cr, cb)

            end = time.time()
            fuse_time.append(end - start)
        
        # time analysis
        if len(fuse_time) > 2:
            mean = statistics.mean(fuse_time[1:])
            print('fps (equivalence): {:.2f}'.format(1. / mean))

        else:
            print(f'fuse avg time: {fuse_time[0]:.2f}')


    @torch.no_grad()
    def _forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fusion = self.net(ir, vi)
        return fusion

    @staticmethod
    def _imread(ir_path: str, vi_path: str, flags=cv2.IMREAD_GRAYSCALE) -> torch.Tensor:
        ir_cv = cv2.imread(ir_path, flags)
        
        bgr_cv = cv2.imread(vi_path)
        height, width = ir_cv.shape[:2]
        bgr_cv = cv2.resize(bgr_cv, (width, height))
        YCrCb_cv = cv2.cvtColor(bgr_cv, cv2.COLOR_BGR2YCR_CB)
        vi_cv, Cr_cv, Cb_cv =  cv2.split(YCrCb_cv)
        
        ir_ts = kornia.utils.image_to_tensor(ir_cv / 255.0).type(torch.FloatTensor)
        vi_ts = kornia.utils.image_to_tensor(vi_cv / 255.0).type(torch.FloatTensor)
        return ir_ts, vi_ts, Cr_cv, Cb_cv

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor, cr, cb):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        im_cv[im_cv>255] = 255

        im_cv = cv2.merge([im_cv.astype('uint8'), cr, cb])
        im_cv = cv2.cvtColor(im_cv, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(str(path), im_cv)
        # assert 0


if __name__ == '__main__':
    model = 'densenet_add_half'
    f = Fuse(f"./cache/{model}/best.pth")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", default='../datasets/test/ir', help="IR path")
    parser.add_argument("--vi", default='../datasets/test/vi', help="VI path")
    args = parser.parse_args()

    f(args.ir, args.vi, f'runs/test/{model}_RGB')