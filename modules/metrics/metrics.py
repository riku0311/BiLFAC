import os
from skimage import  img_as_float
import contextlib
import torch
import imageio
import numpy as np
from typing import Dict
from perceptual_quality import pim
from .utils import convert_range, convert_yuvdict_to_tensor, load_image, load_image_array, write_yuv
import cv2
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

data_range = [0, 1]

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class MetricParent:
    def __init__(self, bits=8, max_val=255, mvn=1, name=''):
        self.__name = name
        self.bits = bits
        self.max_val = max_val
        self.__metric_val_number = mvn
        self.metric_name = ''

    def set_bd_n_maxval(self, bitdepth=None, max_val=None):
        if bitdepth is not None:
            self.bits = bitdepth
        if max_val is not None:
            self.max_val = max_val

    def name(self):
        return self.__name

    def metric_val_number(self):
        return self.__metric_val_number

    def calc(self, orig, rec):
        raise NotImplementedError


class PSNRMetric(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args,
                         **kwards,
                         mvn=3,
                         name=['PSNR_Y', 'PSNR_U', 'PSNR_V'])

    def calc(self, org, dec):
        ans = []

        for plane in org:
            a = org[plane].mul((1 << self.bits) - 1)
            b = dec[plane].mul((1 << self.bits) - 1)
            mse = torch.mean((a - b)**2).item()
            if mse == 0.0:
                ans.append(100)
            else:
                ans.append(20 * np.log10(self.max_val) - 10 * np.log10(mse))
        return ans[0]


class MSSSIMTorch(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (PyTorch)')

    def calc(self, orig, rec):
        ans = 0.0
        from pytorch_msssim import ms_ssim
        if 'Y' not in orig or 'Y' not in rec:
            return -100.0
        plane = 'Y'
        a = orig[plane].mul((1 << self.bits) - 1)
        b = rec[plane].mul((1 << self.bits) - 1)
        a.unsqueeze_(0).unsqueeze_(0)
        b.unsqueeze_(0).unsqueeze_(0)
        ans = ms_ssim(a, b, data_range=self.max_val).item()

        return ans


class MSSSIM_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (IQA)')
        from IQA_pytorch.MS_SSIM import MS_SSIM
        self.ms_ssim = MS_SSIM(channels=1)

    def calc(self, orig, rec):
        ans = 0.0
        if 'Y' not in orig or 'Y' not in rec:
            return -100.0
        plane = 'Y'
        b = orig[plane].unsqueeze(0).unsqueeze(0)
        a = rec[plane].unsqueeze(0).unsqueeze(0)
        ans = self.ms_ssim(a, b, as_loss=False).item()

        return ans


class PSNR_HVS(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='PSNR_HVS')

    def pad_img(self, img, mult):
        import math

        import torch.nn.functional as F
        h, w = img.shape[-2:]
        w_diff = int(math.ceil(w / mult) * mult) - w
        h_diff = int(math.ceil(h / mult) * mult) - h
        return F.pad(img, (0, w_diff, 0, h_diff), mode='replicate')

    def calc(self, orig, rec):
        from psnr_hvsm import psnr_hvs_hvsm

        a = orig['Y']
        b = rec['Y']
        a = convert_range(a, data_range, [0, 1])
        b = convert_range(b,data_range, [0, 1])
        a_img = self.pad_img(a.unsqueeze(0).unsqueeze(0), 8).squeeze()
        b_img = self.pad_img(b.unsqueeze(0).unsqueeze(0), 8).squeeze()
        a_img = a_img.cpu().numpy().astype(np.float64)
        b_img = b_img.cpu().numpy().astype(np.float64)
        p_hvs, p_hvs_m = psnr_hvs_hvsm(a_img, b_img)

        return p_hvs


class VIF_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='VIF')
        from IQA_pytorch import VIFs
        self.vif = VIFs(channels=1)

    def calc(self, orig, rec):
        ans = 0.0
        if 'Y' not in orig or 'Y' not in rec:
            return -100.0
        plane = 'Y'
        b = convert_range(orig[plane].unsqueeze(0).unsqueeze(0), data_range,[0, 1])
        a = convert_range(rec[plane].unsqueeze(0).unsqueeze(0), data_range,[0, 1])
        self.vif = self.vif.to(a.device)
        ans = self.vif(a, b, as_loss=False).item()

        return ans


class FSIM_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='FSIM')
        from piq import FSIM
        self.fsim = FSIM(channels=3)

    def calc(self, orig: torch.Tensor, rec: torch.Tensor):  
        ans = 0.0
        b = orig
        a = rec
        self.fsim = self.fsim.to(a.device)
        ans = self.fsim(a, b, as_loss=False).item()
        return ans


class NLPD_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='NLPD')
        from IQA_pytorch import NLPD
        self.chan = 1
        self.nlpd = NLPD(channels=self.chan)

    def calc(self, orig, rec):
        ans = 0.0
        if 'Y' not in orig or 'Y' not in rec:
            return -100.0
        if self.chan == 1:
            plane = 'Y'
            b = orig[plane].unsqueeze(0).unsqueeze(0)
            a = rec[plane].unsqueeze(0).unsqueeze(0)
        elif self.chan == 3:
            b = convert_yuvdict_to_tensor(orig, orig['Y'].device)
            a = convert_yuvdict_to_tensor(rec, rec['Y'].device)
        self.nlpd = self.nlpd.to(a.device)
        ans = self.nlpd(a, b, as_loss=False).item()

        return ans


class IWSSIM(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='IW-SSIM')
        from .IW_SSIM_PyTorch import IW_SSIM
        self.iwssim = IW_SSIM()

    def calc(self, orig, rec):
        ans = 0.0
        if 'Y' not in orig or 'Y' not in rec:
            return -100.0
        plane = 'Y'
        # IW-SSIM takes input in a range 0-255
        a = convert_range(orig[plane], data_range,[0, 255])
        b = convert_range(rec[plane],data_range,[0, 255])
        ans = self.iwssim.test(a.detach().cpu().numpy(),
                               b.detach().cpu().numpy())

        return ans.item()


class VMAF(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='VMAF')
        import platform
        if platform.system() == 'Linux':
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.linux')
        else:
            # TODO: check that
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf.exe'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.exe')

    def download(self, url, output_path):
        import requests
        r = requests.get(url, stream=True)  # , verify=False)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)

    def check(self):
        if not os.path.exists(self.OUTPUT_NAME):
            import stat
            self.download(self.URL, self.OUTPUT_NAME)
            os.chmod(self.OUTPUT_NAME, stat.S_IEXEC)

    def calc(self, org: Dict[str, torch.Tensor], dec: Dict[str, torch.Tensor]) -> float:

        import subprocess
        import tempfile
        fp_o = tempfile.NamedTemporaryFile(delete=False)
        fp_r = tempfile.NamedTemporaryFile(delete=False)
        write_yuv(org, fp_o, self.bits)
        write_yuv(dec, fp_r, self.bits)

        out_f = tempfile.NamedTemporaryFile(delete=False)
        out_f.close()

        self.check()

        args = [
            self.OUTPUT_NAME, '-r', fp_o.name, '-d', fp_r.name, '-w',
            str(org['Y'].shape[1]), '-h',
            str(org['Y'].shape[0]), '-p', '420', '-b',
            str(self.bits), '-o', out_f.name, '--json'
        ]
        subprocess.run(args,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        import json
        with open(out_f.name, 'r') as f:
            tmp = json.load(f)
        ans = tmp['frames'][0]['metrics']['vmaf']

        os.unlink(fp_o.name)
        os.unlink(fp_r.name)
        os.unlink(out_f.name)

        return ans




class LPIPS_IQA(MetricParent):
    def __init__(self,net='alex', *args, **kwargs):
        super().__init__(*args, **kwargs, name=net)
        import lpips
        if net == 'alex':
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = to_cuda(self.lpips)
        else:
            self.lpips = lpips.LPIPS(net='vgg')
            self.lpips = to_cuda(self.lpips)
            

    def calc(self, org: np.array, dec: np.array):  
        ans = 0.0
        org = img_as_float(org)
        dec = img_as_float(dec)
        
        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            self.lpips = self.lpips.cuda()
            org = org.cuda()
            dec = dec.cuda()

        ans = self.lpips(org, dec).item()  
        return ans

""" from lpips import LPIPS
class LPIPS_IQA(MetricParent):
    def __init__(self,net='alex', *args, **kwargs):
        super().__init__(*args, **kwargs, name=net)
        if net == 'alex':
            self.lpips = LPIPS(net='alex')
        else:
            self.lpips = LPIPS(net='vgg')

        self.lpips = to_cuda(self.lpips)

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        compute_weighted = weight is not None
        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        
        org = to_cuda(org)
        dec = to_cuda(dec)

        if compute_weighted:
            weight = to_cuda(weight)

        if compute_weighted:
            ans = self.lpips(org, dec, weight).item()
        else:
            ans = self.lpips(org, dec).item()  
        return ans """

from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
        
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class MSVGG(MetricParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name='msVGG')
        self.loss_weights = [1, 1, 1, 1, 1]
        self.scales  = [1, 0.5]
        self.vgg = Vgg19()
        self.pyramid = ImagePyramide(self.scales, 3)
 
    def calc(self, org: np.array, dec: np.array): 	
        org = img_as_float(org)
        dec = img_as_float(dec)
        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        if torch.cuda.is_available():
            org = org.cuda()
            dec = dec.cuda()

            self.vgg = self.vgg.cuda()
            self.pyramid = self.pyramid.cuda()
        
        pyramide_real = self.pyramid(org)
        pyramide_generated = self.pyramid(dec)
        value_total = 0.0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
            # for i, weight in enumerate(self.loss_weights):
            value = torch.abs(x_vgg[3] - y_vgg[3].detach()).mean()
            value_total += value.item()
        return value_total
		

class PIM(MetricParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name='PIM')
        self.pim = pim.load_trained("pim-5")


    def calc(self, org: np.array, dec: np.array) -> float:
        org = org/255
        dec = dec/255
        org = org[np.newaxis].astype(np.float32)
        dec = dec[np.newaxis].astype(np.float32)
        value = float(self.pim(org, dec).numpy()[-1])
        return value


class DISTS_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='DISTS')
        from DISTS_pytorch import DISTS
        self.dist = DISTS()
        self.dist = to_cuda(self.dist)
        
    def calc(self, org: torch.Tensor, dec: torch.Tensor):  
        ans = 0.0

        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        if torch.cuda.is_available():
            org = org.cuda()
            dec = dec.cuda()
        ans = self.dist(org, dec).item()
        return ans        

class Metrics:
    org_yuv, org_rgb = {}, None
    dec_yuv, dec_rgb = {}, None

    def _mse(self, org, dec):
        return np.mean((org-dec)**2)

    def _psnr(self):	
        # ref = [frame[:,:,0] for frame in self.org_rgb]
        # dec = [frame[:,:,0] for frame in self.dec_rgb]
        s_psnr = []
        for idx in range(len(self.dec_rgb)):			
            mse_val_r = self._mse(self.org_rgb[idx][:,:,0],self.dec_rgb[idx][:,:,0])
            mse_val_g = self._mse(self.org_rgb[idx][:,:,1],self.dec_rgb[idx][:,:,1])
            mse_val_b = self._mse(self.org_rgb[idx][:,:,2],self.dec_rgb[idx][:,:,2])
            psnr_val_r = 10*np.log10(255**2/mse_val_r)
            psnr_val_g = 10*np.log10(255**2/mse_val_g)
            psnr_val_b = 10*np.log10(255**2/mse_val_b)
            s_psnr.append((psnr_val_r+psnr_val_g+psnr_val_b)/3)
        return s_psnr

    def _psnr_hvsm(self):
        from psnr_hvsm import psnr_hvs_hvsm
        ref = [convert_range(frame, [0,1], [0, 1]) for frame in self.org_rgb]
        dec = [convert_range(frame, [0,1], [0, 1]) for frame in self.dec_rgb]
        s_psnr_hvs = []
        for idx in range(len(dec)):	
            # f_ref = convert_range(ref[idx], [0,1], [0, 1])
            # f_dec = convert_range(dec[idx], [0,1], [0, 1])
            psnr_hvs_val_r,_ = psnr_hvs_hvsm(ref[idx][:,:,0], dec[idx][:,:,0])
            psnr_hvs_val_g,_ = psnr_hvs_hvsm(ref[idx][:,:,1], dec[idx][:,:,1])
            psnr_hvs_val_b,_ = psnr_hvs_hvsm(ref[idx][:,:,2], dec[idx][:,:,2])
            s_psnr_hvs.append(5*(psnr_hvs_val_r+psnr_hvs_val_g+psnr_hvs_val_b)/3)
        return s_psnr_hvs

    def _fsim_iqa(self):
        from piq import fsim
        s_fsim = []
        for idx in range(len(self.dec_rgb)):		
            org_frame = img_as_float(self.org_rgb[idx])
            dec_frame = img_as_float(self.dec_rgb[idx])
            org = torch.tensor(org_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            dec = torch.tensor(dec_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            
            ans = fsim(org, dec).item()
            s_fsim.append(ans)
        return s_fsim

    def _nlpd_iqa(self):
        from IQA_pytorch import NLPD
        nlpd = NLPD(channels=3)
        s_nlpd = []
        for idx in range(len(self.dec_rgb)):		
            org_frame = img_as_float(self.org_rgb[idx])
            dec_frame = img_as_float(self.dec_rgb[idx])
            org = torch.tensor(org_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            dec = torch.tensor(dec_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            
            ans = nlpd(org, dec, as_loss=False).item()
            s_nlpd.append(ans)
        return s_nlpd

    def _iw_ssim(self):
        from .IW_SSIM_PyTorch import IW_SSIM
        iwssim = IW_SSIM()
        s_iw_ssim = []
        for idx in range(len(self.dec_rgb)):
            ans_r = iwssim.test(self.org_rgb[idx][:,:,0],self.dec_rgb[idx][:,:,0])
            ans_g = iwssim.test(self.org_rgb[idx][:,:,1],self.dec_rgb[idx][:,:,1])
            ans_b = iwssim.test(self.org_rgb[idx][:,:,2],self.dec_rgb[idx][:,:,2])
            s_iw_ssim.append(float((ans_r+ans_g+ans_b)/3))
        return s_iw_ssim


    def _ms_ssim_iqa(self):
        from IQA_pytorch.MS_SSIM import MS_SSIM
        ms_ssim = MS_SSIM(channels=3)

        s_ms_ssim = []
        for idx in range(len(self.dec_rgb)):
            org_frame = torch.tensor(img_as_float(self.org_rgb[idx]), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            dec_frame = torch.tensor(img_as_float(self.dec_rgb[idx]), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

            ans = ms_ssim(org_frame,dec_frame, as_loss=False).item()
            s_ms_ssim.append(float(ans))
        return s_ms_ssim


    def _vif(self):
        from IQA_pytorch import VIFs
        vif = VIFs(channels=3)
        s_vif = []
        for idx in range(len(self.dec_rgb)):
            org_frame = torch.tensor(img_as_float(self.org_rgb[idx]), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            dec_frame = torch.tensor(img_as_float(self.dec_rgb[idx]), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

            ans = vif(org_frame,dec_frame, as_loss=False).item()
            s_vif.append(float(ans))
        return s_vif



    def compute_metrics(self,org: List[np.array], dec: List[np.array],
                        metrics = ['psnr','psnr_hvs','ms_ssim','vif','nlpd','iw_ssim','vmaf'])-> Dict[str, List[float]]:
        self.org_rgb = org
        self.dec_rgb = dec
        #preprocess the data into YUV
        for idx, frame in enumerate(org):
            frame_yuv = load_image_array(frame)
            dec_yuv = load_image_array(dec[idx])

            self.org_yuv[idx] = frame_yuv
            self.dec_yuv[idx] = dec_yuv
        
        all_metrics = {}
        #compute specified metrics
        if 'psnr' in metrics:
            psnr = PSNRMetric()
            s_psnr = []
            for idx in self.dec_yuv:
                val = psnr.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_psnr.append(val)
            all_metrics['psnr'] = s_psnr

            
        if 'psnr_hvs' in metrics:
            # all_metrics['psnr_hvs'] = np.mean(self._psnr_hvsm())
            psnr_hvs = PSNR_HVS()
            s_psnr_hvs = []
            for idx in self.dec_yuv:
                val = psnr_hvs.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_psnr_hvs.append(val)
            all_metrics['psnr_hvs'] = s_psnr_hvs

        if 'lpips' in metrics:
            # all_metrics['psnr_hvs'] = np.mean(self._psnr_hvsm())
            with contextlib.redirect_stdout(None):
                lpips = LPIPS_IQA(net='alex')
            s_lpips = []
            for idx, dec_frame in enumerate(self.dec_rgb):
                val = lpips.calc(self.org_rgb[idx], dec_frame)
                s_lpips.append(val)
            all_metrics['lpips'] = s_lpips

        if 'pim' in metrics:
            # all_metrics['psnr_hvs'] = np.mean(self._psnr_hvsm())
            pim = PIM()
            s_pim = []
            for idx, dec_frame in enumerate(self.dec_rgb):
                val = pim.calc(self.org_rgb[idx], dec_frame)
                s_pim.append(val)
            all_metrics['pim'] = s_pim

        if 'lpips_vgg' in metrics:
            # all_metrics['psnr_hvs'] = np.mean(self._psnr_hvsm())
            with contextlib.redirect_stdout(None):
                lpips = LPIPS_IQA(net='vgg')
            s_lpips = []
            for idx, dec_frame in enumerate(self.dec_rgb):
                val = lpips.calc(self.org_rgb[idx], dec_frame)
                s_lpips.append(val)
            all_metrics['lpips_vgg'] = s_lpips

        if 'msVGG' in metrics:
            # all_metrics['psnr_hvs'] = np.mean(self._psnr_hvsm())
            msVGG = MSVGG()
            s_msVGG = []
            for idx, dec_frame in enumerate(self.dec_rgb):
                val = msVGG.calc(self.org_rgb[idx], dec_frame)
                s_msVGG.append(val)
            all_metrics['msVGG'] = s_msVGG

        if 'fsim' in metrics:
            all_metrics['fsim'] = self._fsim_iqa()

        if 'nlpd' in metrics:
            # all_metrics['nlpd'] = np.mean(self._nlpd_iqa())
            nlpd = NLPD_IQA()
            s_nlpd   = []
            for idx in self.dec_yuv:
                val =   nlpd.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_nlpd.append(val)
            all_metrics['nlpd'] = s_nlpd
        
        if 'iw_ssim' in metrics:
            # all_metrics['iw_ssim'] = np.mean(self._iw_ssim())
            iw_ssim = IWSSIM()
            s_iw_ssim   = []
            for idx in self.dec_yuv:
                val =   iw_ssim.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_iw_ssim.append(val)
            all_metrics['iw_ssim'] = s_iw_ssim

        if 'ms_ssim' in metrics:
            ms_ssim = MSSSIM_IQA()
            s_ms_ssim   = []
            for idx in self.dec_yuv:
                val =   ms_ssim.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_ms_ssim.append(val)
            all_metrics['ms_ssim'] = s_ms_ssim

        if 'ms_ssim_pytorch' in metrics:
            ms_ssim = MSSSIMTorch()
            s_ms_ssim   = []
            for idx in self.dec_yuv:
                val =   ms_ssim.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_ms_ssim.append(val)
            all_metrics['ms_ssim_pytorch'] = s_ms_ssim

        if 'vif' in metrics:
            vif = VIF_IQA()
            s_vif   = []
            for idx in self.dec_yuv:
                val =   vif.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_vif.append(val)
            all_metrics['vif'] = s_vif

        if 'vmaf' in metrics:
            vmaf = VMAF()
            s_vmaf   = []
            for idx in self.dec_yuv:
                val =   vmaf.calc(self.org_yuv[idx], self.dec_yuv[idx])
                s_vmaf.append(val)
            all_metrics['vmaf'] = s_vmaf

        if 'dists' in metrics:
            dists = DISTS_IQA()
            s_dists = []
            
            for idx, dec_frame in enumerate(self.dec_rgb):
                val = dists.calc(self.org_rgb[idx], dec_frame)
                s_dists.append(val)
            
            all_metrics['dists'] = s_dists
            #all_metrics['dists'] =  np.mean(s_dists)        

        return all_metrics
        
def save_unique_plot(filename, folder='imgs/plots'):
    # フォルダが存在しない場合は作成
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 拡張子を分離
    base, ext = os.path.splitext(filename)
    
    # ファイル名が重複しないようにインクリメント
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    # 完全なパスを作成
    save_path = os.path.join(folder, new_filename)
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    tgt = 1
    # org, org_rgb = load_image(f"imgs/org/{tgt}.png")
    # dec, dec_rgb = load_image(f"imgs/dec/{tgt}.png")
    metrics = Metrics()
    start_index=0
    end_index=120
    data_num=47
    gop_size=10
    frame_indices = list(range(1,end_index+2))

    psnr_sum_hdac = np.zeros(len(frame_indices))
    psnr_sum_r2hdac = np.zeros(len(frame_indices))
    lpips_sum_hdac = np.zeros(len(frame_indices))
    lpips_sum_r2hdac = np.zeros(len(frame_indices))
    dists_sum_hdac = np.zeros(len(frame_indices))
    dists_sum_r2hdac = np.zeros(len(frame_indices))

    # rdac_path='/home/21x3109_takahashi/animation-based-codecs/results/rdac/RDAC_data22_15_0.05_30.mp4'
    for data in tqdm(range(1,data_num+1)):
        org = imageio.mimread(f'/home/21x3109_takahashi/hdac/BFDAC_testset/data{data}.mp4', memtest=False)[start_index:end_index+1]
        hdac=imageio.mimread(f'/home/21x3109_takahashi/hdac/results/HDAC/video/HDAC_data{data}_{gop_size}_30.mp4', memtest=False)[start_index:end_index+1]
        hdac_metrics = metrics.compute_metrics(org, hdac, metrics=['psnr','lpips','dists'])

        r2hdac=imageio.mimread(f'/home/21x3109_takahashi/hdac/results/R2HDAC/video/R2HDAC_data{data}_{gop_size}_30.mp4', memtest=False)[start_index:end_index+1]
        r2hdac_metrics = metrics.compute_metrics(org, r2hdac, metrics=['psnr','lpips','dists'])

        psnr_sum_hdac += np.array(hdac_metrics['psnr'])
        psnr_sum_r2hdac += np.array(r2hdac_metrics['psnr'])
        lpips_sum_hdac += np.array(hdac_metrics['lpips'])
        lpips_sum_r2hdac += np.array(r2hdac_metrics['lpips'])
        dists_sum_hdac += np.array(hdac_metrics['dists'])
        dists_sum_r2hdac += np.array(r2hdac_metrics['dists'])

        

    # 平均を計算
    psnr_avg_hdac = psnr_sum_hdac / data_num
    psnr_avg_r2hdac = psnr_sum_r2hdac / data_num
    lpips_avg_hdac = lpips_sum_hdac / data_num
    lpips_avg_r2hdac = lpips_sum_r2hdac / data_num
    dists_avg_hdac = dists_sum_hdac / data_num
    dists_avg_r2hdac = dists_sum_r2hdac / data_num

    # プロットを作成
    plt.figure()
    plt.plot(frame_indices, psnr_avg_hdac, linestyle='-', label='HDAC')
    plt.plot(frame_indices, psnr_avg_r2hdac, linestyle='-', label='Ours')
    plt.xlabel('frame index')
    plt.ylabel('psnr')
    plt.legend()
    save_unique_plot(f'average_psnr_plot_gop{gop_size}.png')

    plt.figure()
    plt.plot(frame_indices, lpips_avg_hdac, linestyle='-', label='HDAC')
    plt.plot(frame_indices, lpips_avg_r2hdac, linestyle='-', label='Ours')
    plt.xlabel('frame index')
    plt.ylabel('lpips')
    plt.legend()
    save_unique_plot(f'average_lpips_plot_gop{gop_size}.png')

    plt.figure()
    plt.plot(frame_indices, dists_avg_hdac, linestyle='-', label='HDAC')
    plt.plot(frame_indices, dists_avg_r2hdac, linestyle='-', label='Ours')
    plt.xlabel('frame index')
    plt.ylabel('dists')
    plt.legend()
    save_unique_plot(f'average_dists_plot_gop{gop_size}.png')

    plt.show()

        # plt.figure()
        # plt.plot(frame_indices, hdac_metrics['lpips'], linestyle='-', label='HDAC')
        # plt.plot(frame_indices, r2hdac_metrics['lpips'], linestyle='-', label='Ours')
        # plt.xlabel('frame index')
        # plt.ylabel('lpips')
        # plt.title('LPIPS over frames')
        # plt.legend()
        # save_unique_plot(f'data{data}_lpips_plot.png')

        # plt.figure()
        # plt.plot(frame_indices, hdac_metrics['dists'], linestyle='-', label='HDAC')
        # plt.plot(frame_indices, r2hdac_metrics['dists'], linestyle='-', label='Ours')
        # plt.xlabel('frame index')
        # plt.ylabel('dists')
        # plt.title('DISTS over frames')
        # plt.legend()
        # save_unique_plot(f'data{data}_dists_plot.png')
        
        # plt.show()

    # dac=imageio.mimread(f'/home/21x3109_takahashi/hdac/results/DAC/video/DAC_data{data_num}_15_30.mp4', memtest=False)[start_index:end_index+1]
    # all_metrics = metrics.compute_metrics(org, dac, metrics=['psnr','lpips','dists'])
    # print('DAC')
    # print('psnr: ',all_metrics['psnr'])
    # print('avrage psnr: ',np.mean(all_metrics['psnr']))
    # print('lpips: ',all_metrics['lpips'])
    # print('avrage lpips: ',np.mean(all_metrics['lpips']))
    # print('dists: ',all_metrics['dists'])
    # print('avrage dists: ',np.mean(all_metrics['dists']))
    # print('---------------------------------')

    # hdac=imageio.mimread(f'/home/21x3109_takahashi/hdac/results/HDAC/video/HDAC_data{data_num}_15_30.mp4', memtest=False)[start_index:end_index+1]
    # hdac_metrics = metrics.compute_metrics(org, hdac, metrics=['psnr','lpips','dists'])
    # print('HDAC')
    # print('psnr: ',hdac_metrics['psnr'])
    # print('avrage psnr: ',np.mean(hdac_metrics['psnr']))
    # print('lpips: ',hdac_metrics['lpips'])
    # print('avrage lpips: ',np.mean(hdac_metrics['lpips']))
    # print('dists: ',hdac_metrics['dists'])
    # print('avrage dists: ',np.mean(hdac_metrics['dists']))
    # print('---------------------------------')

    # rdac=imageio.mimread(rdac_path, memtest=False)[start_index:end_index+1]
    # all_metrics = metrics.compute_metrics(org, rdac, metrics=['psnr','lpips','dists'])
    # print('RDAC')
    # print('psnr: ',all_metrics['psnr'])
    # print('avrage psnr: ',np.mean(all_metrics['psnr']))
    # print('lpips: ',all_metrics['lpips'])
    # print('avrage lpips: ',np.mean(all_metrics['lpips']))
    # print('dists: ',all_metrics['dists'])
    # print('avrage dists: ',np.mean(all_metrics['dists']))
    # print('---------------------------------')

    # r2hdac=imageio.mimread(f'/home/21x3109_takahashi/hdac/results/R2HDAC/video/R2HDAC_data{data_num}_15_30.mp4', memtest=False)[start_index:end_index+1]
    # r2hdac_metrics = metrics.compute_metrics(org, r2hdac, metrics=['psnr','lpips','dists'])
    # print('R2HDAC')
    # print('psnr: ',r2hdac_metrics['psnr'])
    # print('avrage psnr: ',np.mean(r2hdac_metrics['psnr']))
    # print('lpips: ',r2hdac_metrics['lpips'])
    # print('avrage lpips: ',np.mean(r2hdac_metrics['lpips']))
    # print('dists: ',r2hdac_metrics['dists'])
    # print('avrage dists: ',np.mean(r2hdac_metrics['dists']))
    # print('---------------------------------')

    # plt.figure()
    # plt.plot(frame_indices, hdac_metrics['lpips'], linestyle='-', label='HDAC')
    # plt.plot(frame_indices, r2hdac_metrics['lpips'], linestyle='-', label='Ours')
    # plt.xlabel('frame index')
    # plt.ylabel('lpips')
    # plt.title('LPIPS over frames')
    # plt.legend()
    # save_unique_plot(f'data{data_num}_lpips_plot.png')

    # plt.figure()
    # plt.plot(frame_indices, hdac_metrics['dists'], linestyle='-', label='HDAC')
    # plt.plot(frame_indices, r2hdac_metrics['dists'], linestyle='-', label='Ours')
    # plt.xlabel('frame index')
    # plt.ylabel('dists')
    # plt.title('DISTS over frames')
    # plt.legend()
    # save_unique_plot(f'data{data_num}_dists_plot.png')
    
    # plt.show()


    # hevc=imageio.mimread('/home/21x3109_takahashi/hdac/results/HEVC/video/HEVC_data1_5_30.mp4', memtest=False)[:120]
    # all_metrics = metrics.compute_metrics(org, hevc, metrics=['lpips','dists'])
    # print('HEVC')
    # print('lpips: ',all_metrics['lpips'][index])
    # print('dists: ',all_metrics['dists'][index])
    # print('---------------------------------')

    # vvc=imageio.mimread('/home/21x3109_takahashi/hdac/results/VVC/video/VVC_data1_5_30.mp4', memtest=False)[:120]
    # all_metrics = metrics.compute_metrics(org, vvc, metrics=['lpips','dists'])
    # print('VVC')
    # print('lpips: ',all_metrics['lpips'][index])
    # print('dists: ',all_metrics['dists'][index])
    # print('---------------------------------')

    # dec = imageio.mimread('videos/dec.mp4', memtest=False)

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/org/original_img_14.png')
    # org_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/DAC_img_14.png')
    # dac_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, dac_image, metrics=['lpips','dists'])
    # print('DAC')
    # print(all_metrics)
    # print('---------------------------------')

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/HDAC_img_14.png')
    # hdac_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, hdac_image, metrics=['lpips','dists'])
    # print('HDAC')
    # print(all_metrics)
    # print('---------------------------------')

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/RDAC_img_14.png')
    # rdac_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, rdac_image, metrics=['lpips','dists'])
    # print('RDAC')
    # print(all_metrics)
    # print('---------------------------------')

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/R2HDAC_img_14.png')
    # r2hdac_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, r2hdac_image, metrics=['lpips','dists'])
    # print('R2HDAC')
    # print(all_metrics)
    # print('---------------------------------')

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/HEVC_img_14.png')
    # hevc_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, hevc_image, metrics=['lpips','dists'])
    # print('HEVC')
    # print(all_metrics)
    # print('---------------------------------')

    # image_bgr = cv2.imread('/home/21x3109_takahashi/hdac/modules/metrics/imgs/dec/VVC_img_14.png')
    # vvc_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # all_metrics = metrics.compute_metrics(org_image, vvc_image, metrics=['lpips','dists'])
    # print('VVC')
    # print(all_metrics)
    # print('---------------------------------')
    
    #Computing PSNR
    # psnr = PSNRMetric()
    # psnr_out = psnr.calc(org, dec)
    # print(psnr_out)

    # ms_ssim_torch = MSSSIMTorch()
    # m_ssim_out = ms_ssim_torch.calc(org, dec)
    # print(m_ssim_out)

    # ms_ssim_iqa = MSSSIM_IQA()
    # out = ms_ssim_iqa.calc(org, dec)
    # print(out)


    # psnr_hvs = PSNR_HVS()
    # out = psnr_hvs.calc(org, dec)
    # print(out)

    # vif = VIF_IQA()
    # out = vif.calc(org, dec)
    # print(out)


    ##TODO: Get compatible package
    # fsim = FSIM_IQA()
    # out = fsim.calc(org_rgb, dec_rgb)
    # print(out)

    # nlpd = NLPD_IQA()
    # out = nlpd.calc(org, dec)
    # print(out)


    # iw_ssim = IWSSIM()
    # out = iw_ssim.calc(org, dec)
    # print(out)

    # vmaf = VMAF()
    # out = vmaf.calc(org, dec)
    # print(out)
