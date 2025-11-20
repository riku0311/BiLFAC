import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from skimage import img_as_float32
from .utils.bitstream import KeypointCompress
from .utils.vtm_comp import vtm_comp
from .utils.x265_comp import x265_comp
from .utils.util import split_into_gop, to_cuda, to_tensor, detach_frame, downsample_frames, upsample_frames                 
import cv2
from .face_sr.face_enhancer import enhancer_list

class DAC(nn.Module):
    """
    Base Deep Animation Coding:: 
        ::Inputs - Group of Pictures
        ::Outputs - 
    """
    def __init__(self, kp_detector=None, generator=None, config=None, adaptive=True, threshold=30,source_quality=6):
        super(DAC, self).__init__()
        #Pretrained models
        self.kp_detector = kp_detector
        self.generator = generator

        ## Coding tools::
        # Keypoint compression
        self.kp_compressor = KeypointCompress(num_kp=config['model_params']['common_params']['num_kp'])
        # self.kp_compressor = KeypointCompress(num_kp=10)
        
        #output visualization
        #self.visualizer = Visualizer(**config['visualizer_params'])

        #coding params
        self.adaptive =adaptive 
        self.threshold = threshold

        self.relative_jacobian = False
        self.relative_movement = True
        self.adapt_movement_scale = True
        # self.bitstream = {}
    
    def forward(self, x):
        kp_driving = self.kp_detector(x['driving'])
        kp_source = self.kp_detector(x['source'])
        generated = self.generator(x['source'], kp_source= kp_source,kp_driving =kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        return generated
    
    def compress(self, sequence, **kwargs):
        gop_size=kwargs['gop_size']
        source_qp=kwargs['source_qp']
        fps=kwargs['fps']

        decoded_video = []
        total_bits = 0
        gop_lst = split_into_gop(sequence, gop_size)
        for gop in gop_lst:
            first_I_frame = gop[0]

            # compress source frame
            source_img, src_bits = vtm_comp(first_I_frame, source_qp, fps, kwargs['vvc_encoder'], kwargs['vvc_decoder'],
                                            kwargs['sample_cfg'], kwargs['vtm_cfg'])
            source = torch.tensor(np.array([img_as_float32(source_img).transpose(2,0,1)]), dtype=torch.float32)
            source=to_cuda(source)
            kp_source = self.kp_detector(source)
            decoded_video.append(source_img)
            
            # drive frames
            driving_frames = gop[1:]
            n_frames = len(driving_frames)
            driving_frames = to_tensor(driving_frames, n_frames=n_frames)

            kp_lst = []
            for frame in driving_frames:
                d_kp = self.kp_detector(to_cuda(frame))
                kp_lst.append(d_kp)

            for idx in tqdm(range(n_frames)):
                kp_driving, _ = self.kp_compressor.encode_kp(kp_source, kp_lst[idx])
                kp_driving['value'] = to_cuda(kp_driving['value'])
                generated = self.generator(source, kp_source= kp_source, kp_driving =kp_driving)
                decoded_video.append(detach_frame(generated['prediction']))
            kp_bits, __ = self.kp_compressor.get_bitstream()

            total_bits += (src_bits+kp_bits)
            self.kp_compressor.reset()
        return decoded_video, total_bits

class BiLFAC(DAC):
    def __init__(self, kp_detector=None, generator=None, config=None):
        super().__init__(kp_detector, generator, config)
    
    def forward(self, x):
        kp_driving = self.kp_detector(x['driving'])
        kp_source = self.kp_detector(x['source'])
        generated = self.generator(x['source'], base_layer=x['hevc'],kp_source= kp_source,kp_driving =kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        return generated

    def compress(self, sequence, **kwargs):
        gop_size=kwargs['gop_size']
        source_qp=kwargs['source_qp']
        driving_qp=kwargs['driving_qp']
        fps=kwargs['fps']
        face_sr = kwargs['face_sr']

        down_flag=True
        decoded_video=[]
        total_bits = 0
        gop_lst = split_into_gop(sequence, gop_size)

        first_source = first_kp_source = None
        last_source = last_kp_source = None
        for gop_idx, gop in enumerate(gop_lst):
            #===== key frame compression =====
            src_bits = 0
            if gop_idx == 0:
                first_I_frame = gop[0]
                first_I_frame, first_I_bits = vtm_comp(first_I_frame, source_qp, fps, kwargs['vvc_encoder'], kwargs['vvc_decoder'],
                                            kwargs['sample_cfg'], kwargs['vtm_cfg'])
                src_bits += first_I_bits
                decoded_video.append(first_I_frame)
                
                first_source = torch.tensor(np.array([img_as_float32(first_I_frame).transpose(2,0,1)]), dtype=torch.float32)
                first_source = to_cuda(first_source)
                first_kp_source = self.kp_detector(first_source)
            else:
                first_I_frame = last_I_frame
                first_source = last_source
                first_kp_source = last_kp_source
            
            if gop_idx != len(gop_lst) - 1:
                last_I_frame = gop_lst[gop_idx+1][0]
                last_I_frame, last_I_bits = vtm_comp(last_I_frame, source_qp, fps, kwargs['vvc_encoder'], kwargs['vvc_decoder'],
                                            kwargs['sample_cfg'], kwargs['vtm_cfg'])
                src_bits += last_I_bits

                last_source = torch.tensor(np.array([img_as_float32(last_I_frame).transpose(2,0,1)]), dtype=torch.float32)
                last_source = to_cuda(last_source)
                last_kp_source = self.kp_detector(last_source)

            frames = gop[1:]
            intermediate_frames = downsample_frames(frames, factor=2) if down_flag else frames
            
            #===== auxiliary frame compression =====
            vvc_frames, vvc_bits = x265_comp(intermediate_frames, intermediate_frames[0].shape[1], intermediate_frames[0].shape[0], fps, driving_qp)

            if down_flag:
                if face_sr:
                    #===== upsample with GFPGAN =====
                    vvc_frames = enhancer_list(vvc_frames, method='gfpgan', bg_upsampler=None)
                else:
                    vvc_frames = upsample_frames(vvc_frames, (sequence[0].shape[1], sequence[0].shape[0]))

            #===== select source frame for animation =====
            if gop_idx != len(gop_lst) - 1:
                ref_flags = []
                for idx in range(len(vvc_frames)):
                    psnr1 = cv2.PSNR(first_I_frame, vvc_frames[idx])
                    psnr2 = cv2.PSNR(last_I_frame, vvc_frames[idx])
                    if psnr1 <= psnr2:
                        ref_flags.append(0)
                    else:
                        ref_flags.append(1)
            else:
                ref_flags = [1] * len(vvc_frames)
          
            n_frames = len(vvc_frames)
            aux_frames = to_tensor(vvc_frames, n_frames=n_frames)
            driving_frames = to_tensor(gop[1:], n_frames=n_frames)

            for idx in tqdm(range(n_frames)):
                driving = to_cuda(driving_frames[idx])
                aux_frame = to_cuda(aux_frames[idx])

                if ref_flags[idx] == 0:
                    current_source = last_source
                    current_kp_source = last_kp_source
                else:
                    current_source = first_source
                    current_kp_source = first_kp_source

                kp_target = self.kp_detector(driving)
                kp_driving, _ = self.kp_compressor.encode_kp(current_kp_source, kp_target)
                kp_driving['value'] = to_cuda(kp_driving['value'])
                generated = self.generator(current_source, base_layer=aux_frame, kp_source=current_kp_source, kp_driving=kp_driving)
                generated_img = detach_frame(generated['prediction'])
                decoded_video.append(generated_img)

            if gop_idx != len(gop_lst) - 1:
                decoded_video.append(last_I_frame)
            
            kp_bits, _ = self.kp_compressor.get_bitstream()
            self.kp_compressor.reset()
            total_bits += (src_bits+kp_bits+vvc_bits)
        return decoded_video, total_bits