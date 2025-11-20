
import matplotlib
matplotlib.use('Agg')

import torch
import yaml
from argparse import ArgumentParser
from modules import KPD
from compression import compress

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='BiLFAC', type=str,) 
    parser.add_argument("--config", default='./config/bilfac.yaml', type=str, help="path to config")
    parser.add_argument("--mode", default="compress", choices=["train","compress"])
    parser.add_argument("--checkpoint", default='./ckpt/bilfac.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--data_dir", default="", type=str)

    parser.add_argument("--vvc_encoder", required=True, default="", type=str)
    parser.add_argument("--vvc_decoder", required=True, default="", type=str)
    parser.add_argument("--sample_cfg", default="./sample.cfg", type=str)
    parser.add_argument("--vtm_cfg", default="./encoder_intra_vtm.cfg", type=str)
    
    #coding params
    parser.add_argument("--keyframe_thresh", type=int, default=33, help="Threshold for keyframe selection")
    parser.add_argument("--n_frames", default=120, type=int, help="Number of frames to compress")
    parser.add_argument("--fps", default=30, type=int, help="")
    parser.add_argument("--gop_size", type=int, default=32, help="GOP size for compression")
    parser.add_argument("--source_qp_lst", nargs='+', type=int, default=[30], help="List of QP values for the reference frames")
    parser.add_argument("--driving_qp", default=45, type=int,help="Compression QP for the driving frames")
    parser.add_argument("--output_dir", default='./results', type=str,help="Output directory for video and metric results")

    parser.add_argument("--face_sr", action='store_true', help='Face super-resolution (Optional).')

    opt = parser.parse_args()
    
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #import keypoint detector
    kp_detector = KPD(**config['model_params']['common_params'],**config['model_params']['kp_detector_params'], training= opt.mode == 'train')
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    compress(config, kp_detector, **vars(opt))