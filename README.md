# Bidirectional Learned Facial Animation Codec for Low Bitrate Talking Head Videos (DCC'25)

## 🔧️ Framework
![Model Architecture](Architecture.png)

## ⚙️ Installation

Create conda environment:

```bash
  conda create -n bilfac python=3.8
  conda activate bilfac
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
```

Please install the VVC reference software (VTM) and FFmpeg by following their official installation guides:

- VTM (VVCSoftware_VTM): https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM  
- FFmpeg: https://www.ffmpeg.org/download.html

Please download the pretrained weights from the link below and place the file inside the `./ckpt` directory:

https://drive.google.com/file/d/1AWQzCKeRCfsDdihFIFCxA0VBUFUzuB4O/view?usp=sharing

## 🚀 Inference

```bash
  python run.py --data_dir <path_to_data> --vvc_encoder <path_to_VTM_EncoderApp> --vvc_decoder <path_to_VTM_DecoderApp> --n_frames 120 --gop_size 30 --source_qp_lst 30 --driving_qp 45 --output_dir ./results
```
