import subprocess
import os
import cv2
import numpy as np
import tempfile

def vtm_comp(img_rgb, qp, fps, vvc_enc, vvc_dec, sample_cfg, vtm_cfg):
    height, width = img_rgb.shape[:2]

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"width/height must be even for YUV420: got {width}x{height}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        yuv_path     = os.path.join(tmp_dir, 'in_source.yuv')
        bin_path     = os.path.join(tmp_dir, "str.bin")
        out_yuv_path = os.path.join(tmp_dir, 'out_source.yuv')
        img_cfg_path = os.path.join(tmp_dir, 'img.cfg')

        yuv420 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV_I420)  # (H*3/2, W), 1ch
        yuv420.tofile(yuv_path)

        n_frames = 1
        with open(sample_cfg, 'r') as f:
            file_contents = f.read()

        file_contents = file_contents.replace('test_yuv', str(yuv_path))
        file_contents = file_contents.replace('inputW', str(width))
        file_contents = file_contents.replace('inputH', str(height))
        file_contents = file_contents.replace('inputNrFrames', str(n_frames))
        file_contents = file_contents.replace('inputFPS', str(fps))

        with open(img_cfg_path, 'w+') as f:
            f.write(file_contents)

        subprocess.run(
            [vvc_enc, '-c', vtm_cfg, '-c', img_cfg_path,
             '--OutputBitDepth=8', '--InternalBitDepth=8',
             '-q', str(qp), '-b', bin_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        vvc_bytes = os.path.getsize(bin_path)

        subprocess.run(
            [vvc_dec, "-b", bin_path, "-o", out_yuv_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        frame_size = height * width * 3 // 2  
        buf = np.fromfile(out_yuv_path, dtype=np.uint8)
        if buf.size != frame_size:
            raise ValueError(f"Decoded YUV size mismatch: expected {frame_size}, got {buf.size}")

        yuv_dec = buf.reshape((height * 3 // 2, width))
        vvc_rgb = cv2.cvtColor(yuv_dec, cv2.COLOR_YUV2RGB_I420)

    return vvc_rgb, vvc_bytes * 8
