import os, shutil, subprocess, tempfile, cv2
import numpy as np

def run(cmd, name):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[{name}] failed:\n{e.stderr}") from e

def yuv420_to_frames(path, w, h):
    fsz = w * h * 3 // 2
    n = os.path.getsize(path) // fsz
    mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(n, h * 3 // 2, w))
    return [cv2.cvtColor(mm[i], cv2.COLOR_YUV2RGB_I420) for i in range(n)]

def x265_comp(gop, width, height, fps, qp, preset="veryfast", tune="zerolatency"):
    tmp = tempfile.mkdtemp(prefix="x265_")
    input_yuv = os.path.join(tmp, "in.yuv")
    hevc = os.path.join(tmp, "out.bin")
    dec_yuv = os.path.join(tmp, "dec.yuv")

    try:
        with open(input_yuv, 'wb') as f:
            for frame in gop:
                yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
                f.write(yuv.tobytes())

        cmd1 = [
            "ffmpeg", "-hide_banner", "-loglevel", "quiet", "-y",
            "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", input_yuv,
            "-c:v", "libx265", "-preset", preset, "-tune", tune,
            "-x265-params", f"qp={int(qp)}:keyint={int(len(gop))}:verbose=1",
            "-f", "hevc", hevc
        ]
        run(cmd1, "x265 encode")

        cmd2 = [
            "ffmpeg", "-hide_banner", "-loglevel", "quiet", "-y",
            "-i", hevc,
            "-pix_fmt", "yuv420p",
            "-f", "rawvideo",
            dec_yuv,
        ]
        run(cmd2, "ffmpeg decode")

        decoded_frames = yuv420_to_frames(dec_yuv, width, height)
        bitstream_size = os.path.getsize(hevc) * 8

        return decoded_frames, bitstream_size

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
