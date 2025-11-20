import os
import torch
import traceback
from modules import BiLFAC
from modules import GeneratorBiLFAC
from utils.io import is_video_file, compute_bitrate, load_video, images2video, write_results

def compress(config_dict, kp_detector, **kwargs):
    data_dir   = kwargs['data_dir']
    model      = kwargs['model']
    output_dir = kwargs['output_dir']
    fps        = kwargs['fps']
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)

    # ---- build generator ----
    generator = GeneratorBiLFAC(**config_dict['model_params']['common_params'], **config_dict['model_params']['generator_params'])

    if torch.cuda.is_available():
        generator.to(0)

    # ---- load weights ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(kwargs['checkpoint'], map_location=device)
    generator.load_state_dict(ckpt['generator'])
    kp_detector.load_state_dict(ckpt['kp_detector'])

    # ---- instantiate codec ----
    bilfac = BiLFAC(kp_detector=kp_detector, generator=generator, config=config_dict)

    if torch.cuda.is_available():
        bilfac = bilfac.cuda()
    bilfac.eval()

    # ---- collect inputs ----
    input_paths = []
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            p = os.path.join(data_dir, f)
            if is_video_file(p):
                input_paths.append(p)
    elif os.path.isfile(data_dir) and is_video_file(data_dir):
        input_paths.append(data_dir)
    else:
        raise ValueError(f"Invalid input: {data_dir} is not a video file or directory.")

    for process_index, video_path in enumerate(input_paths):
        s_name = os.path.splitext(os.path.basename(video_path))[0]
        original_frames = load_video(video_path, n_frames=kwargs['n_frames'])

        for source_qp in kwargs['source_qp_lst']:
            gop_size = kwargs['gop_size']
            driving_qp = kwargs['driving_qp']

            print(f"{process_index}  Processing video: {video_path}, "
                  f"GOP size: {gop_size}, Source QP: {source_qp}, Driving QP: {driving_qp}")

            video_output_dir = os.path.join(output_dir, 'videos')
            output_video_path = os.path.join(video_output_dir, f"{gop_size}_{source_qp}_{s_name}.mp4")

            coding_params = dict(kwargs)
            coding_params.update({"source_qp": source_qp, "fps": fps, "file_name": s_name})

            try:
                out_frames, total_bits = bilfac.compress(original_frames, **coding_params)
            except Exception as e:
                print(f"Error compressing {s_name} (GOP={gop_size}, S_QP={source_qp}): {e}")
                traceback.print_exc()
                continue

            if out_frames is None or total_bits is None:
                print(f"Error: Could not compress video {s_name}")
                continue

            bitrate = compute_bitrate(total_bits, fps, len(original_frames))
            print(f"Total_bits: {total_bits}")
            print(f"Bitrate: {bitrate} kbps")

            images2video(out_frames, wfp=output_video_path, fps=fps)
            print(f"Video saved to: {output_video_path}")

            xlsx_path  = os.path.join(output_dir, f"{model}_results.xlsx")
            result_dict = {"file_name": s_name, "S_QP": source_qp, "GOP": gop_size, "bitrate": bitrate}
            write_results(result_dict, xlsx_path)
