import cv2
import numpy as np
import librosa


def load_video_frames(path, target_fps=2, temporal_patch_size=2, min_frames=4, max_frames=768):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if video_fps <= 0:
        desired = min(total, max_frames)
    else:
        desired = int(total * target_fps / video_fps)
    desired = max(min_frames, min(desired, max_frames, total))
    desired = (desired // temporal_patch_size) * temporal_patch_size
    desired = max(min_frames, min(desired, total))
    idxs = (np.arange(desired) * total) // desired
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    if len(frames) < desired:
        if len(frames) == 0:
            raise ValueError("no frames decoded")
        last = frames[-1]
        while len(frames) < desired:
            frames.append(last)
    frames = np.stack(frames)
    return frames


def load_audio_from_video(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y


def build_inputs_from_video(video_path, question, target_fps=2, system_prompt=None):
    system_prompt = system_prompt or (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    frames = load_video_frames(video_path, target_fps=target_fps)
    audio = load_audio_from_video(video_path, sr=16000)
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "video": frames,
            "audio": audio,
        },
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }
    limits = {"audio": 1, "video": 1}
    return inputs, limits

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build inputs from a local video with 2 fps sampling")
    parser.add_argument("--video", required=True, help="Path to a local video file")
    parser.add_argument("--question", default="Describe the video and transcribe speech.")
    parser.add_argument("--target-fps", type=float, default=2)
    args = parser.parse_args()

    inputs, limits = build_inputs_from_video(args.video, args.question, target_fps=args.target_fps)
    frames = inputs["multi_modal_data"]["video"]
    audio = inputs["multi_modal_data"]["audio"]

    preview = inputs["prompt"][:160].replace("\n", " ")
    print(f"frames shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"num frames: {frames.shape[0]}")
    print(f"audio samples: {audio.shape[0]}, duration_sec: {audio.shape[0]/16000:.2f}")
    print(f"prompt preview: {preview}...")
    print(f"mm_processor_kwargs: {inputs.get('mm_processor_kwargs')}")
    print(f"limits: {limits}")


    
    # frames shape: (18, 360, 640, 3), dtype: uint8
    # num frames: 18
    # audio samples: 155296, duration_sec: 9.71
    # prompt preview: <|im_start|>system You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generat...
    # mm_processor_kwargs: {'use_audio_in_video': True}
    # limits: {'audio': 1, 'video': 1}
