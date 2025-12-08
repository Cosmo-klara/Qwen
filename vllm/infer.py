import os
import cv2
import numpy as np
import librosa
from vllm import LLM, SamplingParams
from torch import bfloat16

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
os.environ["VLLM_USE_V1"] = "0"

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
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
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

    model_name = "../cache/modelscope/Qwen/Qwen2.5-Omni-3B"

    import argparse
    parser = argparse.ArgumentParser(description="Run vLLM inference on a local video with 2 fps sampling")
    parser.add_argument("--video", required=True, help="Path to a local video file")
    parser.add_argument("--question", default="Describe the content of the video, then convert what the baby say into text")
    parser.add_argument("--target-fps", type=float, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    inputs, limits = build_inputs_from_video(args.video, args.question, target_fps=args.target_fps)

    llm = LLM(
        model=model_name,
        dtype=bfloat16,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt=limits,
        seed=args.seed,

        # gpu_memory_utilization=0.6,
        # 并行推理开启
        # tensor_parallel_size=2,
        # enforce_eager=True,

        # 使用 modelscope
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=64)
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)