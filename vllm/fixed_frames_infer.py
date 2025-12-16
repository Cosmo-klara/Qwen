import os
import cv2
import numpy as np
import librosa
from vllm import LLM, SamplingParams
from torch import bfloat16

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["VLLM_USE_V1"] = "0"

def load_video_frames_fixed(
    path,
    num_frames=100,
    temporal_patch_size=2,
    min_frames=4,
    max_frames=768,
):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Invalid total frame count")

    num_frames = round(num_frames / temporal_patch_size) * temporal_patch_size
    num_frames = max(min_frames, min(num_frames, max_frames, total))

    idxs = np.linspace(
        0,
        total - 1,
        num=num_frames,
        dtype=np.int64,
    )

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    if len(frames) < num_frames:
        last = frames[-1]
        while len(frames) < num_frames:
            frames.append(last)

    cap.release()
    return np.stack(frames)


def load_audio_from_video(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y


def build_inputs_from_video(video_path, question, system_prompt=None):
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
    frames = load_video_frames_fixed(
        video_path,
        num_frames=100,
        temporal_patch_size=2,
    )
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

    model_name = "./cache/modelscope/Qwen/Qwen2.5-Omni-3B"

    import argparse
    parser = argparse.ArgumentParser(description="Run vLLM inference on a local video or dataset with 2 fps sampling")
    parser.add_argument("--video", help="Path to a local video file")
    parser.add_argument("--question", default="Describe the content of the video, then convert what the baby say into text")
    parser.add_argument("--target-fps", type=float, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-json")
    parser.add_argument("--video-dir")
    parser.add_argument("--task", type=str, default="all", choices=["captioning","grounding","seg_captioning","all"]) 
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    limits = {"audio": 1, "video": 1}

    llm = LLM(
        model=model_name,
        dtype=bfloat16,
        max_model_len=32760,
        max_num_seqs=5,
        limit_mm_per_prompt=limits,
        gpu_memory_utilization=0.6,
        seed=args.seed,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=64)

    if args.data_json and args.video_dir:
        from dataset import ProcessedVideoDataset
        ds = ProcessedVideoDataset(args.data_json, args.video_dir, task=args.task)
        batch_inputs = []
        metas = []
        for item in ds:
            inp, _ = build_inputs_from_video(item["video_path"], item["prompt"])
            batch_inputs.append(inp)
            metas.append((item["video_id"], item["task"], item.get("query_id", 0)))
            if len(batch_inputs) == args.batch_size:
                outs = llm.generate(batch_inputs, sampling_params=sampling_params)
                for meta, o in zip(metas, outs):
                    print(f"{meta[0]} [{meta[1]}#{meta[2]}]: {o.outputs[0].text}")
                batch_inputs, metas = [], []
        if batch_inputs:
            outs = llm.generate(batch_inputs, sampling_params=sampling_params)
            for meta, o in zip(metas, outs):
                print(f"{meta[0]} [{meta[1]}#{meta[2]}]: {o.outputs[0].text}")
    else:
        inputs, _limits = build_inputs_from_video(args.video, args.question, target_fps=args.target_fps)
        outs = llm.generate(inputs, sampling_params=sampling_params)
        for o in outs:
            print(o.outputs[0].text)