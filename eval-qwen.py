import os
import json
import argparse
import re
from tqdm import tqdm
from modelscope import snapshot_download
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch import bfloat16
from modelscope import Qwen2_5OmniForConditionalGeneration
from dataset import ProcessedVideoDataset
import torch
import numpy as np


def make_prefix_allowed_tokens_fn(tokenizer, input_len):
    def ids_for(s):
        try:
            return set(tokenizer.encode(s, add_special_tokens=False))
        except Exception:
            return set()
    digits_ids = set()
    for d in "0123456789":
        digits_ids |= ids_for(d)
    from_ids = ids_for("From ") or ids_for("From")
    to_ids = ids_for(" to ") or ids_for(" to") or ids_for("to ")
    and_ids = ids_for(" and ") or ids_for(" and") or ids_for("and ")
    dot_ids = ids_for(".")
    space_ids = ids_for(" ")

    def fn(batch_id, input_ids):
        gen_ids = input_ids[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        digit_count = len(re.findall(r"\d", text))
        if not text.startswith("From "):
            allowed = from_ids | space_ids
            return list(allowed) if allowed else list(digits_ids)
        if digit_count < 2:
            return list(digits_ids) if digits_ids else list(space_ids)
        if (" to " not in text) and (" and " not in text):
            allowed = to_ids | and_ids
            return list(allowed) if allowed else list(space_ids)
        if digit_count < 4:
            return list(digits_ids) if digits_ids else list(space_ids)
        return list(dot_ids) if dot_ids else list(space_ids)

    return fn


def to_device_and_dtype(inputs, device, model_dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                v = v.to(model_dtype)
            out[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if k in ("input_ids", "attention_mask", "position_ids"):
                t = t.to(torch.long)
            elif t.dtype.is_floating_point:
                t = t.to(model_dtype)
            out[k] = t.to(device)
        elif isinstance(v, (list, tuple)):
            try:
                t = torch.tensor(v)
            except Exception:
                t = torch.as_tensor(np.array(v))
            if k in ("input_ids", "attention_mask", "position_ids"):
                t = t.to(torch.long)
            elif t.dtype.is_floating_point:
                t = t.to(model_dtype)
            out[k] = t.to(device)
        else:
            out[k] = v
    return out


def normalize_grounding_answer(ans: str, duration: float | None):
    s = ans.strip()
    m = re.search(r"from\s+(\d+(?:\.\d+)?)\s*(to|and)\s+(\d+(?:\.\d+)?)", s.lower())
    if m:
        a = int(round(float(m.group(1))))
        b = int(round(float(m.group(3))))
        a = max(0, min(99, a))
        b = max(0, min(99, b))
        if a > b:
            a, b = b, a
        return f"From {a:02d} to {b:02d}."
    if duration and duration > 0:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds", s.lower())
        if m2:
            start_s = float(m2.group(1))
            end_s = float(m2.group(2))
            a = max(0, min(99, int(round(start_s / duration * 100))))
            b = max(0, min(99, int(round(end_s / duration * 100))))
            if a > b:
                a, b = b, a
            return f"From {a:02d} to {b:02d}."
    return s


def iou(outputs, gt):
    m = re.search(r"(\d{2}) (to|and) (\d{2})", outputs.strip().lower())
    if not m:
        return 0.0
    a, b = float(m.group(1))/100.0, float(m.group(3))/100.0
    s, e = gt
    inter = max(0, min(b, e) - max(a, s))
    union = max(b, e) - min(a, s)
    return round(inter/union if union > 0 else 0.0, 2)


def run(model, processor, video_path, prompt, use_audio_in_video=True, task=None):
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are Qwen, a time-aware multimodal assistant. Follow the required format strictly."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    proc_kwargs = {"text": text, "videos": videos, "padding": True, "return_tensors": "pt"}
    if use_audio_in_video and audios is not None:
        proc_kwargs["audio"] = audios
    try:
        inputs = processor(**proc_kwargs)
    except TypeError:
        proc_kwargs.pop("return_tensors", None)
        inputs = processor(**proc_kwargs)
    inputs = to_device_and_dtype(inputs, model.device, bfloat16)
    input_ids = inputs.get("input_ids", None)
    input_len = input_ids.shape[1] if input_ids is not None else 0
    prefix_fn = None
    if task == "grounding":
        prefix_fn = make_prefix_allowed_tokens_fn(processor.tokenizer, input_len)
    out = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        do_sample=False,
        max_new_tokens=64,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    input_ids = inputs.get("input_ids", None)
    gen_ids = out[:, input_ids.shape[1]:] if input_ids is not None else out
    ans = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return ans.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--video_dir", type=str, required=True)
    ap.add_argument("--log_path", type=str, default="qwen_3b_eval.log")
    ap.add_argument("--task", type=str, default="all",
                    choices=["all", "grounding", "captioning", "seg_captioning"])
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--cache_dir", type=str, default="./cache/modelscope")
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--flush_every", type=int, default=3)
    ap.add_argument("--use_audio_in_video", action="store_true", help="Enable audio extraction from video for multimodal inference")
    args = ap.parse_args()

    model_dir = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, device_map=args.device, torch_dtype=bfloat16, attn_implementation="flash_attention_2"
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

    dur_js = json.load(open(args.data_path, "r", encoding="utf-8"))
    dur_map = {k: float(v.get("duration", 0)) for k, v in dur_js.items()}

    ds = ProcessedVideoDataset(args.data_path, args.video_dir, task=args.task)
    total = len(ds)
    print(f"total samples: {total}")
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as lf:
        wcount = 0
        for sample in tqdm(ds, desc="eval", total=total):
            vid = sample["video_id"]
            vpath = sample["video_path"]
            prompt = sample["prompt"]
            task = sample["task"]
            qid = sample["query_id"]
            if not os.path.isfile(vpath):
                print(f"skip: missing file {vpath}")
                continue
            if task == "grounding":
                prompt = prompt + " Only output exactly: From xx to xx. Use two digits 00-99. Examples:\nFrom 00 to 07.\nFrom 08 to 23. No explanation."
            elif task == "captioning":
                prompt = prompt + " use two digits 00-99. Examples:\nFrom 00 to 07.\nFrom 08 to 23. Do not use punctuation that requires escaping like \\' or \\\"."
            elif task == "seg_captioning":
                prompt = prompt + " Please give the event description directly. Do not start with phrases like 'The event is:'. Do not use punctuation that requires escaping like \\' or \\\"."
            try:
                ans = run(model, processor, vpath, prompt, use_audio_in_video=args.use_audio_in_video, task=task)
            except Exception as e:
                print(f"error on {vid}: {e}")
                continue
            if task == "grounding":
                ans = normalize_grounding_answer(ans, dur_map.get(vid))
            rec = {
                "video_id": vid, "task": task,
                "query_id": qid, "answer": ans
            }
            if task == "grounding" and "gt" in sample:
                rec["info"] = {
                    "sentence_id": qid,
                    "iou": iou(ans, sample["gt"])
                }
            elif task == "seg_captioning":
                rec["info"] = {"sentence_id": qid}
            lf.write(json.dumps(rec) + "\n")
            wcount += 1
            if args.flush_every and wcount % args.flush_every == 0:
                lf.flush()


if __name__ == "__main__":
    main()