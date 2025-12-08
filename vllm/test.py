from vllm.utils import FlexibleArgumentParser
from vllm.assets.video import VideoAsset
from vllm import LLM, SamplingParams
import vllm.envs as envs
from typing import NamedTuple
import os
from torch import bfloat16

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"

# 这里面是说 V1 版本的 vllm 不支持 use_audio_in_video: 0.8.5.post1
# ？ 或许可以直接提取音频出来？
os.environ["VLLM_USE_V1"] = "0"

class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def get_use_audio_in_video_query() -> QueryResult:
    question = ("Describe the content of the video, then convert what the baby say into text.")
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    asset = VideoAsset(name="sample_demo_1.mp4", num_frames=18)
    audio = asset.get_audio(sampling_rate=16000)
    assert not envs.VLLM_USE_V1, (
        "V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`."
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": asset.np_ndarrays,
                "audio": audio,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={
            "audio": 1,
            "video": 1
        },
    )


def main(args):
    model_name = "../cache/modelscope/Qwen/Qwen2.5-Omni-3B"

    query_result = get_use_audio_in_video_query()
    llm = LLM(
        model=model_name,
        dtype=bfloat16,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt=query_result.limit_mm_per_prompt,
        seed=args.seed,
        # gpu_memory_utilization=0.6,
        # 并行推理开启
        # tensor_parallel_size=2,
        # enforce_eager=True,

        # 使用 modelscope
        trust_remote_code=True
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=64)

    outputs = llm.generate(
        query_result.inputs,
        sampling_params=sampling_params
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with audio language models'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`."
    )

    args = parser.parse_args()
    main(args)