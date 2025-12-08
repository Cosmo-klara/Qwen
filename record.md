## 实验记录

### 在实验之前

- **Qwen2.5-Omni(Method A)**：动态帧率采样，根据音频的采样间隔来调整视频帧的提取，确保视觉帧与对应的音频片段在时间上保持一致。eg. 如果音频是每40ms生成一个表示，那么视觉帧也可能尽量调整到以40ms的间隔进行采样，或者以40ms为基础单位进行调整。

    导致长视频的输入在不进行调整的情况下是几乎不可能的

    Qwen2.5-Omni Github上给出的显存要求图

    | 模型 | 精度 | 15(s) 视频 | 30(s) 视频 | 60(s) 视频 |
    |--------------|-----------| ------------- | ------------- | ------------------ |
    | Qwen-Omni-3B | FP32 | 89.10 GB  | 不推荐 | 不推荐 |
    | Qwen-Omni-3B | BF16 | 18.38 GB  | 22.43 GB | 28.22 GB |
    | Qwen-Omni-7B | FP32 | 93.56 GB  | 不推荐 | 不推荐 |
    | Qwen-Omni-7B | BF16 | 31.11 GB  | 41.85 GB | 60.19 GB  |

    其量化模型：

    | Evaluation Set | Task | Metrics | Qwen2.5-Omni-7B | Qwen2.5-Omni-7B-GPTQ-Int4 | Qwen2.5-Omni-7B-AWQ |
    |--------------|-----------| ------------- | ------------- | ------------------ |  ------------------ |
    | LibriSpeech test-other   | ASR                   | WER ⬇️      | 3.4   | 3.71  | 3.91  |
    | WenetSpeech test-net     | ASR                   | WER ⬇️      | 5.9   | 6.62  | 6.31  |
    | Seed-TTS test-hard       | TTS (Speaker: Chelsie)| WER ⬇️      | 8.7   | 10.3  | 8.88  |
    | MMLU-Pro                 | Text -> Text          | Accuracy ⬆️ | 47.0  | 43.76 | 45.66 |
    | OmniBench                | Speech -> Text        | Accuracy ⬆️ | 56.13 | 53.59 | 54.64 |
    | VideoMME                 | Multimodality -> Text | Accuracy ⬆️ | 72.4  | 68.0  | 72.0  |

    Qwen2.5-Omni-7B-AWQ 的话多模态性能下降不多，显存要求几乎降了一半，把数据集处理一下，调整成 <30s 的数据测试？

    |Model | Precision | 15(s) Video | 30(s) Video | 60(s) Video |
    |--------------|-----------| ------------- | ------------- | ------------------ |
    | Qwen-Omni-7B | FP32 | 93.56 GB  | 不推荐 | 不推荐 |
    | Qwen-Omni-7B | BF16 | 31.11 GB  | 41.85 GB | 60.19 GB  |
    | Qwen-Omni-7B | GPTQ-Int4 | 11.64 GB  | 17.43 GB | 29.51 GB |
    | Qwen-Omni-7B | AWQ | 11.77 GB | 17.84 GB | 30.31 GB |

- **LongVALE(Method B)**：均匀采样固定数量（100）的视觉帧。

    或者把帧融合回视频？但是音频和视频在 Qwen 中需要对齐？

### Qwen2.5-omni-7B

#### 环境相关

```zsh
conda create -n qwen python=3.10
conda activate qwen
pip install av==14.3.0
pip install qwen-omni-utils==0.0.4
pip install librosa==0.11.0
pip install ffmpeg==1.4
pip install ffmpeg-python==0.2.0
pip install soundfile==0.13.1
pip install modelscope_studio==1.2.2
pip install transformers==4.52.3
pip install accelerate
pip install torchvision
pip install modelscope
```

#### 模型下载

```py
from modelscope import snapshot_download
model_dir = snapshot_download(
    "Qwen/Qwen2.5-Omni-7B", 
    cache_dir="./cache/modelscope"
)
```

### Qwen2.5-omni-7B

[测试用脚本](./test.ipynb)

爆显存，20s的视频

处理数据集：

```zsh
python tools/trim_dataset.py --input_json longvale-annotations-eval.json --video_dir raw_videos_test/video_test_1171 --output_json longvale-annotations-eval-30s.json --output_dir processed/videos --dry_run
```

- 视频数目：1172 -> 5386（丢弃83个原始视频）
- 事件数目：13867 -> 11612（丢弃2255个事件）




```zsh
nohup python longvalellm/eval/eval.py --video_feat_folder output/video_features --audio_feat_folder output/audio_features --asr_feat_folder output/speech_features --task all --log_path log_new > output_new.log 2>&1 & 
python longvalellm/eval/metric.py  --task seg_captioning --log_path log_new --data_path longvalellm/eval/longvale-annotations-eval-30s.json
```

longvale:

```zsh
====================== Grounding ======================
mIoU: 42.63
R1@0.3: 59.52
R1@0.5: 39.49
R1@0.7: 23.42
```



```zsh
nohup python qwen_eval.py --data_path longvale-annotations-eval-30s.json --video_dir processed/videos --task grounding --use_audio_in_video --log_path qwen_3b_eval_fps.log > output_fps.log 2>&1 &
nohup python qwen_eval.py --data_path longvale-annotations-eval-30s.json --video_dir processed/videos --task captioning --use_audio_in_video --log_path qwen_3b_eval_fps.log > output_fps.log 2>&1 &
nohup python qwen_eval.py --data_path longvale-annotations-eval-30s.json --video_dir processed/videos --task seg_captioning --use_audio_in_video --log_path qwen_3b_eval_fps_seg.log > output_fps_seg.log 2>&1 &
```

```zsh
python longvalellm/eval/metric.py  --task grounding --log_path longvalellm/eval/qwen_3b_eval.log --data_path longvalellm/eval/longvale-annotations-eval-30s.json
```

qwen：

```zsh
eval:  92%|█████████▏| 10637/11612 [161:38:43<20:17:51, 74.94s/it]WARNING:root:System prompt modified, audio output may not work as expected. Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
Unused or unrecognized kwargs: return_tensors, images.


====================== Grounding ======================
mIoU: 16.80
R1@0.3: 19.21
R1@0.5: 6.98
R1@0.7: 2.62
```


