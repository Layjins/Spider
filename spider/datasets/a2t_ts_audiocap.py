import os
import logging
import warnings
import random
import json

import torch
from torch.utils.data import Dataset
import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision import transforms

from spider.common.registry import registry
from spider.processors import *


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


class A2TTsAudioCapDataset(Dataset):
    def __init__(self, audio_dir,
                 ann_path,
                 num_mel_bins=128,
                 target_length=204,
                 sample_rate=16000,
                 clip_duration=2,
                 clips_per_video=3,
                 mean=-4.268,
                 std=9.138,):
        self.audio_dir = audio_dir

        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.mean = mean
        self.std = std

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )

        with open(ann_path, 'r') as file:
            self.annotations = json.load(file)

        self.instruction_pool = [
            "What is this audio",
            "Please generate a caption for this audio",
            "Could you provide a descriptive caption for this audio",
            "I need your help in generating a caption for this audio",
            "Could you describe this audio for me",
            "I would like you to generate a caption for this audio",
            "Please provide a descriptive caption for this audio",
            "Generate similar contents to this audio",
            "Please generate a multimodal caption for this audio",
            "Please generate the contents that are similar to this audio",
            "Please generate an <IMAGE> that is similar to this audio",
            "Please generate a <VIDEO> that is similar to this audio",
            "Please generate an <AUDIO> that is similar to this audio",
            "Please generate an <IMAGE> and a <VIDEO> that are similar to this audio",
            "Please generate an <IMAGE> and an <AUDIO> that are similar to this audio",
            "Please generate a <VIDEO> and an <AUDIO> that are similar to this audio",
            "Please generate an <IMAGE>, a <VIDEO>, and an <AUDIO> that are similar to this audio",
        ]

    def __len__(self):
        return len(self.annotations)

    def preprocess(self, index):
        # import pdb
        # pdb.set_trace()
        ann = self.annotations[index]
        audio_name = ann['audio_name']
        caption = ann['caption']
        caption = text_processor(caption)

        audio_path = os.path.join(self.audio_dir, audio_name)

        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                            :,
                            int(clip_timepoints[0] * self.sample_rate): int(
                                clip_timepoints[1] * self.sample_rate
                            ),
                            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        # import pdb
        # pdb.set_trace()
        return all_clips, caption

    def __getitem__(self, index):
        all_clips, caption = self.preprocess(index)

        instruction = random.choice(self.instruction_pool)
        instruction_q = instruction

        # Determine which modalities to include based on the instruction
        modalities = {
            "<IMAGE>": "<IMAGE>{}<IMAGE-Placeholder></IMAGE>",
            "<VIDEO>": "<VIDEO>{}<VIDEO-Placeholder></VIDEO>",
            "<AUDIO>": "<AUDIO>{}<AUDIO-Placeholder></AUDIO>",
        }

        # Build the question and answer based on the selected instruction
        answer_components = []
        if "<IMAGE>" in instruction:
            instruction_q = instruction_q.replace("<IMAGE>", "image")
            answer_components.append(modalities["<IMAGE>"].format(caption))
        if "<VIDEO>" in instruction:
            instruction_q = instruction_q.replace("<VIDEO>", "video")
            answer_components.append(modalities["<VIDEO>"].format(caption))
        if "<AUDIO>" in instruction:
            instruction_q = instruction_q.replace("<AUDIO>", "audio")
            answer_components.append(modalities["<AUDIO>"].format(caption))
        if ("<IMAGE>" not in instruction) and ("<VIDEO>" not in instruction) and ("<AUDIO>" not in instruction):
            answer_components.append(modalities["<IMAGE>"].format(caption))
            answer_components.append(modalities["<VIDEO>"].format(caption))
            answer_components.append(modalities["<AUDIO>"].format(caption))

        question = "<AUDIO><AUDIO-Placeholder></AUDIO> {}".format(instruction_q)
        answer = "{}{}".format(caption, "".join(answer_components))

        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "AUDIO": all_clips,
            "Caption": caption
        }


@registry.register_builder("a2t_ts_audiocap")
class A2TTsAudioCapBuilder:
    train_dataset_cls = A2TTsAudioCapDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building A2TTsAudioCap datasets...")

        build_info = self.config.build_info
        audio_dir = build_info.audio_dir
        ann_path = build_info.ann_path

        if not os.path.exists(audio_dir):
            warnings.warn("audio dir {} does not exist.".format(audio_dir))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            audio_dir=audio_dir,
            ann_path=ann_path
        )
