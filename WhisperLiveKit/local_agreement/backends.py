import io
import logging
import math
import sys
from typing import List

import numpy as np
import soundfile as sf

from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.text_utils import traditional_to_simplified
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe

logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, lora_path=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(
            self.start + offset,
            self.end + offset,
            self.text,
            speaker=getattr(self, 'speaker', -1),
            detected_language=getattr(self, 'detected_language', None),
            probability=getattr(self, 'probability', None)
        )

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_whisper_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)            
            if resolved_path.is_dir():
                model_info = detect_model_format(resolved_path)
                if not model_info.has_pytorch:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )            
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_whisper_model(str(resolved_path), lora_path=self.lora_path)

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_whisper_model(model_size, download_root=cache_dir, lora_path=self.lora_path)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                # 繁体转简体
                simplified_word = traditional_to_simplified(word["word"])
                token = ASRToken(
                    word["start"],
                    word["end"],
                    simplified_word,
                    probability=word.get("probability"),
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device 
        # 默认的int8_float16可能会有问题 需要环境
        compute_type = "auto"
        logger.info(f"Loading model with device={device}, compute_type={compute_type}")                      

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        print(f"Loaded model with device={model.model.device}, compute_type={model.model.compute_type}")
        logger.info(f"Actual device: {model.model.device}")
        logger.info(f"Actual compute_type: {model.model.compute_type}")
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        logger.info(f"Transcribing with init_prompt: {init_prompt}")
        logger.info(f"transcribe_kargs: {self.transcribe_kargs}")
        # Use beam_size from transcribe_kargs or default to 5
        beam_size = self.transcribe_kargs.get('beam_size', 5)
        logger.info(f"Use beam_size from transcribe_kargs: {beam_size}")
        # Create a copy of transcribe_kargs without beam_size to avoid duplicate argument
        transcribe_args = self.transcribe_kargs.copy()
        transcribe_args.pop('beam_size', None)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,  # 禁用，提高速度
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),  # 更积极的静音检测
            **transcribe_args,
        )
        return list(segments)
    def ts_words(self, segments) -> List[ASRToken]:
        """
        将 faster-whisper 的 segments 转换为 ASRToken 列表
        支持 word_timestamps=True/False 两种情况，并自动进行繁简转换
        """
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            # 检查是否有词级时间戳
            if hasattr(segment, 'words') and segment.words:
                # 有词级时间戳，逐个词处理
                for word in segment.words:
                    # 繁体转简体
                    simplified_word = traditional_to_simplified(word.word)
                    token = ASRToken(
                        word.start,
                        word.end,
                        simplified_word,
                        speaker=getattr(word, 'speaker', -1),
                        detected_language=getattr(word, 'detected_language', None),
                        probability=getattr(word, 'probability', None)
                    )
                    tokens.append(token)
            else:
                # 没有词级时间戳，使用整个段作为一个 token
                simplified_text = traditional_to_simplified(segment.text)
                token = ASRToken(
                    segment.start,
                    segment.end,
                    simplified_text,
                    speaker=getattr(segment, 'speaker', -1),
                    detected_language=getattr(segment, 'detected_language', None),
                    probability=getattr(segment, 'probability', None)
                )
                tokens.append(token)
        return tokens
        
    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                probability=word["probability"]
                # 繁体转简体
                simplified_word = traditional_to_simplified(word["word"])
                token = ASRToken(word["start"], word["end"], simplified_word)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if not self.direct_english_translation and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True
