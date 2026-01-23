import asyncio
import base64
import json
import os
import time
import uuid
import warnings
from enum import Enum
import threading
from threading import Event
from typing import (
    Any,
    ByteString,
    Generator,
    Optional,
    Sequence,
    Tuple
)

import azure.cognitiveservices.speech as speechsdk
import boto3
import pvcheetah
import pvleopard
import requests
import soundfile
import torch
import whisper
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import (
    TranscriptEvent,
    TranscriptResultStream
)
from azure.cognitiveservices.speech import SpeechRecognitionEventArgs
from google.cloud import speech
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
import dashscope
from deepgram import DeepgramClient
from deepgram.core.api_error import ApiError as DeepgramApiError

from languages import (
    LANGUAGE_TO_CODE,
    Languages
)

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings(
    "ignore", message="Performing inference on CPU when CUDA is available")

NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


class Engines(Enum):
    AMAZON_TRANSCRIBE = "AMAZON_TRANSCRIBE"
    AMAZON_TRANSCRIBE_STREAMING = "AMAZON_TRANSCRIBE_STREAMING"
    AZURE_SPEECH_TO_TEXT = "AZURE_SPEECH_TO_TEXT"
    AZURE_SPEECH_TO_TEXT_REAL_TIME = "AZURE_SPEECH_TO_TEXT_REAL_TIME"
    GOOGLE_SPEECH_TO_TEXT = "GOOGLE_SPEECH_TO_TEXT"
    GOOGLE_SPEECH_TO_TEXT_STREAMING = "GOOGLE_SPEECH_TO_TEXT_STREAMING"
    GOOGLE_SPEECH_TO_TEXT_ENHANCED = "GOOGLE_SPEECH_TO_TEXT_ENHANCED"
    GOOGLE_SPEECH_TO_TEXT_ENHANCED_STREAMING = "GOOGLE_SPEECH_TO_TEXT_ENHANCED_STREAMING"
    IBM_WATSON_SPEECH_TO_TEXT = "IBM_WATSON_SPEECH_TO_TEXT"
    WHISPER_TINY = "WHISPER_TINY"
    WHISPER_BASE = "WHISPER_BASE"
    WHISPER_SMALL = "WHISPER_SMALL"
    WHISPER_MEDIUM = "WHISPER_MEDIUM"
    WHISPER_LARGE = "WHISPER_LARGE"
    WHISPER_LARGE_V2 = "WHISPER_LARGE_V2"
    WHISPER_LARGE_V3 = "WHISPER_LARGE_V3"
    PICOVOICE_CHEETAH = "PICOVOICE_CHEETAH"
    PICOVOICE_CHEETAH_FAST = "PICOVOICE_CHEETAH_FAST"
    PICOVOICE_LEOPARD = "PICOVOICE_LEOPARD"
    SONIOX = "SONIOX"
    DEEPGRAM = "DEEPGRAM"
    ELEVENLABS = "ELEVENLABS"
    DASHSCOPE = "DASHSCOPE"
    IFLYREC = "IFLYREC"
    IFLYREC_BATCH = "IFLYREC_BATCH"
    IFLYREC_IST = "IFLYREC_IST"
    SONIOX_REALTIME = "SONIOX_REALTIME"


StreamingEngines = [
    Engines.AMAZON_TRANSCRIBE_STREAMING,
    Engines.AZURE_SPEECH_TO_TEXT_REAL_TIME,
    Engines.GOOGLE_SPEECH_TO_TEXT_STREAMING,
    Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED_STREAMING,
    Engines.PICOVOICE_CHEETAH,
    Engines.PICOVOICE_CHEETAH_FAST,
    Engines.SONIOX_REALTIME,
]


class Engine(object):
    def __init__(self, no_cache: bool = False):
        self._no_cache = no_cache

    def transcribe(self, path: str) -> str:
        raise NotImplementedError()

    def audio_sec(self) -> float:
        raise NotImplementedError()

    def process_sec(self) -> float:
        raise NotImplementedError()

    def delete(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Engines, language: Languages, no_cache: bool = False, **kwargs):
        if x is Engines.AMAZON_TRANSCRIBE:
            return AmazonTranscribeEngine(language=language, no_cache=no_cache)
        if x is Engines.AMAZON_TRANSCRIBE_STREAMING:
            return AmazonTranscribeStreamingEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.AZURE_SPEECH_TO_TEXT:
            return AzureSpeechToTextEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.AZURE_SPEECH_TO_TEXT_REAL_TIME:
            return AzureSpeechToTextRealTimeEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.GOOGLE_SPEECH_TO_TEXT:
            return GoogleSpeechToTextEngine(language=language, no_cache=no_cache)
        elif x is Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED:
            return GoogleSpeechToTextEnhancedEngine(language=language, no_cache=no_cache)
        elif x is Engines.GOOGLE_SPEECH_TO_TEXT_STREAMING:
            return GoogleSpeechToTextStreamingEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED_STREAMING:
            return GoogleSpeechToTextEnhancedStreamingEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.WHISPER_TINY:
            return WhisperTiny(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_BASE:
            return WhisperBase(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_SMALL:
            return WhisperSmall(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_MEDIUM:
            return WhisperMedium(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_LARGE:
            return WhisperLarge(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_LARGE_V2:
            return WhisperLargeV2(language=language, no_cache=no_cache)
        elif x is Engines.WHISPER_LARGE_V3:
            return WhisperLargeV3(language=language, no_cache=no_cache)
        elif x is Engines.PICOVOICE_CHEETAH:
            return PicovoiceCheetahEngine(no_cache=no_cache, **kwargs)
        elif x is Engines.PICOVOICE_CHEETAH_FAST:
            return PicovoiceCheetahEngine(no_cache=no_cache, **kwargs)
        elif x is Engines.PICOVOICE_LEOPARD:
            return PicovoiceLeopardEngine(no_cache=no_cache, **kwargs)
        elif x is Engines.IBM_WATSON_SPEECH_TO_TEXT:
            return IBMWatsonSpeechToTextEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.SONIOX:
            return SonioxAsyncEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.SONIOX_REALTIME:
            return SonioxRealtimeEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.DEEPGRAM:
            return DeepgramEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.ELEVENLABS:
            return ElevenLabsEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.DASHSCOPE:
            return DashscopeEngine(language=language, no_cache=no_cache, **kwargs)
        elif x is Engines.IFLYREC:
            return IflyrecEngine(language=language, no_cache=no_cache)
        elif x is Engines.IFLYREC_BATCH:
            return IflyrecBatchEngine(language=language, no_cache=no_cache)
        elif x is Engines.IFLYREC_IST:
            return IflyrecIstEngine(language=language, no_cache=no_cache, **kwargs)
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{x}`")


WordLatencyOutputType = Tuple[Sequence[str], Sequence[float], Sequence[float]]


class StreamingEngine(Engine):
    @property
    def is_async(self) -> bool:
        raise NotImplementedError()

    async def _measure_word_latency_async(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        raise NotImplementedError()

    def _measure_word_latency(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        raise NotImplementedError()

    def measure_word_latency(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        if self.is_async:
            return asyncio.run(self._measure_word_latency_async(path, alignments))
        else:
            return self._measure_word_latency(path, alignments)

    def transcribe(self, path: str) -> str:
        words, _, _ = self.measure_word_latency(path, alignments=None)
        return " ".join(words)

    def get_chunk_size_ms(self) -> int:
        raise NotImplementedError()

    def load_pcm(self, path: str) -> ByteString:
        pcm, sample_rate = soundfile.read(path, dtype="int16")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Incorrect sample rate for `{path}`: expected {SAMPLE_RATE} got {sample_rate}")
        return pcm.tobytes()

    def get_chunk_size_bytes(self) -> int:
        chunk_ms = self.get_chunk_size_ms()
        return int((chunk_ms / 1000) * (SAMPLE_RATE * BYTES_PER_SAMPLE))


class AmazonTranscribeEngine(Engine):
    def __init__(self, language: Languages, aws_location: str = "us-west-2", no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]

        self._s3_client = boto3.client("s3")
        self._s3_bucket = str(uuid.uuid4())
        self._s3_client.create_bucket(
            ACL="private",
            Bucket=self._s3_bucket,
            CreateBucketConfiguration={"LocationConstraint": aws_location},
        )

        self._transcribe_client = boto3.client("transcribe")

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".aws")

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                res = f.read()
            return res

        job_name = str(uuid.uuid4())
        s3_object = os.path.basename(path)
        self._s3_client.upload_file(path, self._s3_bucket, s3_object)

        self._transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={
                "MediaFileUri": f"https://s3-us-west-2.amazonaws.com/{self._s3_bucket}/{s3_object}"},
            MediaFormat="flac",
            LanguageCode=self._language_code,
        )

        while True:
            status = self._transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
            if job_status == "COMPLETED":
                break
            elif job_status == "FAILED":
                error = status["TranscriptionJob"].get(
                    "FailureReason", "Unknown error")
                raise RuntimeError(
                    f"Amazon Transcribe job {job_name} failed: {error}")
            time.sleep(1)

        content = requests.get(
            status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"])

        res = json.loads(content.content.decode("utf8"))[
            "results"]["transcripts"][0]["transcript"]

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket)
        while response["KeyCount"] > 0:
            self._s3_client.delete_objects(
                Bucket=self._s3_bucket,
                Delete={"Objects": [{"Key": obj["Key"]}
                                    for obj in response["Contents"]]},
            )
            response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket)

        self._s3_client.delete_bucket(Bucket=self._s3_bucket)

    def __str__(self):
        return "Amazon Transcribe"


class AmazonTranscribeStreamingEngine(StreamingEngine):
    def __init__(
        self,
        language: Languages,
        chunk_size_ms: int,
        apply_delay: bool,
        ignore_punctuation: bool,
        aws_location: str = "us-west-2",
        no_cache: bool = False,
    ) -> None:
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]
        self._chunk_size_ms = chunk_size_ms
        self._apply_delay = apply_delay
        self._ignore_punctuation = ignore_punctuation
        self._location = aws_location

        self._client = TranscribeStreamingClient(region=self._location)

    @property
    def is_async(self) -> bool:
        return True

    def get_chunk_size_ms(self) -> int:
        return self._chunk_size_ms

    async def _measure_word_latency_async(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        cache_path = path.replace(".flac", ".awsrt")

        if not self._no_cache and alignments is None and os.path.exists(cache_path):
            with open(cache_path) as f:
                res = f.read()
            return res.split(), [], []

        stream = await self._client.start_stream_transcription(
            language_code=self._language_code,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )

        handler = AmazonTranscribeStreamingHandler(
            stream.output_stream, ignore_punctuation=self._ignore_punctuation)
        send_timings = []

        async def write_chunks():
            current_audio_time = 0.0
            word_timings = [aln[-1]
                            for aln in alignments] if alignments is not None else []
            pcm = self.load_pcm(path)

            total_bytes = len(pcm)
            current_byte = 0
            chunk_size_bytes = self.get_chunk_size_bytes()

            while current_byte < total_bytes:
                chunk = pcm[current_byte: current_byte + chunk_size_bytes]
                chunk_end_time = current_audio_time + \
                    (self._chunk_size_ms / 1000)

                send_time = time.time()
                await stream.input_stream.send_audio_event(audio_chunk=chunk)

                for word_time in word_timings:
                    if current_audio_time < word_time <= chunk_end_time:
                        send_timings.append(send_time)

                if self._apply_delay:
                    await asyncio.sleep(self._chunk_size_ms / 1000)

                current_audio_time = chunk_end_time
                current_byte += chunk_size_bytes

            await stream.input_stream.end_stream()

        await asyncio.gather(write_chunks(), handler.handle_events())

        if alignments is None:
            with open(cache_path, "w") as f:
                f.write(" ".join(handler._emitted_words))

        return handler._emitted_words, handler._receive_timings, send_timings

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Amazon Transcribe Streaming"


class AmazonTranscribeStreamingHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream: TranscriptResultStream, ignore_punctuation: bool) -> None:
        super().__init__(transcript_result_stream)
        self._emitted_words = []
        self._receive_timings = []
        self._last_word_index = 0

        self._ignore_punctuation = ignore_punctuation
        self._punctuation_trans = str.maketrans({".": "", ",": "", "?": ""})

    async def handle_transcript_event(self, transcript_event: TranscriptEvent) -> None:
        current_time = time.time()

        results = transcript_event.transcript.results
        for result in results:
            if result.alternatives:
                for alt in result.alternatives:
                    if alt.transcript:
                        if self._ignore_punctuation:
                            words = alt.transcript.translate(
                                self._punctuation_trans).split()
                        else:
                            words = alt.transcript.split()

                        partial_transcript_reset = len(
                            words) < self._last_word_index
                        if partial_transcript_reset:
                            self._last_word_index = 0

                        if self._last_word_index > 0:
                            last_emitted_word_changed = self._emitted_words[-1] != words[self._last_word_index - 1]
                            if last_emitted_word_changed:
                                self._emitted_words[-1] = words[self._last_word_index - 1]
                                self._receive_timings[-1] = current_time

                        if len(words) > self._last_word_index:
                            new_words = words[self._last_word_index:]
                            for word in new_words:
                                self._emitted_words.append(word)
                                self._receive_timings.append(current_time)

                            self._last_word_index = len(words)


class AzureSpeechToTextEngine(Engine):
    def __init__(
        self,
        azure_speech_key: str,
        azure_speech_location: str,
        language: Languages,
        no_cache: bool = False,
    ):
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]
        self._azure_speech_key = azure_speech_key
        self._azure_speech_location = azure_speech_location

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".ms")

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res

        wav_path = path.replace(".flac", ".wav")
        soundfile.write(
            wav_path,
            soundfile.read(path, dtype="int16")[0],
            samplerate=SAMPLE_RATE,
        )

        speech_config = speechsdk.SpeechConfig(
            subscription=self._azure_speech_key,
            region=self._azure_speech_location,
            speech_recognition_language=self._language_code,
        )
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        res = ""

        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                nonlocal res
                res += " " + evt.result.text

        done = False

        def stop_cb(_):
            nonlocal done
            done = True

        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        speech_recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.5)

        speech_recognizer.stop_continuous_recognition()

        os.remove(wav_path)

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Microsoft Azure Speech-to-text"


class AzureSpeechToTextRealTimeEngine(StreamingEngine):
    def __init__(
        self,
        language: Languages,
        chunk_size_ms: int,
        apply_delay: bool,
        ignore_punctuation: bool,
        azure_speech_key: str,
        azure_speech_location: str,
        no_cache: bool = False,
    ) -> None:
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]
        self._chunk_size_ms = chunk_size_ms
        self._apply_delay = apply_delay
        self._ignore_punctuation = ignore_punctuation
        self._azure_speech_key = azure_speech_key
        self._azure_speech_location = azure_speech_location

    @property
    def is_async(self) -> bool:
        return True

    def get_chunk_size_ms(self) -> int:
        return self._chunk_size_ms

    async def _measure_word_latency_async(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        cache_path = path.replace(".flac", ".msrt")

        if not self._no_cache and alignments is None and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res.split(), [], []

        speech_config = speechsdk.SpeechConfig(
            subscription=self._azure_speech_key,
            region=self._azure_speech_location,
            speech_recognition_language=self._language_code,
        )

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=SAMPLE_RATE)
        push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        handler = AzureSpeechToTextRealTimeHandler(
            ignore_punctuation=self._ignore_punctuation)
        speech_recognizer.recognizing.connect(handler.recognizing_cb)
        speech_recognizer.recognized.connect(handler.recognized_cb)
        speech_recognizer.session_stopped.connect(handler.session_stopped_cb)
        speech_recognizer.canceled.connect(handler.canceled_cb)

        send_timings = []

        async def write_chunks() -> None:
            current_audio_time = 0.0
            word_timings = [aln[-1]
                            for aln in alignments] if alignments is not None else []
            pcm = self.load_pcm(path)

            total_bytes = len(pcm)
            current_byte = 0
            chunk_size_bytes = self.get_chunk_size_bytes()

            while current_byte < total_bytes:
                chunk = pcm[current_byte: current_byte + chunk_size_bytes]
                chunk_end_time = current_audio_time + \
                    (self._chunk_size_ms / 1000)

                send_time = time.time()
                push_stream.write(chunk)

                for word_time in word_timings:
                    if current_audio_time < word_time <= chunk_end_time:
                        send_timings.append(send_time)

                if self._apply_delay:
                    await asyncio.sleep(self._chunk_size_ms / 1000)

                current_audio_time = chunk_end_time
                current_byte += chunk_size_bytes

            push_stream.close()

        speech_recognizer.start_continuous_recognition_async()
        await write_chunks()

        await asyncio.get_event_loop().run_in_executor(None, handler._done_event.wait, 10)

        speech_recognizer.stop_continuous_recognition_async()

        if alignments is None:
            with open(cache_path, "w") as f:
                f.write(" ".join(handler._emitted_words))

        return handler._emitted_words, handler._receive_timings, send_timings

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Microsoft Azure Speech-to-text Real-time"


class AzureSpeechToTextRealTimeHandler:
    def __init__(self, ignore_punctuation: bool) -> None:
        self._emitted_words = []
        self._receive_timings = []
        self._last_word_index = 0
        self._done_event = Event()

        self._ignore_punctuation = ignore_punctuation
        self._punctuation_trans = str.maketrans({".": "", ",": "", "?": ""})

    def _recognize_helper(self, evt: SpeechRecognitionEventArgs) -> None:
        current_time = time.time()
        if self._ignore_punctuation:
            words = evt.result.text.translate(self._punctuation_trans).split()
        else:
            words = evt.result.text.split()

        partial_transcript_reset = len(words) < self._last_word_index
        if partial_transcript_reset:
            self._last_word_index = 0

        if self._last_word_index > 0:
            last_emitted_word_changed = self._emitted_words[-1] != words[self._last_word_index - 1]
            if last_emitted_word_changed:
                self._emitted_words[-1] = words[self._last_word_index - 1]
                self._receive_timings[-1] = current_time

        if len(words) > self._last_word_index:
            new_words = words[self._last_word_index:]
            for word in new_words:
                self._emitted_words.append(word)
                self._receive_timings.append(current_time)

            self._last_word_index = len(words)

    def recognized_cb(self, evt: SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.text:
                self._recognize_helper(evt)

    def recognizing_cb(self, evt: SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            if evt.result.text:
                self._recognize_helper(evt)

    def session_stopped_cb(self, evt: SpeechRecognitionEventArgs) -> None:
        self._done_event.set()

    def canceled_cb(self, evt: SpeechRecognitionEventArgs) -> None:
        self._done_event.set()


class GoogleSpeechToTextEngine(Engine):
    def __init__(
        self,
        language: Languages,
        cache_extension: str = ".ggl",
        model: Optional[str] = None,
        no_cache: bool = False,
    ):
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]

        self._client = speech.SpeechClient()

        self._config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self._language_code,
            model=model,
            enable_automatic_punctuation=True,
        )

        self._cache_extension = cache_extension

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", self._cache_extension)
        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                res = f.read()
            return res

        with open(path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)

        response = self._client.recognize(config=self._config, audio=audio)

        res = " ".join(
            result.alternatives[0].transcript for result in response.results)

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Google Speech-to-Text"


class GoogleSpeechToTextEnhancedEngine(GoogleSpeechToTextEngine):
    def __init__(self, language: Languages, no_cache: bool = False):
        if language != Languages.EN:
            raise ValueError(
                "GOOGLE_SPEECH_TO_TEXT_ENHANCED engine only supports EN language")
        super().__init__(language=language, cache_extension=".ggle",
                         model="video", no_cache=no_cache)

    def __str__(self) -> str:
        return "Google Speech-to-Text Enhanced"


class GoogleSpeechToTextStreamingEngine(StreamingEngine):
    def __init__(
        self,
        language: Languages,
        chunk_size_ms: int,
        apply_delay: bool,
        ignore_punctuation: bool,
        cache_extension: str = ".gglrt",
        model: Optional[str] = None,
        no_cache: bool = False,
    ) -> None:
        super().__init__(no_cache=no_cache)
        self._language_code = LANGUAGE_TO_CODE[language]
        self._chunk_size_ms = chunk_size_ms
        self._apply_delay = apply_delay
        self._ignore_punctuation = ignore_punctuation
        self._cache_extension = cache_extension

        self._client = speech.SpeechClient()

        self._config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self._language_code,
            model=model,
            enable_automatic_punctuation=True,
        )

        self._streaming_config = speech.StreamingRecognitionConfig(
            config=self._config, interim_results=True, single_utterance=False
        )

    @property
    def is_async(self) -> bool:
        return False

    def get_chunk_size_ms(self) -> int:
        return self._chunk_size_ms

    def _measure_word_latency(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        cache_path = path.replace(".flac", self._cache_extension)
        if not self._no_cache and alignments is None and os.path.exists(cache_path):
            with open(cache_path) as f:
                res = f.read()
            return res.split(), [], []

        word_timings = [aln[-1]
                        for aln in alignments] if alignments is not None else []
        pcm = self.load_pcm(path)

        streamer = GoogleSpeechToTextStreamingAudioGenerator(
            pcm=pcm,
            word_timings=word_timings,
            chunk_size_bytes=self.get_chunk_size_bytes(),
            chunk_size_ms=self._chunk_size_ms,
            apply_delay=self._apply_delay,
        )
        handler = GoogleSpeechToTextStreamingHandler(
            ignore_punctuation=self._ignore_punctuation)

        def request_generator():
            yield from streamer.stream_generator()

        responses = self._client.streaming_recognize(
            config=self._streaming_config, requests=request_generator())

        for response in responses:
            if len(response.results) == 0:
                continue

            if response.results[0].is_final or len(response.results) == 2:
                handler._process_result(response.results[0])

        streamer.stop()

        if alignments is None:
            with open(cache_path, "w") as f:
                f.write(" ".join(handler._emitted_words))

        return handler._emitted_words, handler._receive_timings, streamer._send_timings

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Google Speech-to-Text Streaming"


class GoogleSpeechToTextEnhancedStreamingEngine(GoogleSpeechToTextStreamingEngine):
    def __init__(
        self,
        language: Languages,
        chunk_size_ms: int,
        apply_delay: bool,
        ignore_punctuation: bool,
        no_cache: bool = False,
    ) -> None:
        if language != Languages.EN:
            raise ValueError(
                "GOOGLE_SPEECH_TO_TEXT_ENHANCED_STREAMING engine only supports EN language")
        super().__init__(
            chunk_size_ms=chunk_size_ms,
            apply_delay=apply_delay,
            ignore_punctuation=ignore_punctuation,
            language=language,
            cache_extension=".gglert",
            model="video",
            no_cache=no_cache,
        )

    def __str__(self) -> str:
        return "Google Speech-to-Text Enhanced Streaming"


class GoogleSpeechToTextStreamingAudioGenerator(object):
    def __init__(
        self,
        pcm: ByteString,
        word_timings: Sequence[float],
        chunk_size_bytes: int,
        chunk_size_ms: int,
        apply_delay: bool,
    ) -> None:
        self._pcm = pcm
        self._word_timings = word_timings
        self._chunk_size_bytes = chunk_size_bytes
        self._chunk_size_ms = chunk_size_ms
        self._apply_delay = apply_delay

        self._send_timings = []
        self._finished = False

    def stream_generator(self) -> Generator[speech.StreamingRecognizeRequest, Any, Any]:
        total_bytes = len(self._pcm)
        current_byte = 0
        current_audio_time = 0.0

        while current_byte < total_bytes and not self._finished:
            chunk = self._pcm[current_byte: current_byte +
                              self._chunk_size_bytes]
            chunk_end_time = current_audio_time + (self._chunk_size_ms / 1000)

            send_time = time.time()

            yield speech.StreamingRecognizeRequest(audio_content=chunk)

            for word_time in self._word_timings:
                if current_audio_time < word_time <= chunk_end_time:
                    self._send_timings.append(send_time)

            if self._apply_delay:
                time.sleep(self._chunk_size_ms / 1000)

            current_audio_time = chunk_end_time
            current_byte += self._chunk_size_bytes

    def stop(self) -> None:
        self._finished = True


class GoogleSpeechToTextStreamingHandler(object):
    def __init__(self, ignore_punctuation: bool) -> None:
        self._emitted_words = []
        self._receive_timings = []
        self._last_word_index = 0

        self._ignore_punctuation = ignore_punctuation
        self._punctuation_trans = str.maketrans({".": "", ",": "", "?": ""})

    def _process_result(self, result) -> None:
        current_time = time.time()

        if not result.alternatives:
            return

        transcript = result.alternatives[0].transcript
        if not transcript:
            return

        if self._ignore_punctuation:
            words = transcript.translate(self._punctuation_trans).split()
        else:
            words = transcript.split()

        partial_transcript_reset = len(words) < self._last_word_index
        if partial_transcript_reset:
            self._last_word_index = 0

        if self._last_word_index > 0:
            last_emitted_word_changed = self._emitted_words[-1] != words[self._last_word_index - 1]
            if last_emitted_word_changed:
                self._emitted_words[-1] = words[self._last_word_index - 1]
                self._receive_timings[-1] = current_time

        if len(words) > self._last_word_index:
            new_words = words[self._last_word_index:]
            for word in new_words:
                self._emitted_words.append(word)
                self._receive_timings.append(current_time)

            self._last_word_index = len(words)


class IBMWatsonSpeechToTextEngine(Engine):
    def __init__(
        self,
        watson_speech_to_text_api_key: str,
        watson_speech_to_text_url: str,
        language: Languages,
        no_cache: bool = False,
    ):
        super().__init__(no_cache=no_cache)
        if language != Languages.EN:
            raise ValueError(
                "IBM_WATSON_SPEECH_TO_TEXT engine only supports EN language")

        self._service = SpeechToTextV1(
            authenticator=IAMAuthenticator(watson_speech_to_text_api_key))
        self._service.set_service_url(watson_speech_to_text_url)

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".ibm")
        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res

        with open(path, "rb") as f:
            response = self._service.recognize(
                audio=f,
                content_type="audio/flac",
                smart_formatting=True,
                end_of_phrase_silence_time=15,
            ).get_result()

        res = ""
        if response and ("results" in response) and response["results"]:
            res = response["results"][0]["alternatives"][0]["transcript"]

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "IBM Watson Speech-to-Text"


class Whisper(Engine):
    LANGUAGE_TO_WHISPER_CODE = {
        Languages.EN: "en",
        Languages.DE: "de",
        Languages.ES: "es",
        Languages.FR: "fr",
        Languages.IT: "it",
        Languages.PT_PT: "pt",
        Languages.PT_BR: "pt",
        Languages.ZH: "zh",
    }

    def __init__(self, cache_extension: str, model: str, language: Languages, no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        # Use MPS if available (macOS Apple Silicon), otherwise CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self._device = "mps"
        else:
            self._device = "cpu"
        print(f"Using {self._device} device")
        self._model = whisper.load_model(model, device=self._device)
        self._cache_extension = cache_extension
        self._language_code = self.LANGUAGE_TO_WHISPER_CODE[language]
        self._audio_sec = 0.0
        self._proc_sec = 0.0

    def transcribe(self, path: str) -> str:
        audio, sample_rate = soundfile.read(path, dtype="int16")
        assert sample_rate == SAMPLE_RATE
        self._audio_sec += audio.size / sample_rate

        cache_path = path.replace(".flac", self._cache_extension)
        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                res = f.read()
            return res

        start_sec = time.time()
        # MPS requires fp16=False to avoid NaN errors
        res = self._model.transcribe(
            path, language=self._language_code, fp16=(self._device != "mps"))["text"]
        self._proc_sec += time.time() - start_sec

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return self._audio_sec

    def process_sec(self) -> float:
        return self._proc_sec

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        raise NotImplementedError()


class WhisperTiny(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        model = "tiny.en" if language == Languages.EN else "tiny"
        super().__init__(cache_extension=".wspt", model=model,
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Tiny"


class WhisperBase(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        model = "base.en" if language == Languages.EN else "base"
        super().__init__(cache_extension=".wspb", model=model,
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Base"


class WhisperSmall(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        model = "small.en" if language == Languages.EN else "small"
        super().__init__(cache_extension=".wsps", model=model,
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Small"


class WhisperMedium(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        model = "medium.en" if language == Languages.EN else "medium"
        super().__init__(cache_extension=".wspm", model=model,
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Medium"


class WhisperLarge(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        super().__init__(cache_extension=".wspl", model="large-v1",
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Large-v1"


class WhisperLargeV2(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        super().__init__(cache_extension=".wspl2", model="large-v2",
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Large-v2"


class WhisperLargeV3(Whisper):
    def __init__(self, language: Languages, no_cache: bool = False):
        super().__init__(cache_extension=".wspl3", model="large-v3",
                         language=language, no_cache=no_cache)

    def __str__(self) -> str:
        return "Whisper Large-v3"


class PicovoiceCheetahEngine(StreamingEngine):
    def __init__(
        self,
        access_key: str,
        model_path: Optional[str],
        library_path: Optional[str],
        punctuation: bool = False,
        no_cache: bool = False,
    ):
        super().__init__(no_cache=no_cache)
        self._cheetah = pvcheetah.create(
            access_key=access_key,
            model_path=model_path,
            library_path=library_path,
            enable_automatic_punctuation=punctuation,
        )
        self._audio_sec = 0.0
        self._proc_sec = 0.0

    @property
    def is_async(self) -> bool:
        return False

    def transcribe(self, path: str) -> str:
        audio, sample_rate = soundfile.read(path, dtype="int16")
        assert sample_rate == self._cheetah.sample_rate
        self._audio_sec += audio.size / sample_rate

        start_sec = time.time()
        res = ""
        for i in range(audio.size // self._cheetah.frame_length):
            partial, _ = self._cheetah.process(
                audio[i *
                      self._cheetah.frame_length: (i + 1) * self._cheetah.frame_length]
            )
            res += partial
        res += self._cheetah.flush()
        self._proc_sec += time.time() - start_sec

        return res

    def _measure_word_latency(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        pcm, sample_rate = soundfile.read(path, dtype="int16")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Incorrect sample rate for `{path}`: expected {SAMPLE_RATE} got {sample_rate}")

        send_timings = [aln[-1]
                        for aln in alignments] if alignments is not None else []

        emitted_words = []
        receive_timings = []
        for i in range(pcm.size // self._cheetah.frame_length):
            partial, _ = self._cheetah.process(
                pcm[i *
                    self._cheetah.frame_length: (i + 1) * self._cheetah.frame_length]
            )

            if len(partial) > 0:
                words = partial.split()
                emitted_words.extend(words)

                end_sec = ((i + 1) * self._cheetah.frame_length) / SAMPLE_RATE
                receive_timings.extend([end_sec] * len(words))

        flushed_words = self._cheetah.flush()
        if len(flushed_words) > 0:
            words = flushed_words.split()
            emitted_words.extend(words)
            receive_timings.extend([pcm.size / SAMPLE_RATE] * len(words))

        return emitted_words, receive_timings, send_timings

    def audio_sec(self) -> float:
        return self._audio_sec

    def process_sec(self) -> float:
        return self._proc_sec

    def delete(self) -> None:
        self._cheetah.delete()

    def __str__(self) -> str:
        return "Picovoice Cheetah"


class PicovoiceLeopardEngine(Engine):
    def __init__(
        self,
        access_key: str,
        model_path: Optional[str],
        library_path: Optional[str],
        punctuation: bool = False,
        no_cache: bool = False,
    ):
        super().__init__(no_cache=no_cache)
        self._leopard = pvleopard.create(
            access_key=access_key,
            model_path=model_path,
            library_path=library_path,
            enable_automatic_punctuation=punctuation,
        )
        self._audio_sec = 0.0
        self._proc_sec = 0.0

    def transcribe(self, path: str) -> str:
        audio, sample_rate = soundfile.read(path, dtype="int16")
        assert sample_rate == self._leopard.sample_rate
        self._audio_sec += audio.size / sample_rate

        start_sec = time.time()
        res = self._leopard.process(audio)
        self._proc_sec += time.time() - start_sec

        return res[0]

    def audio_sec(self) -> float:
        return self._audio_sec

    def process_sec(self) -> float:
        return self._proc_sec

    def delete(self) -> None:
        self._leopard.delete()

    def __str__(self):
        return "Picovoice Leopard"


class SonioxAsyncEngine(Engine):
    LANGUAGE_TO_SONIOX_CODE = {
        Languages.EN: "en",
        Languages.DE: "de",
        Languages.ES: "es",
        Languages.FR: "fr",
        Languages.IT: "it",
        Languages.PT_PT: "pt",
        Languages.PT_BR: "pt",
        Languages.ZH: "zh",
    }

    API_BASE_URL = "https://api.soniox.com/v1"
    MODEL = "stt-async-preview"
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 2.0

    def __init__(self, soniox_api_key: str, language: Languages, no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        self._api_key = soniox_api_key
        self._language_code = self.LANGUAGE_TO_SONIOX_CODE[language]
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with exponential backoff for rate limits."""
        backoff = self.INITIAL_BACKOFF
        for attempt in range(self.MAX_RETRIES):
            response = requests.request(method, url, **kwargs)
            if response.ok:
                return response
            # Check for rate limit error
            if response.status_code == 400 or response.status_code == 429:
                try:
                    error_data = response.json()
                    if error_data.get("error_type") == "limit_exceeded":
                        if attempt < self.MAX_RETRIES - 1:
                            time.sleep(backoff)
                            backoff *= 2
                            continue
                except (ValueError, KeyError):
                    pass
            # For non-rate-limit errors, raise immediately
            return response
        return response

    def _upload_file(self, path: str) -> str:
        """Upload a local audio file and return the file_id."""
        with open(path, "rb") as f:
            file_content = f.read()
        files = {"file": (os.path.basename(path), file_content)}
        response = self._request_with_retry(
            "POST",
            f"{self.API_BASE_URL}/files",
            headers=self._headers,
            files=files,
        )
        if not response.ok:
            raise RuntimeError(
                f"Soniox file upload failed: {response.status_code} - {response.text}")
        data = response.json()
        # Handle both possible key names from API
        file_id = data.get("fileId") or data.get("file_id") or data.get("id")
        if not file_id:
            raise RuntimeError(
                f"Soniox file upload response missing file ID. Response: {data}")
        return file_id

    def _create_transcription(self, file_id: str) -> str:
        """Create a transcription job and return the transcription ID."""
        payload = {
            "model": self.MODEL,
            "file_id": file_id,
            "language_hints": [self._language_code, "en"],
        }
        response = self._request_with_retry(
            "POST",
            f"{self.API_BASE_URL}/transcriptions",
            headers=self._headers,
            json=payload,
        )
        if not response.ok:
            raise RuntimeError(
                f"Soniox create transcription failed: {response.status_code} - {response.text}")
        data = response.json()
        transcription_id = data.get("id")
        if not transcription_id:
            raise RuntimeError(
                f"Soniox create transcription response missing ID. Response: {data}")
        return transcription_id

    def _get_transcription_status(self, transcription_id: str) -> dict:
        """Get the status of a transcription job."""
        response = self._request_with_retry(
            "GET",
            f"{self.API_BASE_URL}/transcriptions/{transcription_id}",
            headers=self._headers,
        )
        if not response.ok:
            raise RuntimeError(
                f"Soniox get status failed: {response.status_code} - {response.text}")
        return response.json()

    def _get_transcript(self, transcription_id: str) -> str:
        """Get the transcript text for a completed transcription."""
        response = self._request_with_retry(
            "GET",
            f"{self.API_BASE_URL}/transcriptions/{transcription_id}/transcript",
            headers=self._headers,
        )
        if not response.ok:
            raise RuntimeError(
                f"Soniox get transcript failed: {response.status_code} - {response.text}")
        data = response.json()
        text = data.get("text", "")
        return text

    def transcribe(self, path: str) -> str:
        cache_path = os.path.splitext(path)[0] + ".snx"

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res

        # Upload file
        file_id = self._upload_file(path)

        # Create transcription job
        transcription_id = self._create_transcription(file_id)

        # Poll for completion
        while True:
            status_response = self._get_transcription_status(transcription_id)
            status = status_response["status"]

            if status == "completed":
                break
            elif status == "error":
                error_msg = status_response.get(
                    "error_message", "Unknown error")
                raise RuntimeError(
                    f"Soniox transcription {transcription_id} failed: {error_msg}")

            time.sleep(1)

        # Get transcript
        res = self._get_transcript(transcription_id)

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Soniox"


class SonioxRealtimeEngine(StreamingEngine):
    SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
    MODEL = "stt-rt-v3"

    def __init__(
        self,
        language: Languages,
        chunk_size_ms: int = 120,
        apply_delay: bool = False,
        ignore_punctuation: bool = False,
        no_cache: bool = False,
    ) -> None:
        super().__init__(no_cache=no_cache)
        from dotenv import load_dotenv
        load_dotenv()

        self._api_key = os.environ.get("SONIOX_API_KEY")
        if not self._api_key:
            raise ValueError("SONIOX_API_KEY must be set in .env file")

        self._language_code = SonioxAsyncEngine.LANGUAGE_TO_SONIOX_CODE[language]
        self._chunk_size_ms = chunk_size_ms
        self._apply_delay = apply_delay
        self._ignore_punctuation = ignore_punctuation

    @property
    def is_async(self) -> bool:
        return False

    def get_chunk_size_ms(self) -> int:
        return self._chunk_size_ms

    def load_pcm(self, path: str) -> ByteString:
        import numpy as np
        from scipy import signal

        pcm, sample_rate = soundfile.read(path, dtype="int16")
        if sample_rate != SAMPLE_RATE:
            num_samples = int(len(pcm) * SAMPLE_RATE / sample_rate)
            pcm = signal.resample(pcm, num_samples).astype(np.int16)
        return pcm.tobytes()

    def _measure_word_latency(
        self, path: str, alignments: Optional[Sequence[Tuple[float, float]]]
    ) -> WordLatencyOutputType:
        from websockets.sync.client import connect
        from websockets import ConnectionClosedOK

        cache_path = os.path.splitext(path)[0] + ".snxrt"

        if not self._no_cache and alignments is None and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res.split(), [], []

        pcm = self.load_pcm(path)
        chunk_size_bytes = self.get_chunk_size_bytes()

        send_timings = []
        if alignments is not None:
            word_end_times = [aln[-1] for aln in alignments]
        else:
            word_end_times = []

        final_tokens_text = []  # Raw token texts with spacing for proper transcript
        emitted_words = []      # Individual words for latency tracking
        receive_timings = []
        streaming_done = threading.Event()
        error_message = None

        def stream_audio(ws) -> None:
            nonlocal error_message
            try:
                total_bytes = len(pcm)
                current_byte = 0
                current_audio_time = 0.0
                chunk_duration_sec = self._chunk_size_ms / 1000.0

                while current_byte < total_bytes:
                    chunk = pcm[current_byte: current_byte + chunk_size_bytes]
                    chunk_end_time = current_audio_time + chunk_duration_sec

                    send_time = time.time()
                    ws.send(chunk)

                    for word_time in word_end_times:
                        if current_audio_time < word_time <= chunk_end_time:
                            send_timings.append(send_time)

                    time.sleep(chunk_duration_sec)

                    current_audio_time = chunk_end_time
                    current_byte += chunk_size_bytes

                ws.send("")
            except Exception as e:
                error_message = str(e)
            finally:
                streaming_done.set()

        # Include English as secondary language for Chinese (code-switching support)
        if self._language_code == "zh":
            language_hints = ["zh", "en"]
        else:
            language_hints = [self._language_code]

        config = {
            "api_key": self._api_key,
            "model": self.MODEL,
            "language_hints": language_hints,
            "audio_format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
        }

        with connect(self.SONIOX_WEBSOCKET_URL) as ws:
            ws.send(json.dumps(config))

            audio_thread = threading.Thread(
                target=stream_audio,
                args=(ws,),
                daemon=True,
            )
            audio_thread.start()

            try:
                while True:
                    message = ws.recv()
                    res = json.loads(message)

                    if res.get("error_code") is not None:
                        raise RuntimeError(
                            f"Soniox error: {res['error_code']} - {res.get('error_message', 'Unknown error')}")

                    receive_time = time.time()
                    for token in res.get("tokens", []):
                        if token.get("is_final"):
                            raw_text = token.get("text", "")
                            final_tokens_text.append(raw_text)

                            # Extract words for latency tracking
                            stripped = raw_text.strip()
                            if stripped:
                                words = stripped.split()
                                emitted_words.extend(words)
                                receive_timings.extend([receive_time] * len(words))

                    if res.get("finished"):
                        break

            except ConnectionClosedOK:
                pass

            audio_thread.join(timeout=5.0)

        if error_message:
            raise RuntimeError(f"Soniox streaming error: {error_message}")

        if alignments is None:
            with open(cache_path, "w") as f:
                f.write("".join(final_tokens_text).strip())

        return emitted_words, receive_timings, send_timings

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Soniox Real-time"


class DeepgramEngine(Engine):
    LANGUAGE_TO_DEEPGRAM_CODE = {
        Languages.EN: "en",
        Languages.DE: "de",
        Languages.ES: "es",
        Languages.FR: "fr",
        Languages.IT: "it",
        Languages.PT_PT: "pt",
        Languages.PT_BR: "pt",
        Languages.ZH: "zh",
    }

    def __init__(self, deepgram_api_key: str, language: Languages, no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        self._client = DeepgramClient(api_key=deepgram_api_key)
        self._language_code = self.LANGUAGE_TO_DEEPGRAM_CODE[language]
        # nova-3 doesn't support Chinese yet, use nova-2 for Chinese
        self._model = "nova-2" if language == Languages.ZH else "nova-3"

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".dg")

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                res = f.read()
            return res

        with open(path, "rb") as audio_file:
            audio_data = audio_file.read()

        try:
            response = self._client.listen.v1.media.transcribe_file(
                request=audio_data,
                model=self._model,
                language=self._language_code,
            )
        except DeepgramApiError as e:
            # Re-raise as RuntimeError so it can be pickled across process boundaries
            raise RuntimeError(f"Deepgram API error: {e}") from None

        res = ""
        if response and response.results and response.results.channels:
            alternatives = response.results.channels[0].alternatives
            if alternatives:
                res = alternatives[0].transcript

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Deepgram"


class ElevenLabsEngine(Engine):
    LANGUAGE_TO_ELEVENLABS_CODE = {
        Languages.EN: "en",
        Languages.DE: "de",
        Languages.ES: "es",
        Languages.FR: "fr",
        Languages.IT: "it",
        Languages.PT_PT: "pt",
        Languages.PT_BR: "pt",
        Languages.ZH: "zho",
    }

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
    MODEL = "scribe_v1"

    def __init__(self, elevenlabs_api_key: str, language: Languages):
        self._api_key = elevenlabs_api_key
        self._language_code = self.LANGUAGE_TO_ELEVENLABS_CODE[language]

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".el")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        with open(path, "rb") as audio_file:
            files = {"file": (os.path.basename(path), audio_file)}
            data = {
                "model_id": self.MODEL,
                "language_code": self._language_code,
            }
            headers = {"xi-api-key": self._api_key}
            response = requests.post(
                self.API_URL, headers=headers, files=files, data=data)

        if not response.ok:
            raise RuntimeError(
                f"ElevenLabs transcription failed: {response.status_code} - {response.text}")

        res = response.json().get("text", "")

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "ElevenLabs"


class DashscopeEngine(Engine):
    LANGUAGE_TO_DASHSCOPE_CODE = {
        Languages.EN: "en",
        Languages.DE: "de",
        Languages.ES: "es",
        Languages.FR: "fr",
        Languages.IT: "it",
        Languages.PT_PT: "pt",
        Languages.PT_BR: "pt",
        Languages.ZH: "zh",
    }

    def __init__(self, dashscope_api_key: str, language: Languages, no_cache: bool = False):
        super().__init__(no_cache)
        self._api_key = dashscope_api_key
        self._language_code = self.LANGUAGE_TO_DASHSCOPE_CODE[language]

    def transcribe(self, path: str) -> str:
        cache_path = path.replace(".flac", ".ds")

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        # Read audio file and encode as base64
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        # Create data URI in the format the API expects
        data_uri = f"data:audio/flac;base64,{audio_b64}"

        # Set international API endpoint (use dashscope.aliyuncs.com for China region)
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

        response = dashscope.MultiModalConversation.call(
            api_key=self._api_key,
            model="qwen3-asr-flash",
            messages=[
                {"role": "system", "content": [{"text": ""}]},
                {"role": "user", "content": [{"audio": data_uri}]}
            ],
            result_format="message",
            asr_options={
                "language": self._language_code
            }
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Dashscope transcription failed: {response.status_code} - {response.message}")

        res = response["output"]["choices"][0]["message"]["content"][0]["text"]

        with open(cache_path, "w") as f:
            f.write(res)

        return res

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "Dashscope"


class IflyrecEngine(Engine):
    """iFlyRec Global (xfyun.cn international) speech-to-text engine."""

    LANGUAGE_TO_IFLYREC = {
        Languages.ZH: ("zh_cn", "mandarin"),
        Languages.EN: ("en_us", "mandarin"),
    }

    WS_URL = "wss://iat-api-sg.xf-yun.com/v2/iat"
    HOST = "iat-api-sg.xf-yun.com"
    MAX_AUDIO_SECONDS = 60

    STATUS_FIRST_FRAME = 0
    STATUS_CONTINUE_FRAME = 1
    STATUS_LAST_FRAME = 2

    def __init__(self, language: Languages, no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        from dotenv import load_dotenv
        load_dotenv()

        self._app_id = os.environ.get("IFLYREC_APP_ID")
        self._api_key = os.environ.get("IFLYREC_API_KEY")
        self._api_secret = os.environ.get("IFLYREC_API_SECRET")

        if not all([self._app_id, self._api_key, self._api_secret]):
            raise ValueError(
                "IFLYREC_APP_ID, IFLYREC_API_KEY, and IFLYREC_API_SECRET must be set in .env file")

        if language not in self.LANGUAGE_TO_IFLYREC:
            raise ValueError(
                f"IFLYREC engine does not support language: {language}")

        self._language, self._accent = self.LANGUAGE_TO_IFLYREC[language]

    def _create_auth_url(self) -> str:
        """Create authenticated WebSocket URL with HMAC-SHA256 signature."""
        import hashlib
        import hmac
        from wsgiref.handlers import format_date_time
        from datetime import datetime
        from time import mktime
        from urllib.parse import urlencode

        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: {self.HOST}\n"
        signature_origin += f"date: {date}\n"
        signature_origin += "GET /v2/iat HTTP/1.1"

        signature_sha = hmac.new(
            self._api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(
            signature_sha).decode(encoding='utf-8')

        authorization_origin = (
            f'api_key="{self._api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(
            authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        params = {
            "authorization": authorization,
            "date": date,
            "host": self.HOST
        }
        return f"{self.WS_URL}?{urlencode(params)}"

    def _convert_flac_to_pcm(self, flac_path: str) -> tuple:
        """Convert FLAC file to 16-bit PCM bytes. Returns (pcm_bytes, sample_rate)."""
        import numpy as np
        audio, sample_rate = soundfile.read(flac_path, dtype="int16")
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Unsupported sample rate for `{flac_path}`: got {sample_rate}, expected 8000 or 16000")
        return audio.astype(np.int16).tobytes(), sample_rate

    def transcribe(self, path: str) -> str:
        import ssl
        import websocket
        import threading

        cache_path = path.replace(".flac", ".ifly")

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        audio_info = soundfile.info(path)
        if audio_info.duration > self.MAX_AUDIO_SECONDS:
            print(f"Warning: Audio file {path} is {audio_info.duration:.1f}s, "
                  f"exceeds {self.MAX_AUDIO_SECONDS}s limit. Skipping.")
            return ""

        pcm_data, sample_rate = self._convert_flac_to_pcm(path)
        audio_format = f"audio/L16;rate={sample_rate}"
        ws_url = self._create_auth_url()

        # Use WebSocketApp with callbacks for concurrent send/receive
        result_parts = []
        error_info = {"code": None, "message": None}
        done_event = threading.Event()

        # Frame size and interval per official iFlytek demo
        frame_size = 8000  # 8KB per frame
        interval = 0.04  # 40ms between frames

        def on_message(ws, message):
            try:
                msg = json.loads(message)
                code = msg.get("code", 0)
                if code != 0:
                    error_info["code"] = code
                    error_info["message"] = msg.get("message", "Unknown error")
                    done_event.set()
                    return

                data = msg.get("data", {})
                result = data.get("result") or {}
                ws_data = result.get("ws") or []

                for item in ws_data:
                    for cw in item.get("cw", []):
                        word = cw.get("w", "")
                        if word:
                            result_parts.append(word)

                if data.get("status") == 2:
                    done_event.set()
            except Exception as e:
                error_info["code"] = -1
                error_info["message"] = str(e)
                done_event.set()

        def on_error(ws, error):
            error_info["code"] = -1
            error_info["message"] = str(error)
            done_event.set()

        def on_close(ws, close_status_code, close_msg):
            done_event.set()

        def on_open(ws):
            def send_audio():
                try:
                    offset = 0
                    total_len = len(pcm_data)
                    status = self.STATUS_FIRST_FRAME

                    while offset < total_len:
                        chunk = pcm_data[offset:offset + frame_size]
                        if not chunk:
                            status = self.STATUS_LAST_FRAME
                        audio_b64 = base64.b64encode(chunk).decode('utf-8')

                        if status == self.STATUS_FIRST_FRAME:
                            frame_data = {
                                "common": {"app_id": self._app_id},
                                "business": {
                                    "domain": "iat",
                                    "language": "mq_cbm",
                                    "accent": self._accent,
                                    "vad_eos": 10000,
                                    "ptt": 1,
                                },
                                "data": {
                                    "status": 0,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }
                            status = self.STATUS_CONTINUE_FRAME
                        elif status == self.STATUS_CONTINUE_FRAME:
                            frame_data = {
                                "data": {
                                    "status": 1,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }
                        else:  # STATUS_LAST_FRAME
                            frame_data = {
                                "data": {
                                    "status": 2,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }

                        ws.send(json.dumps(frame_data))
                        offset += frame_size

                        # Check if this was the last chunk
                        if offset >= total_len:
                            # Send final frame with status 2
                            if status != self.STATUS_LAST_FRAME:
                                final_frame = {
                                    "data": {
                                        "status": 2,
                                        "format": audio_format,
                                        "audio": "",
                                        "encoding": "raw"
                                    }
                                }
                                ws.send(json.dumps(final_frame))
                            break

                        # 40ms interval between frames per iFlytek spec
                        time.sleep(interval)

                except Exception as e:
                    error_info["code"] = -1
                    error_info["message"] = f"Send error: {e}"
                    done_event.set()

            # Start sender thread
            threading.Thread(target=send_audio, daemon=True).start()

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Run WebSocket in a separate thread with timeout
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}),
            daemon=True
        )
        ws_thread.start()

        # Wait for completion with timeout (60 seconds max for 60 second audio)
        done_event.wait(timeout=120)
        ws.close()

        if error_info["code"] is not None:
            raise RuntimeError(
                f"iFlyRec API error {error_info['code']}: {error_info['message']}")

        result = "".join(result_parts)

        with open(cache_path, "w") as f:
            f.write(result)

        return result

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "iFlyRec"


class IflyrecBatchEngine(Engine):
    """iFlyRec Global Speed Transcription (OST) engine for batch processing.

    Supports long audio files (up to 5 hours) with Chinese-English codeswitching.
    Uses HTTP-based upload → create task → poll for results workflow.
    """

    UPLOAD_HOST = "sgw-gp.xf-yun.com"
    UPLOAD_URL = "https://sgw-gp.xf-yun.com/api/v1/spost"
    OST_HOST = "ost-api-sg.xf-yun.com"
    CREATE_URL = "https://ost-api-sg.xf-yun.com/v2/ost/create"
    QUERY_URL = "https://ost-api-sg.xf-yun.com/v2/ost/query"

    # Task types for different languages
    # Chinese task type supports Chinese-English codeswitching by default
    LANGUAGE_TO_TASK_TYPE = {
        Languages.ZH: "iflyrec_voice_cn_10m_ed",
    }

    POLL_INTERVAL = 5  # seconds between status checks
    MAX_POLL_TIME = 3600  # max 1 hour for very long files

    def __init__(self, language: Languages, no_cache: bool = False):
        super().__init__(no_cache=no_cache)
        from dotenv import load_dotenv
        load_dotenv()

        self._app_id = os.environ.get("IFLYREC_APP_ID")
        self._api_key = os.environ.get("IFLYREC_API_KEY")
        self._api_secret = os.environ.get("IFLYREC_API_SECRET")

        if not all([self._app_id, self._api_key, self._api_secret]):
            raise ValueError(
                "IFLYREC_APP_ID, IFLYREC_API_KEY, and IFLYREC_API_SECRET must be set in .env file")

        if language not in self.LANGUAGE_TO_TASK_TYPE:
            raise ValueError(
                f"IFLYREC_BATCH engine does not support language: {language}. "
                f"Supported: {list(self.LANGUAGE_TO_TASK_TYPE.keys())}")

        self._language = language
        self._task_type = self.LANGUAGE_TO_TASK_TYPE[language]

    def _create_upload_auth_headers(self, url: str, body: bytes) -> dict:
        """Create authentication headers for file upload endpoint."""
        import hashlib
        import hmac
        from wsgiref.handlers import format_date_time
        from datetime import datetime
        from time import mktime
        from urllib.parse import urlparse

        # Body digest
        body_digest = hashlib.sha256(body).digest()
        body_sign = "SHA256=" + base64.b64encode(body_digest).decode('utf-8')

        # Parse URL
        u = urlparse(url)
        host = u.hostname

        # Date
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Signature string
        request_line = f"POST {u.path} HTTP/1.1"
        sign_str = f"host: {host}\ndate: {date}\n{request_line}\ndigest: {body_sign}"

        # HMAC signature
        signature = hmac.new(
            self._api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        authorization = (
            f'api_key="{self._api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line digest", signature="{signature_b64}"'
        )

        return {
            "host": host,
            "date": date,
            "authorization": authorization,
            "digest": body_sign,
            "Content-Length": str(len(body)),
            "X-TTL": "100",
        }

    def _create_ost_auth_headers(self, uri: str, body: str) -> dict:
        """Create authentication headers for OST create/query endpoints."""
        import hashlib
        import hmac
        from wsgiref.handlers import format_date_time
        from datetime import datetime
        from time import mktime

        # Body digest (SHA-256)
        body_bytes = body.encode('utf-8')
        body_hash = hashlib.sha256(body_bytes).digest()
        digest = "SHA-256=" + base64.b64encode(body_hash).decode('utf-8')

        # Date
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Signature string
        sign_str = f"host: {self.OST_HOST}\ndate: {date}\nPOST {uri} HTTP/1.1\ndigest: {digest}"

        # HMAC signature
        signature = hmac.new(
            self._api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        authorization = (
            f'api_key="{self._api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line digest", signature="{signature_b64}"'
        )

        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Host": self.OST_HOST,
            "Date": date,
            "Digest": digest,
            "Authorization": authorization,
        }

    def _upload_file(self, path: str) -> str:
        """Upload audio file and return the file URL."""
        with open(path, 'rb') as f:
            body = f.read()

        url = f"{self.UPLOAD_URL}?get_link=true&link_ttl=3600&split_host=true"
        headers = self._create_upload_auth_headers(url, body)

        response = requests.post(url, data=body, headers=headers, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(
                f"iFlyRec file upload failed: {response.status_code} - {response.text}")

        data = response.json()
        link_path = data.get("data", {}).get("link_path")
        if not link_path:
            raise RuntimeError(
                f"iFlyRec file upload response missing link_path: {data}")

        return f"https://sgw-gp.xf-yun.com{link_path}"

    def _create_task(self, file_url: str, sample_rate: int) -> str:
        """Create transcription task and return task_id."""
        # Determine audio format based on sample rate
        audio_format = f"audio/L16;rate={sample_rate}"

        body = {
            "common": {"app_id": self._app_id},
            "business": {
                "task_type": self._task_type,
                "request_id": str(uuid.uuid4()),
                "pd": "edu",
                "vspp_on": 1,
                "smoothproc": True,
                "colloqproc": False,
            },
            "data": {
                "audio_src": "http",
                "audio_url": file_url,
                "format": audio_format,
                "encoding": "raw"
            }
        }
        body_str = json.dumps(body)
        headers = self._create_ost_auth_headers("/v2/ost/create", body_str)

        response = requests.post(
            self.CREATE_URL, data=body_str, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"iFlyRec create task failed: {response.status_code} - {response.text}")

        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(
                f"iFlyRec create task error: {data.get('message', 'Unknown error')}")

        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(
                f"iFlyRec create task response missing task_id: {data}")

        return task_id

    def _query_task(self, task_id: str) -> dict:
        """Query task status and return response data."""
        body = {
            "common": {"app_id": self._app_id},
            "business": {"task_id": task_id},
        }
        body_str = json.dumps(body)
        headers = self._create_ost_auth_headers("/v2/ost/query", body_str)

        response = requests.post(
            self.QUERY_URL, data=body_str, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"iFlyRec query task failed: {response.status_code} - {response.text}")

        return response.json()

    def _extract_transcript(self, result: dict) -> str:
        """Extract transcript text from OST result structure."""
        text_parts = []

        lattice = result.get("data", {}).get("result", {}).get("lattice", [])
        for item in lattice:
            json_1best = item.get("json_1best", {})
            st = json_1best.get("st", {})
            for rt in st.get("rt", []):
                for ws in rt.get("ws", []):
                    for cw in ws.get("cw", []):
                        word = cw.get("w", "")
                        if word:
                            text_parts.append(word)

        return "".join(text_parts)

    def _convert_to_pcm(self, path: str) -> tuple:
        """Convert audio file to 16-bit PCM and return (pcm_path, sample_rate).

        The OST API expects raw PCM data, so we convert the input file.
        Returns a tuple of (temporary_pcm_path, sample_rate).
        """
        import tempfile
        import numpy as np

        audio, sample_rate = soundfile.read(path, dtype="int16")

        # Create temp file for PCM data
        fd, pcm_path = tempfile.mkstemp(suffix=".pcm")
        os.close(fd)

        audio.astype(np.int16).tofile(pcm_path)

        return pcm_path, sample_rate

    def transcribe(self, path: str) -> str:
        cache_ext = ".iflybatch"
        cache_path = os.path.splitext(path)[0] + cache_ext

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        # Convert audio to PCM format
        pcm_path, sample_rate = self._convert_to_pcm(path)

        try:
            # Upload file
            file_url = self._upload_file(pcm_path)

            # Create transcription task
            task_id = self._create_task(file_url, sample_rate)

            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < self.MAX_POLL_TIME:
                result = self._query_task(task_id)

                if result.get("code") != 0:
                    raise RuntimeError(
                        f"iFlyRec query error: {result.get('message', 'Unknown error')}")

                task_status = result.get("data", {}).get("task_status")

                # Status: '1' = processing, '2' = queued, other values = complete
                # Success is determined by code=0, not by task_status value
                if task_status not in ('1', '2'):
                    break

                time.sleep(self.POLL_INTERVAL)

            # Extract transcript
            transcript = self._extract_transcript(result)

        finally:
            # Clean up temp PCM file
            if os.path.exists(pcm_path):
                os.remove(pcm_path)

        # Cache result
        with open(cache_path, "w") as f:
            f.write(transcript)

        return transcript

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return "iFlyRec Batch"


class IflyrecIstEngine(Engine):
    """iFlyRec IST (International Speech Transcription) engine.

    Supports two model types:
    - basic: Chinese-English bilingual (ist_hy domain)
    - llm: Multilingual auto-detect (ist_cbm_mix domain)
    """

    WS_URL = "wss://ist-api-sg.xf-yun.com/v2/ist"
    HOST = "ist-api-sg.xf-yun.com"
    MAX_AUDIO_SECONDS = 18000  # 5 hours

    # Model configurations
    MODEL_CONFIGS = {
        "basic": {
            "domain": "ist_hy",
            "language": "zh_en",
            "accent": "mandarin",
            "cache_ext": ".ist_basic",
        },
        "llm": {
            "domain": "ist_cbm_mix",
            "language": "mix",
            "accent": "mandarin",
            "cache_ext": ".ist_llm",
        },
    }

    STATUS_FIRST_FRAME = 0
    STATUS_CONTINUE_FRAME = 1
    STATUS_LAST_FRAME = 2

    def __init__(self, language: Languages, no_cache: bool = False, ist_model: str = "basic"):
        super().__init__(no_cache=no_cache)
        from dotenv import load_dotenv
        load_dotenv()

        self._app_id = os.environ.get("IFLYREC_APP_ID")
        self._api_key = os.environ.get("IFLYREC_API_KEY")
        self._api_secret = os.environ.get("IFLYREC_API_SECRET")

        if not all([self._app_id, self._api_key, self._api_secret]):
            raise ValueError(
                "IFLYREC_APP_ID, IFLYREC_API_KEY, and IFLYREC_API_SECRET must be set in .env file")

        if ist_model not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid ist_model: {ist_model}. Must be one of: {list(self.MODEL_CONFIGS.keys())}")

        self._ist_model = ist_model
        self._config = self.MODEL_CONFIGS[ist_model]
        self._language = language

    def _create_auth_url(self) -> str:
        """Create authenticated WebSocket URL with HMAC-SHA256 signature."""
        import hashlib
        import hmac
        from wsgiref.handlers import format_date_time
        from datetime import datetime
        from time import mktime
        from urllib.parse import urlencode

        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: {self.HOST}\n"
        signature_origin += f"date: {date}\n"
        signature_origin += "GET /v2/ist HTTP/1.1"

        signature_sha = hmac.new(
            self._api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(
            signature_sha).decode(encoding='utf-8')

        authorization_origin = (
            f'api_key="{self._api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(
            authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        params = {
            "authorization": authorization,
            "date": date,
            "host": self.HOST
        }
        return f"{self.WS_URL}?{urlencode(params)}"

    def _convert_flac_to_pcm(self, flac_path: str) -> tuple:
        """Convert FLAC file to 16-bit PCM bytes. Returns (pcm_bytes, sample_rate)."""
        import numpy as np
        audio, sample_rate = soundfile.read(flac_path, dtype="int16")
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Unsupported sample rate for `{flac_path}`: got {sample_rate}, expected 8000 or 16000")
        return audio.astype(np.int16).tobytes(), sample_rate

    def transcribe(self, path: str) -> str:
        import ssl
        import websocket
        import threading

        cache_ext = self._config["cache_ext"]
        cache_path = path.replace(".flac", cache_ext)

        if not self._no_cache and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        audio_info = soundfile.info(path)
        if audio_info.duration > self.MAX_AUDIO_SECONDS:
            print(f"Warning: Audio file {path} is {audio_info.duration:.1f}s, "
                  f"exceeds {self.MAX_AUDIO_SECONDS}s limit. Skipping.")
            return ""

        pcm_data, sample_rate = self._convert_flac_to_pcm(path)
        audio_format = f"audio/L16;rate={sample_rate}"
        ws_url = self._create_auth_url()

        # Use WebSocketApp with callbacks for concurrent send/receive
        result_parts = []
        error_info = {"code": None, "message": None}
        done_event = threading.Event()

        # Frame size and interval per official iFlytek spec
        frame_size = 1280  # 1280 bytes per frame (40ms of 16kHz 16-bit audio)
        interval = 0.04  # 40ms between frames

        config = self._config

        def on_message(ws, message):
            try:
                msg = json.loads(message)
                code = msg.get("code", 0)
                if code != 0:
                    error_info["code"] = code
                    error_info["message"] = msg.get("message", "Unknown error")
                    done_event.set()
                    return

                data = msg.get("data", {})
                result = data.get("result") or {}
                ws_data = result.get("ws") or []

                for item in ws_data:
                    for cw in item.get("cw", []):
                        word = cw.get("w", "")
                        if word:
                            result_parts.append(word)

                if data.get("status") == 2:
                    done_event.set()
            except Exception as e:
                error_info["code"] = -1
                error_info["message"] = str(e)
                done_event.set()

        def on_error(ws, error):
            error_info["code"] = -1
            error_info["message"] = str(error)
            done_event.set()

        def on_close(ws, close_status_code, close_msg):
            done_event.set()

        def on_open(ws):
            def send_audio():
                try:
                    offset = 0
                    total_len = len(pcm_data)
                    status = self.STATUS_FIRST_FRAME

                    while offset < total_len:
                        chunk = pcm_data[offset:offset + frame_size]
                        if not chunk:
                            status = self.STATUS_LAST_FRAME
                        audio_b64 = base64.b64encode(chunk).decode('utf-8')

                        if status == self.STATUS_FIRST_FRAME:
                            frame_data = {
                                "common": {"app_id": self._app_id},
                                "business": {
                                    "domain": config["domain"],
                                    "language": config["language"],
                                    "accent": config["accent"],
                                },
                                "data": {
                                    "status": 0,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }
                            status = self.STATUS_CONTINUE_FRAME
                        elif status == self.STATUS_CONTINUE_FRAME:
                            frame_data = {
                                "data": {
                                    "status": 1,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }
                        else:  # STATUS_LAST_FRAME
                            frame_data = {
                                "data": {
                                    "status": 2,
                                    "format": audio_format,
                                    "audio": audio_b64,
                                    "encoding": "raw"
                                }
                            }

                        ws.send(json.dumps(frame_data))
                        offset += frame_size

                        # Check if this was the last chunk
                        if offset >= total_len:
                            # Send final frame with status 2
                            if status != self.STATUS_LAST_FRAME:
                                final_frame = {
                                    "data": {
                                        "status": 2,
                                        "format": audio_format,
                                        "audio": "",
                                        "encoding": "raw"
                                    }
                                }
                                ws.send(json.dumps(final_frame))
                            break

                        # 40ms interval between frames per iFlytek spec
                        time.sleep(interval)

                except Exception as e:
                    error_info["code"] = -1
                    error_info["message"] = f"Send error: {e}"
                    done_event.set()

            # Start sender thread
            threading.Thread(target=send_audio, daemon=True).start()

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Run WebSocket in a separate thread with timeout
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}),
            daemon=True
        )
        ws_thread.start()

        # Wait for completion with timeout (generous for long audio)
        timeout = max(120, audio_info.duration * 2)
        done_event.wait(timeout=timeout)
        ws.close()

        if error_info["code"] is not None:
            raise RuntimeError(
                f"iFlyRec IST API error {error_info['code']}: {error_info['message']}")

        result = "".join(result_parts)

        with open(cache_path, "w") as f:
            f.write(result)

        return result

    def audio_sec(self) -> float:
        return -1.0

    def process_sec(self) -> float:
        return -1.0

    def delete(self) -> None:
        pass

    def __str__(self) -> str:
        return f"iFlyRec IST ({self._ist_model})"


__all__ = [
    "Engine",
    "Engines",
]
