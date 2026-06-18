#!/home/mustar/catkin_ws/src/room_guide/.venv/bin/python3
import os
import subprocess
import threading
import numpy as np
import sounddevice as sd
import torch
import whisper
from silero_vad import load_silero_vad, VADIterator
from openwakeword.model import Model as WakeWordModel

OWW_CHUNK  = 1280
VAD_CHUNK  = 512
BASE_BLOCK = 256


def env(key, default):
    return os.environ.get(key, default)


class WhisperNode:
    def __init__(self):
        self.output_topic        = env("OUTPUT_TOPIC",        "/llm_input")
        self.sample_rate         = int(env("SAMPLE_RATE",     16000))
        self.language            = env("LANGUAGE",            "")
        self.wakeword_name       = env("WAKEWORD_NAME",       "hey_jarvis")
        self.wakeword_threshold  = float(env("WAKEWORD_THRESHOLD",  0.5))
        self.vad_threshold       = float(env("VAD_THRESHOLD",       0.5))
        self.min_silence_ms      = int(env("MIN_SILENCE_MS",        600))
        self.no_speech_threshold = float(env("NO_SPEECH_THRESHOLD", 0.6))
        self.min_audio_duration  = float(env("MIN_AUDIO_DURATION",  0.5))
        model_size               = env("WHISPER_MODEL",       "small")

        torch.set_num_threads(1)

        print(f"[whisper] Loading wake word model…", flush=True)
        self.wakeword = WakeWordModel(inference_framework="onnx")

        print(f"[whisper] Loading Silero VAD…", flush=True)
        self.vad = VADIterator(
            load_silero_vad(onnx=False),
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
        )

        print(f"[whisper] Loading Whisper '{model_size}'…", flush=True)
        self.whisper = whisper.load_model(model_size)

        print(f"[whisper] Ready. Listening for '{self.wakeword_name}'…", flush=True)

    def _detected_wakeword(self, chunk: np.ndarray) -> bool:
        pcm = (chunk * 32768.0).astype(np.int16)
        scores = self.wakeword.predict(pcm)
        return scores.get(self.wakeword_name, 0.0) > self.wakeword_threshold

    def _reset_wakeword(self):
        silence = np.zeros(OWW_CHUNK, dtype=np.int16)
        for _ in range(40):
            self.wakeword.predict(silence)

    def _speech_ended(self, chunk: np.ndarray) -> bool:
        result = self.vad(torch.from_numpy(chunk))
        return result is not None and "end" in result

    def _transcribe(self, audio: np.ndarray) -> str | None:
        kwargs = {"language": self.language} if self.language else {}
        result = self.whisper.transcribe(audio, **kwargs)
        text   = result["text"].strip()

        segments = result.get("segments", [])
        if segments and text:
            avg_no_speech = np.mean([s["no_speech_prob"] for s in segments])
            if avg_no_speech > self.no_speech_threshold:
                print(f"[whisper] Hallucination filtered ({avg_no_speech:.2f}): '{text}'", flush=True)
                return None

        return text or None

    def _publish(self, text: str):
        print(f"[whisper] Publishing: {text}", flush=True)
        subprocess.run([
            "rostopic", "pub", "-1",
            self.output_topic,
            "std_msgs/String",
            f"data: '{text}'"
        ])

    def _transcribe_and_publish(self, audio: np.ndarray):
        try:
            text = self._transcribe(audio)
            if text:
                print(f"[whisper] Transcribed: {text}", flush=True)
                self._publish(text)
        except Exception as e:
            print(f"[whisper] Transcription error: {e}", flush=True)

    def run(self):
        state     = "WAKEWORD"
        recording = []
        oww_buf   = np.array([], dtype=np.float32)
        vad_buf   = np.array([], dtype=np.float32)

        min_samples = int(self.sample_rate * self.min_audio_duration)

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                            dtype="float32", blocksize=BASE_BLOCK) as stream:
            while True:
                block, _ = stream.read(BASE_BLOCK)
                audio    = block.flatten()

                if state == "WAKEWORD":
                    oww_buf = np.append(oww_buf, audio)
                    while len(oww_buf) >= OWW_CHUNK:
                        if self._detected_wakeword(oww_buf[:OWW_CHUNK]):
                            print("[whisper] Wake word detected!", flush=True)
                            state     = "RECORDING"
                            recording = []
                            vad_buf   = np.array([], dtype=np.float32)
                            self.vad.reset_states()
                            oww_buf   = np.array([], dtype=np.float32)
                            break
                        oww_buf = oww_buf[OWW_CHUNK:]

                elif state == "RECORDING":
                    recording.append(audio)
                    vad_buf = np.append(vad_buf, audio)

                    while len(vad_buf) >= VAD_CHUNK:
                        if self._speech_ended(vad_buf[:VAD_CHUNK]):
                            full_audio = np.concatenate(recording)

                            if len(full_audio) >= min_samples:
                                threading.Thread(
                                    target=self._transcribe_and_publish,
                                    args=(full_audio,),
                                    daemon=True,
                                ).start()
                            else:
                                print("[whisper] Utterance too short, ignoring.", flush=True)

                            self._reset_wakeword()
                            stream.stop()
                            stream.start()

                            state     = "WAKEWORD"
                            recording = []
                            oww_buf   = np.array([], dtype=np.float32)
                            vad_buf   = np.array([], dtype=np.float32)
                            break

                        vad_buf = vad_buf[VAD_CHUNK:]


if __name__ == "__main__":
    WhisperNode().run()
