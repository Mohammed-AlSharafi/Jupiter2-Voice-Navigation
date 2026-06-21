#!/home/mnajib/distrobox/ros_noetic_home/catkin_ws/src/room_guide/.venv/bin/python3
import os
import subprocess
import threading

import numpy as np
import sounddevice as sd
import torch
import whisper
from openwakeword.model import Model as WakeWordModel
from openwakeword import utils
from silero_vad import VADIterator, load_silero_vad

BASE_BLOCK = 256
OWW_CHUNK = 1280
VAD_CHUNK = 512


class WhisperNode:
    def __init__(self):
        self.output_topic = os.environ.get("OUTPUT_TOPIC", "/llm_input")
        self.sample_rate = int(os.environ.get("SAMPLE_RATE", "16000"))
        self.language = os.environ.get("LANGUAGE", "")
        self.wakeword_name = os.environ.get("WAKEWORD_NAME", "hey_jarvis")
        self.wakeword_threshold = float(os.environ.get("WAKEWORD_THRESHOLD", "0.5"))
        self.vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.5"))
        self.min_silence_ms = int(os.environ.get("MIN_SILENCE_MS", "600"))
        self.no_speech_threshold = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6"))
        self.min_audio_duration = float(os.environ.get("MIN_AUDIO_DURATION", "0.5"))

        torch.set_num_threads(1)

        print("[whisper] Loading wake word model…", flush=True)
        try:
            utils.download_models()
        except Exception:
            pass
        self.wakeword = WakeWordModel(inference_framework="onnx")

        print("[whisper] Loading Silero VAD…", flush=True)
        self.vad = VADIterator(
            load_silero_vad(onnx=False),
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
        )

        model_size = os.environ.get("WHISPER_MODEL", "small")
        print(f"[whisper] Loading Whisper '{model_size}'…", flush=True)
        self.whisper = whisper.load_model(model_size)

        print(f"[whisper] Ready. Listening for '{self.wakeword_name}'…", flush=True)

    def _detected_wakeword(self, chunk):
        pcm = (chunk * 32768.0).astype(np.int16)
        scores = self.wakeword.predict(pcm)
        return scores.get(self.wakeword_name, 0.0) > self.wakeword_threshold

    def _reset_wakeword(self):
        silence = np.zeros(OWW_CHUNK, dtype=np.int16)
        for _ in range(40):
            self.wakeword.predict(silence)

    def _speech_ended(self, chunk):
        result = self.vad(torch.from_numpy(chunk))
        return result is not None and "end" in result

    def _transcribe(self, audio):
        kwargs = {"language": self.language} if self.language else {}
        result = self.whisper.transcribe(audio, **kwargs)
        text = result["text"].strip()

        segments = result.get("segments", [])
        if segments and text:
            avg_no_speech = np.mean([s["no_speech_prob"] for s in segments])
            if avg_no_speech > self.no_speech_threshold:
                print(
                    f"[whisper] Hallucination filtered ({avg_no_speech:.2f}): '{text}'",
                    flush=True,
                )
                return None

        return text or None

    def _publish(self, text):
        print(f"[whisper] Publishing: {text}", flush=True)
        subprocess.run(
            [
                "rostopic", "pub", "-1",
                self.output_topic,
                "std_msgs/String",
                f"data: '{text}'",
            ],
            check=False,
        )

    def _transcribe_and_publish(self, audio):
        try:
            text = self._transcribe(audio)
            if text:
                print(f"[whisper] Transcribed: {text}", flush=True)
                self._publish(text)
        except Exception as e:
            print(f"[whisper] Transcription error: {e}", flush=True)

    def run(self):
        state = "LISTEN"
        recording = []
        oww_buf = np.array([], dtype=np.float32)
        vad_buf = np.array([], dtype=np.float32)
        min_samples = int(self.sample_rate * self.min_audio_duration)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=BASE_BLOCK,
        ) as stream:
            while True:
                block, _ = stream.read(BASE_BLOCK)
                audio = block.flatten()

                if state == "LISTEN":
                    oww_buf = np.append(oww_buf, audio)
                    while len(oww_buf) >= OWW_CHUNK:
                        if self._detected_wakeword(oww_buf[:OWW_CHUNK]):
                            print("[whisper] Wake word detected!", flush=True)
                            self._enter_recording_mode()
                            state = "RECORD"
                            recording = []
                            oww_buf = np.array([], dtype=np.float32)
                            break
                        oww_buf = oww_buf[OWW_CHUNK:]

                elif state == "RECORD":
                    recording.append(audio)
                    vad_buf = np.append(vad_buf, audio)
                    while len(vad_buf) >= VAD_CHUNK:
                        if not self._speech_ended(vad_buf[:VAD_CHUNK]):
                            vad_buf = vad_buf[VAD_CHUNK:]
                            continue

                        full_audio = np.concatenate(recording)
                        if len(full_audio) >= min_samples:
                            threading.Thread(
                                target=self._transcribe_and_publish,
                                args=(full_audio,),
                                daemon=True,
                            ).start()
                        else:
                            print("[whisper] Utterance too short, ignoring.", flush=True)

                        self._back_to_listen_mode(stream)
                        state = "LISTEN"
                        recording = []
                        oww_buf = np.array([], dtype=np.float32)
                        vad_buf = np.array([], dtype=np.float32)
                        break

    def _enter_recording_mode(self):
        self.vad.reset_states()

    def _back_to_listen_mode(self, stream):
        self._reset_wakeword()
        stream.stop()
        stream.start()


if __name__ == "__main__":
    WhisperNode().run()
