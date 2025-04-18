# server.py (audio detection integrated)
import asyncio
import json
import os
import subprocess
import tempfile
from functools import partial
import task
import websockets
import tempfile
from pathlib import Path
# Audio detection dependencies
import numpy as np
from pydub import AudioSegment
from ai_edge_litert.interpreter  import Interpreter          # pip install ai-edge-litert
from pydub import AudioSegment                  # pip install pydub

#AUDIO_MODEL = "whistle_classifier.tflite"       # adjust to your file
WHISTLE_CLASS_INDEX = 1                         # index of “whistle” in the model’s output
CONFIDENCE_THRESHOLD = 0.6
MIN_SEPARATION_SEC = 1.5                        # don’t fire twice inside this gap

VIDEO_DB = "videos.json"
AUDIO_MODEL = "soundclassifier_with_metadata.tflite"   # default model path, adjust as needed

def detect_whistles(video_path: str,
                    model_path: str | Path = AUDIO_MODEL) -> list[dict]:
    """
    Detect whistle events in an MP4 using a TFLite model via ai‑edge‑litert.

    Returns:
        [{"time": float, "label": str, "score": float}, …]
    """

    # ────────────────────────────────────────────────────────────────────────────
    # 1. Extract and convert audio → mono WAV (44.1 kHz, 16‑bit PCM)
    # ────────────────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        stereo_wav = os.path.join(tmpdir, "extracted.wav")
        mono_wav   = os.path.join(tmpdir, "mono.wav")

        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", stereo_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        AudioSegment.from_wav(stereo_wav).set_channels(1)\
                    .export(mono_wav, format="wav")

        # ────────────────────────────────────────────────────────────────────
        # 2. Load the model with ai‑edge‑litert
        # ────────────────────────────────────────────────────────────────────
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        in_details  = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        in_index    = in_details["index"]
        out_index   = out_details["index"]

        # Buffer size – assume waveform model with shape [1, N]
        # (If your model uses spectrograms, adjust here.)
        buffer_size = int(np.prod(in_details["shape"][1:]))

        # ────────────────────────────────────────────────────────────────────
        # 3. Prepare audio samples
        # ────────────────────────────────────────────────────────────────────
        raw_audio    = AudioSegment.from_wav(mono_wav)
        samples      = np.array(raw_audio.get_array_of_samples(), dtype=np.float32)
        samples      /= 32768.0                 # int16 → [-1, 1] float32
        sample_rate  = raw_audio.frame_rate

        step_size    = int(buffer_size * 0.5)   # 50 % overlap
        events: list[dict] = []
        last_time    = -1e9                     # far in the past

        # ────────────────────────────────────────────────────────────────────
        # 4. Sliding‑window inference
        # ────────────────────────────────────────────────────────────────────
        for start in range(0, len(samples) - buffer_size, step_size):
            chunk = samples[start : start + buffer_size]
            if chunk.shape[0] != buffer_size:
                break

            # Interpreter expects shape [1, N]
            interpreter.set_tensor(in_index, chunk.reshape(1, -1))
            interpreter.invoke()
            probs = interpreter.get_tensor(out_index)[0]

            score = float(probs[WHISTLE_CLASS_INDEX])
            if score >= CONFIDENCE_THRESHOLD:
                timestamp = start / sample_rate
                if timestamp - last_time >= MIN_SEPARATION_SEC:
                    events.append({
                        "time": round(timestamp, 2),
                        "label": "whistle",
                        "score": round(score, 2)
                    })
                    last_time = timestamp

        return events



class VideoStore:
  def __init__(self, db_path):
    self.db_path = db_path
    self.videos = self.load_videos()

  def load_videos(self):
    if os.path.exists(self.db_path):
      with open(self.db_path, "r", encoding="utf-8") as f:
        return json.load(f)
    return []

  def save_videos(self):
    with open(self.db_path, "w", encoding="utf-8") as f:
      json.dump(self.videos, f, indent=2)

  def get_all_videos(self):
    return self.videos

  def add_video(self, video_obj):
    self.videos.append(video_obj)
    self.save_videos()

  def remove_video(self, video_id):
    self.videos = [v for v in self.videos if v["id"] != video_id]
    self.save_videos()

  def toggle_heart(self, video_id):
    for v in self.videos:
      if v["id"] == video_id:
        v["liked"] = not v.get("liked", False)
        self.save_videos()
        break

  def edit_video(self, updated_video):
    for vid in self.videos:
      if vid["id"] == updated_video["id"]:
        vid["name"] = updated_video.get("name", vid["name"])
        break
    self.save_videos()

def extract_tags_from_video(video_filepath):
  print("[SERVER] extract_tags_from_video:", video_filepath)
  command = [
    "ffprobe", "-v", "quiet", "-print_format", "json",
    "-show_entries", "format_tags=comment",
    video_filepath
  ]
  process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if process.returncode != 0:
    raise Exception(f"FFprobe error: {process.stderr}")

  metadata = json.loads(process.stdout or "{}")
  tags_json = metadata.get("format", {}).get("tags", {}).get("comment", "")
  if not tags_json:
    return []
  try:
    tags = json.loads(tags_json)
    print("[SERVER] Found tags:", tags)
    return tags
  except json.JSONDecodeError:
    return []

def embed_tags_into_video(video_filepath, tags):
  print("[SERVER] embed_tags_into_video:", video_filepath, "with tags:", tags)
  tags_json = json.dumps(tags)
  temp_filepath = video_filepath + "_tmp.mp4"
  command = [
    "ffmpeg", "-y", "-i", video_filepath,
    "-map_metadata", "0",
    "-metadata", f"comment={tags_json}",
    "-c:v", "copy",
    "-c:a", "copy",
    "-movflags", "use_metadata_tags",
    temp_filepath
  ]
  process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if process.returncode != 0:
    print("[SERVER] FFmpeg stderr:", process.stderr)
    raise Exception(f"FFmpeg error: {process.stderr}")

  os.replace(temp_filepath, video_filepath)

  print("[SERVER] Embedded tags successfully")

async def handler(websocket):
  print("[SERVER] Client connected")
  async for msg in websocket:
    try:
      data = json.loads(msg)
      action = data.get("action")
      print(f"[SERVER] action={action}, data={data}")

      if action == "fetchVideos":
        await websocket.send(json.dumps({
          "type": "videoList",
          "videos": store.get_all_videos()
        }))

      elif action == "detectWhistles":
        video_path = data.get("videoPath")
        model_path = data.get("modelPath", AUDIO_MODEL)
        if not video_path or not os.path.exists(video_path):
          await websocket.send(json.dumps({"type": "error", "message": "Invalid videoPath"}))
        else:
          events = await asyncio.to_thread(detect_whistles, video_path, model_path)
          await websocket.send(json.dumps({"type": "whistleEvents", "events": events}))

      elif action == "addVideo":
        video_obj = data.get("video")
        store.add_video(video_obj)
        await websocket.send(json.dumps({
          "type": "videoList",
          "videos": store.get_all_videos()
        }))

      elif action == "removeVideo":
        vid_id = data.get("videoId")
        store.remove_video(vid_id)
        await websocket.send(json.dumps({
          "type": "videoList",
          "videos": store.get_all_videos()
        }))

      elif action == "toggleHeart":
        vid_id = data.get("videoId")
        store.toggle_heart(vid_id)
        await websocket.send(json.dumps({
          "type": "videoList",
          "videos": store.get_all_videos()
        }))

      elif action == "editVideo":
        updated_video = data.get("video", {})
        store.edit_video(updated_video)
        await websocket.send(json.dumps({
          "type": "videoList",
          "videos": store.get_all_videos()
        }))

      elif action == "fetchTags":
        path_ = data.get("videoPath")
        tags = extract_tags_from_video(path_)
        await websocket.send(json.dumps({"type": "tagList", "tags": tags}))

      elif action == "addTag":
        path_ = data.get("videoPath")
        new_tag = data.get("tag")
        tags = extract_tags_from_video(path_)
        tags.append(new_tag)
        embed_tags_into_video(path_, tags)
        await websocket.send(json.dumps({"type": "tagList", "tags": tags}))

      elif action == "editTag":
        path_ = data.get("videoPath")
        idx = data.get("index")
        updated_tag = data.get("tag")
        existing = extract_tags_from_video(path_)
        if 0 <= idx < len(existing):
          existing[idx] = updated_tag
          embed_tags_into_video(path_, existing)
          await websocket.send(json.dumps({"type": "tagList", "tags": existing}))
        else:
          await websocket.send(json.dumps({"type": "error", "message": "Invalid tag index"}))

      else:
        await websocket.send(json.dumps({"type": "error", "message": f"Unknown action: {action}"}))

    except Exception as e:
      print("[SERVER] Exception:", e)
      await websocket.send(json.dumps({"type": "error", "message": str(e)}))

async def main():
  global store
  store = VideoStore(VIDEO_DB)

  async with websockets.serve(handler, "localhost", 8765):
    print("[SERVER] WebSocket server on ws://localhost:8765")
    await asyncio.Future()

if __name__ == "__main__":
  asyncio.run(main())
