#!/usr/bin/env python3
import asyncio
import websockets
import json
import os
import sys
import cv2
import shutil
import subprocess
import numpy as np
import torch
import tempfile
import torch.nn.functional as F
import time
from PIL import Image
import base64
import io

from sam2.build_sam import build_sam2_video_predictor

# Import the audio event detection module
import audio_event_detector

###############################################################################
# 1) Utility
###############################################################################
def extract_frames_to_folder(video_path, frames_dir, timings):
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    start_t = time.perf_counter()
    cmd = [
        "ffmpeg","-y",
        "-i",video_path,
        "-q:v","2",
        "-start_number","0",
        os.path.join(frames_dir,"%05d.jpg")
    ]
    subprocess.run(cmd, check=True)
    end_t = time.perf_counter()
    timings["extract_time"] = (end_t - start_t)

def upsample_logits_to_fullres(mask_logits, orig_h, orig_w):
    if mask_logits.dim()==2:
        mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
    elif mask_logits.dim()==3:
        mask_logits = mask_logits.unsqueeze(0)
    upsampled = F.interpolate(mask_logits, size=(orig_h,orig_w),
                              mode='bilinear', align_corners=False)
    return (upsampled[0,0]>0).cpu().numpy()

def overlay_mask_on_frame(frame_bgr, mask, color=(0,255,0), alpha=0.4):
    overlay = frame_bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1-alpha, 0)

def write_segmented_video(frames_dir, video_segments, output_path, fps=20):
    frame_names = [p for p in os.listdir(frames_dir) if p.lower().endswith(".jpg")]
    frame_names.sort(key=lambda x:int(os.path.splitext(x)[0]))
    if not frame_names:
        print("[ERROR] No frames found.")
        return

    sample_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    if sample_frame is None:
        print("[ERROR] Cannot read sample frame.")
        return
    h,w = sample_frame.shape[:2]

    temp_avi = output_path+".tmp.avi"
    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(temp_avi, fourcc_avi, fps, (w,h))
    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter.")
        return

    for idx,fname in enumerate(frame_names):
        fpath = os.path.join(frames_dir,fname)
        frame_bgr = cv2.imread(fpath)
        if frame_bgr is None:
            continue
        mask = video_segments.get(idx,None)
        if mask is not None:
            frame_bgr = overlay_mask_on_frame(frame_bgr,mask,(0,255,0),0.4)
        writer.write(frame_bgr)

    writer.release()
    ffmpeg_cmd = [
        "ffmpeg","-y","-i",temp_avi,
        "-c:v","libx264","-crf","18","-preset","fast",output_path
    ]
    subprocess.run(ffmpeg_cmd,check=True)
    os.remove(temp_avi)
    print(f"[INFO] Wrote video => {output_path}")

def bgr_to_base64_png(frame_bgr, downscale_factor=0.5):
    h,w = frame_bgr.shape[:2]
    new_h = int(h*downscale_factor)
    new_w = int(w*downscale_factor)
    if new_h>0 and new_w>0:
        small = cv2.resize(frame_bgr,(new_w,new_h),cv2.INTER_AREA)
    else:
        small = frame_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf,format='PNG')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

###############################################################################
# 2) Global State
###############################################################################
STATE = {
    "video_path": None,
    "frames_dir": None,
    "first_frame_bgr": None,
    "H":None,"W":None,
    "box_prompt":None,
    "clicks":[],
    "predictor":None,
    "inference_state":None,
    "timings":{
        "extract_time":0.0,
        "propagate_frame_total":0.0,
        "propagate_frame_count":0
    },
    "video_segments":{},
    "device":None,
    "total_frames":0  # we store total frame count for partial progress
}

###############################################################################
# 3) WebSocket
###############################################################################
import websockets
connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    print("[DEBUG] Client connected.")
    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error":"Invalid JSON"}))
                continue
            action = data.get("action")
            payload= data.get("payload",{})
            if action=="upload_video":
                await handle_upload_video(websocket,payload)
            elif action=="add_click":
                await handle_add_click(websocket,payload)
            elif action=="set_box":
                await handle_set_box(websocket,payload)
            elif action=="undo":
                await handle_undo(websocket,payload)
            elif action=="propagate":
                await handle_propagate(websocket,payload)
            # New action: detect audio events using audio_event_detector.py
            elif action=="detect_audio_events":
                await handle_detect_audio_events(websocket, payload)
            else:
                await websocket.send(json.dumps({"error":f"Unknown action: {action}"}))
    except websockets.ConnectionClosed as e:
        print(f"[DEBUG] Client disconnected => reason={e.reason}")
    finally:
        connected_clients.remove(websocket)

###############################################################################
# 4) Helper for partial mask
###############################################################################
def get_current_mask_logits():
    s = STATE
    if not s["clicks"] and not s["box_prompt"]:
        return None
    predictor = s["predictor"]
    inference_state = s["inference_state"]
    points_np, labels_np, box_np = None, None, None
    if s["clicks"]:
        points_np = np.array([(c[0], c[1]) for c in s["clicks"]],dtype=np.float32)
        labels_np = np.array([c[2] for c in s["clicks"]],dtype=np.int32)
    if s["box_prompt"]:
        box_np = np.array(s["box_prompt"],dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points_np,
        labels=labels_np,
        box=box_np,
    )
    return out_mask_logits[0]

###############################################################################
# 5) Action handlers
###############################################################################
async def handle_upload_video(ws,payload):
    s = STATE
    video_path = payload.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        await ws.send(json.dumps({"error":f"Invalid video_path: {video_path}"}))
        return

    # reset
    s["video_path"] = video_path
    s["box_prompt"]=None
    s["clicks"]=[]
    s["video_segments"]={}
    for k in ["extract_time","propagate_frame_total","propagate_frame_count"]:
        s["timings"][k]=0.0

    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    s["frames_dir"]=frames_dir

    # 1) Extract frames (no partial progress, just once done => 100%)
    extract_frames_to_folder(video_path, frames_dir, s["timings"])

    # load frames
    frame_names = sorted([fn for fn in os.listdir(frames_dir) if fn.endswith(".jpg")],
                         key=lambda x:int(os.path.splitext(x)[0]))
    if not frame_names:
        await ws.send(json.dumps({"error":"No frames extracted"}))
        return
    s["total_frames"] = len(frame_names)  # store total for partial progress in propagate

    first_frame_path = os.path.join(frames_dir, frame_names[0])
    first_frame_bgr = cv2.imread(first_frame_path)
    if first_frame_bgr is None:
        await ws.send(json.dumps({"error":f"Failed to load first frame at {first_frame_path}"}))
        return

    s["first_frame_bgr"] = first_frame_bgr
    s["H"], s["W"] = first_frame_bgr.shape[:2]

    # pick device
    if torch.cuda.is_available():
        s["device"] = torch.device("cuda")
    elif torch.backends.mps.is_available():
        s["device"] = torch.device("mps")
    else:
        s["device"] = torch.device("cpu")

    # build predictor
    checkpoint = payload.get("checkpoint","/home/dc/PycharmProjects/SegmentationRaw/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    config = payload.get("config","configs/sam2.1/sam2.1_hiera_t.yaml")
    predictor = build_sam2_video_predictor(config, checkpoint, device=s["device"])
    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state)

    s["predictor"]=predictor
    s["inference_state"]=inference_state

    # send partial overlay
    base64_img = bgr_to_base64_png(first_frame_bgr,0.5)
    resp = {
        "action":"video_ready",
        "msg":"Frames extracted & SAM2 initialized",
        "first_frame":base64_img
    }
    await ws.send(json.dumps(resp))

async def handle_add_click(ws,payload):
    s = STATE
    x = payload.get("x")
    y = payload.get("y")
    label = payload.get("label",1)
    s["clicks"].append((x,y,label))

    s["predictor"].reset_state(s["inference_state"])
    mask_logits = get_current_mask_logits()
    if mask_logits is not None:
        fullres_mask = upsample_logits_to_fullres(mask_logits, s["H"], s["W"])
        overlay_bgr = overlay_mask_on_frame(s["first_frame_bgr"], fullres_mask,(0,255,0),0.4)
    else:
        overlay_bgr = s["first_frame_bgr"]

    overlay_b64 = bgr_to_base64_png(overlay_bgr,0.5)
    resp = {"action":"mask_update","overlay":overlay_b64}
    await ws.send(json.dumps(resp))

async def handle_set_box(ws,payload):
    s = STATE
    x1 = payload.get("x1")
    y1 = payload.get("y1")
    x2 = payload.get("x2")
    y2 = payload.get("y2")
    s["box_prompt"] = [x1,y1,x2,y2]

    s["predictor"].reset_state(s["inference_state"])
    mask_logits = get_current_mask_logits()
    if mask_logits is not None:
        fullres_mask = upsample_logits_to_fullres(mask_logits, s["H"], s["W"])
        overlay_bgr = overlay_mask_on_frame(s["first_frame_bgr"], fullres_mask,(0,255,0),0.4)
    else:
        overlay_bgr = s["first_frame_bgr"]

    overlay_b64 = bgr_to_base64_png(overlay_bgr,0.5)
    resp = {"action":"mask_update","overlay":overlay_b64}
    await ws.send(json.dumps(resp))

async def handle_undo(ws,payload):
    s = STATE
    mode = payload.get("mode","click")
    if mode=="click":
        if s["clicks"]:
            s["clicks"].pop()
    else:
        s["box_prompt"]=None

    s["predictor"].reset_state(s["inference_state"])
    mask_logits = get_current_mask_logits()
    if mask_logits is not None:
        fullres_mask = upsample_logits_to_fullres(mask_logits, s["H"], s["W"])
        overlay_bgr = overlay_mask_on_frame(s["first_frame_bgr"], fullres_mask,(0,255,0),0.4)
    else:
        overlay_bgr = s["first_frame_bgr"]

    overlay_b64 = bgr_to_base64_png(overlay_bgr,0.5)
    resp = {"action":"mask_update","overlay":overlay_b64}
    await ws.send(json.dumps(resp))

async def handle_propagate(ws,payload):
    s = STATE
    s["predictor"].reset_state(s["inference_state"])
    get_current_mask_logits() # re-add final prompts on frame0

    total_frames = s["total_frames"]  # number of frames
    video_segments = {}
    count=0
    start_t = time.perf_counter()

    for out_frame_idx, out_obj_ids, out_mask_logits in s["predictor"].propagate_in_video(s["inference_state"]):
        iteration_start = time.perf_counter()
        logits = out_mask_logits[0]
        mask_fullres = upsample_logits_to_fullres(logits, s["H"], s["W"])
        video_segments[out_frame_idx] = mask_fullres
        iteration_end = time.perf_counter()
        s["timings"]["propagate_frame_total"] += (iteration_end - iteration_start)
        s["timings"]["propagate_frame_count"] += 1
        count+=1

        # partial progress
        percent = int((count/total_frames)*100)
        await ws.send(json.dumps({
            "action":"progress",
            "percent":percent,
            "msg":f"Propagated frame {count}/{total_frames}"
        }))

    end_t = time.perf_counter()
    print(f"[INFO] Propagation => {count} frames in {(end_t - start_t):.2f}s")

    s["video_segments"] = video_segments

    output_path = payload.get("output_path","output_segmented.mp4")
    fps = payload.get("fps",30)
    write_segmented_video(s["frames_dir"],video_segments,output_path,fps=fps)

    resp = {
        "action":"propagation_done",
        "msg":f"Segmentation done. Wrote video: {output_path}",
        "output_path":output_path
    }
    await ws.send(json.dumps(resp))

# New handler for audio event detection
async def handle_detect_audio_events(ws, payload):
    video_path = payload.get("video_path")
    model_path = payload.get("model_path")
    if not video_path or not os.path.isfile(video_path):
        await ws.send(json.dumps({"error":f"Invalid video_path: {video_path}"}))
        return
    if not model_path or not os.path.isfile(model_path):
        await ws.send(json.dumps({"error":f"Invalid model_path: {model_path}"}))
        return

    # Optional parameters
    event_index = payload.get("event_index", 1)
    confidence_threshold = payload.get("confidence_threshold", 0.5)
    min_separation = payload.get("min_separation", 1.5)

    loop = asyncio.get_running_loop()
    try:
        # Run the synchronous detect_events in a separate thread to avoid blocking the event loop
        events = await loop.run_in_executor(
            None,
            audio_event_detector.detect_events,
            video_path,
            model_path,
            event_index,
            confidence_threshold,
            min_separation
        )
        response = {
            "action": "audio_event_detection_done",
            "events": events,
            "msg": "Audio event detection completed successfully."
        }
        await ws.send(json.dumps(response))
    except Exception as e:
        await ws.send(json.dumps({"error": f"Audio event detection failed: {str(e)}"}))

###############################################################################
# 6) Main
###############################################################################
async def main():
    print("[DEBUG] Starting server on ws://localhost:8765 with max_size=None")
    async with websockets.serve(handler, "localhost", 8765, max_size=None):
        print("[DEBUG] WebSocket server started => ws://localhost:8765")
        await asyncio.Future()

if __name__=="__main__":
    asyncio.run(main())
