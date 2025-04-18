/********************************************
 * VIDEO EDITOR FUNCTIONALITY
 ********************************************/
let currentVideoPath = null;
let timelineTagsData = [];
let videoDuration = 0;
let draggingHandle = null;
let selectionStart = 0;
let selectionEnd = 0;

// Drawing state arrays with timestamp tracking
let drawnElements = [];         // General drawings
let homographyElements = [];      // Halo overlays from homography tool

// Drawing state variables for general drawing
let currentTool = 'pointer'; // pointer, pen, square, polygon, text, circle, homography
let isDrawing = false;
let lastX = 0, lastY = 0;
let startX, startY;
let currentTextElement = null;
let currentPolygonPoints = [];

// Variables for homography tool
let currentHomographyBox = null;
let homographyDrawing = false;
let draggingHomographyIndex = null;
let homographyDragOffset = { x: 0, y: 0 };
let editingHomographyIndex = null;
let defaultHomographyRingColor = "#ffffff";
let defaultHomographySpotlightColor = "#ffffff";

let zoomLevel = 1;
let zoomOffsetX = 0;
let zoomOffsetY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;

const videoPlayer = document.getElementById("videoPlayer");
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const videoContainer = document.querySelector('.video-container');

// For touch pinch zoom
let pinchStartDistance = null;
let pinchStartZoom = null;

// WebSocket (adjust URL/port as needed)
let socketOpened = false;
const socket = new WebSocket("ws://localhost:8765");
const pendingMessages = [];
socket.addEventListener("open", () => {
  console.log("[CLIENT] WebSocket connected");
  pendingMessages.forEach(msg => socket.send(msg));
  pendingMessages.length = 0;
  sendSocketMessage("fetchVideos", { videoDBPath: "videos.json" });
});

socket.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  console.log("[CLIENT] Received:", msg);
  if (msg.type === "videoList") {
    updateVideoList(msg.videos);
  } else if (msg.type === "tagList") {
    updateTagList(msg.tags);
  } else if (msg.type === "whistleEvents") {
    // Convert every detected event to a one‑second tag
    const autoTags = msg.events.map(ev => ({
      name: ev.label || "Whistle",
      startSec: ev.time,
      endSec: ev.time + 1,
      color: getRandomMutedColor(),
      timestamp: Date.now()
    }));
    // Merge with any existing timeline tags and redraw
    timelineTagsData = [...timelineTagsData, ...autoTags];
    renderSidebarTags();
    renderTimelineTags();
    showToast(`Added ${autoTags.length} whistle tag(s)`);
  }
  else if (msg.type === "error") {
    showToast("Error: " + msg.message);
  } else if (msg.type === "toast") {
    showToast(msg.message);
  }
});
socket.addEventListener("close", () => {
  showToast(socket.readyState === WebSocket.OPEN
    ? "WebSocket disconnected"
    : "WebSocket not connected");
});
socket.addEventListener("error", () => {
  showToast("WebSocket error");
});


/********************************************
 * UTILITY FUNCTIONS (Video Editor)
 ********************************************/
function showToast(msg) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast show bg-success text-white p-2 mb-2";
  toast.style.position = "relative";
  toast.innerHTML = `<div class="toast-body">${msg}</div>`;
  container.appendChild(toast);
  setTimeout(() => { toast.remove(); }, 3000);
}
function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) return "0:00";
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(hrs).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
}
function getRandomMutedColor() {
  const hue = Math.floor(Math.random() * 360);
  return `hsl(${hue}, 50%, 50%)`;
}
function toFileURL(rawPath) {
  if (!rawPath) return "";
  let path = rawPath.replace(/\\/g, "/");
  if (!path.startsWith("/")) { path = "/" + path; }
  return "file://" + encodeURI(path);
}
function sendSocketMessage(action, payload = {}) {
  if (!action) return;
  const json = JSON.stringify({ action, ...payload });
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(json);
  } else {
    console.log(`[CLIENT] WS not open, queueing ${action}`);
    pendingMessages.push(json);
  }
}

/********************************************
 * SIDEBAR & LAYOUT (Video Editor)
 ********************************************/
function toggleSidebar(id) {
  document.getElementById(id).classList.toggle("collapsed");
  updateMainContentSize();
}
function updateMainContentSize() {
  const leftPanel = document.getElementById("videoPanel");
  const rightPanel = document.getElementById("tagPanel");
  const mainContent = document.getElementById("mainContent");
  const leftWidth = leftPanel.classList.contains("collapsed") ? 0 : 250;
  const rightWidth = rightPanel.classList.contains("collapsed") ? 0 : 250;
  const availableWidth = window.innerWidth - leftWidth - rightWidth;
  mainContent.style.width = availableWidth + "px";
  mainContent.style.marginLeft = leftWidth + "px";
  mainContent.style.marginRight = rightWidth + "px";
}
window.addEventListener("resize", updateMainContentSize);

/********************************************
 * VIDEO LIST & ADDING VIDEOS (Video Editor)
 ********************************************/
function toggleVideoForm() {
  const form = document.getElementById("addVideoForm");
  const btn = document.getElementById("toggleVideoFormBtn");
  if (form.style.display === "none") {
    form.style.display = "block";
    btn.innerHTML = '<i class="fa-solid fa-minus"></i>';
  } else {
    form.style.display = "none";
    btn.innerHTML = '<i class="fa-solid fa-plus"></i>';
  }
}
async function addVideo() {
  const fileInput = document.getElementById("videoInput");
  const videoNameInput = document.getElementById("videoName");
  if (!fileInput.files.length || !videoNameInput.value.trim()) {
    Swal.fire({
      background: '#2f2f2f',
      color: '#ffffff',
      icon: 'error',
      title: 'Missing Input',
      text: 'Please select a video and enter a name!'
    });
    return;
  }
  const videoFile = fileInput.files[0];
  let absolutePath = videoFile.path;
  if (!absolutePath || absolutePath === videoFile.name) {
    showToast("Unable to retrieve absolute path in browser environment");
  }
  const videoObj = {
    id: Date.now(),
    name: videoNameInput.value.trim(),
    filePath: absolutePath,
    liked: false,
    timestamp: Date.now()
  };
  sendSocketMessage("addVideo", { video: videoObj });
  // When switching videos, clear both drawing arrays
  drawnElements = [];
  homographyElements = [];
  redrawCanvas();
  fileInput.value = "";
  videoNameInput.value = "";
  toggleVideoForm();
}
function updateVideoList(videos) {
  const videoListElem = document.getElementById("videoList");
  videoListElem.innerHTML = "";
  videos.forEach((video) => {
    const videoItem = document.createElement("div");
    videoItem.className = "video-item";
    videoItem.innerHTML = `
      <span class="video-name" title="${video.name}">${video.name}</span>
      <div class="video-buttons">
        <button class="favorite">
          <i class="fa-solid fa-heart${video.liked ? " active" : ""}"></i>
        </button>
        <button class="edit">
          <i class="fa-solid fa-cog"></i>
        </button>
        <button class="delete">
          <i class="fa-solid fa-trash"></i>
        </button>
      </div>
    `;
    videoItem.addEventListener("click", () => {
      currentVideoPath = video.filePath;
      videoPlayer.src = toFileURL(video.filePath);
      document.getElementById("videoTitle").textContent = video.name;
      // Clear drawings when switching videos
      drawnElements = [];
      homographyElements = [];
      redrawCanvas();
      sendSocketMessage("fetchTags", { videoPath: currentVideoPath });
    });
    videoItem.querySelector(".favorite").addEventListener("click", (e) => {
      e.stopPropagation();
      sendSocketMessage("toggleHeart", { videoId: video.id });
    });
    videoItem.querySelector(".delete").addEventListener("click", (e) => {
      e.stopPropagation();
      Swal.fire({
        background: '#2f2f2f',
        color: '#ffffff',
        title: 'Are you sure?',
        text: "Are you sure you want to delete this video?",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Yes, delete it!',
        cancelButtonText: 'Cancel'
      }).then((result) => {
        if (result.isConfirmed) {
          sendSocketMessage("removeVideo", { videoId: video.id });
        }
      });
    });
    videoItem.querySelector(".edit").addEventListener("click", (e) => {
      e.stopPropagation();
      Swal.fire({
        background: '#2f2f2f',
        color: '#ffffff',
        title: 'Edit Video Name',
        input: 'text',
        inputValue: video.name,
        showCancelButton: true,
        confirmButtonText: 'Save',
        cancelButtonText: 'Cancel'
      }).then((result) => {
        if (result.isConfirmed && result.value && result.value.trim() !== "") {
          video.name = result.value.trim();
          sendSocketMessage("editVideo", { video });
        }
      });
    });
    videoListElem.appendChild(videoItem);
  });
}

/********************************************
 * TAGGING SYSTEM (Video Editor)
 ********************************************/
function updateTagList(tags) {
  timelineTagsData = tags.map((tag) => {
    if (!tag.color) tag.color = getRandomMutedColor();
    return { ...tag, isSelected: false, rangeEditing: false, domElement: null };
  });
  renderSidebarTags();
  renderTimelineTags();
}
function addTag() {
  const tagInput = document.getElementById("tagInput");
  const presetSelect = document.getElementById("presetTags");
  let tagValue = tagInput.value.trim() || presetSelect.value.trim();
  if (!tagValue) return;
  if (selectionStart === selectionEnd) {
    Swal.fire({
      background: '#2f2f2f',
      color: '#ffffff',
      icon: 'error',
      title: 'Invalid Range',
      text: 'Please select a time range on the timeline first!'
    });
    return;
  }
  const newTag = {
    name: tagValue,
    startSec: selectionStart,
    endSec: selectionEnd,
    timestamp: Date.now()
  };
  sendSocketMessage("addTag", { videoPath: currentVideoPath, tag: newTag });
  tagInput.value = "";
  presetSelect.value = "";
}
function renderSidebarTags() {
  const tagListElem = document.getElementById("tagList");
  tagListElem.innerHTML = "";
  timelineTagsData.forEach((tag, index) => {
    const displayText = `${tag.name} (${formatTime(tag.startSec)} - ${formatTime(tag.endSec)})`;
    const tagItem = document.createElement("div");
    tagItem.className = "tag-item";
    tagItem.setAttribute("data-tag-index", index);
    tagItem.style.borderLeft = `8px solid ${tag.color}`;
    tagItem.innerHTML = `
      <input type="checkbox" class="tag-select">
      <span class="tag-display" title="${displayText}">${displayText}</span>
      <div class="tag-buttons">
        <button class="editNameBtn"><i class="fa-solid fa-pen"></i></button>
        <button class="editRangeBtn"><i class="fa-solid fa-arrows-left-right-to-line"></i></button>
        <button class="delete"><i class="fa-solid fa-trash"></i></button>
      </div>
    `;
    tagItem.querySelector(".editNameBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      Swal.fire({
        background: '#2f2f2f',
        color: '#ffffff',
        title: 'Edit Tag Name',
        input: 'text',
        inputValue: tag.name,
        showCancelButton: true,
        confirmButtonText: 'Save',
        cancelButtonText: 'Cancel'
      }).then((result) => {
        if(result.isConfirmed && result.value && result.value.trim()) {
          const updatedTag = { ...tag, name: result.value.trim() };
          sendSocketMessage("editTag", {
            videoPath: currentVideoPath,
            tagIndex: index,
            tag: updatedTag
          });
        }
      });
    });
    tagItem.querySelector(".editRangeBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      if (!tag.rangeEditing) {
        timelineTagsData.forEach((t) => (t.rangeEditing = false));
        tag.rangeEditing = true;
        selectionStart = tag.startSec;
        selectionEnd = tag.endSec;
        updateTimelineHandles();
        showToast(`Editing range for "${tag.name}". Drag handles or scrub the video.`);
      } else {
        const updatedTag = { ...tag, startSec: selectionStart, endSec: selectionEnd, rangeEditing: false };
        sendSocketMessage("editTag", {
          videoPath: currentVideoPath,
          tagIndex: index,
          tag: updatedTag
        });
      }
    });
    tagItem.querySelector(".delete").addEventListener("click", (e) => {
      e.stopPropagation();
      Swal.fire({
        background: '#2f2f2f',
        color: '#ffffff',
        title: 'Are you sure?',
        text: "Are you sure you want to delete this tag?",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Yes, delete it!',
        cancelButtonText: 'Cancel'
      }).then((result) => {
        if(result.isConfirmed) {
          sendSocketMessage("removeTag", {
            videoPath: currentVideoPath,
            tagIndex: index
          });
        }
      });
    });
    tagItem.addEventListener("click", (e) => {
      e.stopPropagation();
      timelineTagsData.forEach((t) => (t.isSelected = false));
      tag.isSelected = true;
      renderTimelineTags();
      videoPlayer.currentTime = tag.startSec;
    });
    tagListElem.appendChild(tagItem);
  });
}
function renderTimelineTags() {
  const timelineTagsEl = document.getElementById("timelineTags");
  timelineTagsEl.innerHTML = "";
  const width = timelineContainer.offsetWidth;
  timelineTagsData.forEach((tag) => {
    const startPx = (tag.startSec / videoDuration) * width;
    const endPx = (tag.endSec / videoDuration) * width;
    const left = Math.min(startPx, endPx);
    const tagW = Math.abs(endPx - startPx);
    const tagDiv = document.createElement("div");
    tagDiv.className = "timeline-tag";
    if (tag.isSelected) tagDiv.classList.add("selected");
    tagDiv.style.left = left + "px";
    tagDiv.style.width = Math.max(2, tagW) + "px";
    tagDiv.style.backgroundColor = tag.color;
    tagDiv.addEventListener("click", (evt) => {
      evt.stopPropagation();
      timelineTagsData.forEach((t) => (t.isSelected = false));
      tag.isSelected = true;
      renderTimelineTags();
      videoPlayer.currentTime = tag.startSec || 0;
    });
    timelineTagsEl.appendChild(tagDiv);
    tag.domElement = tagDiv;
  });
}

/********************************************
 * TIMELINE & PLAYER BAR (Video Editor)
 ********************************************/
const timelineContainer = document.getElementById("timelineContainer");
const selectionEl = document.getElementById("selection");
const handleStartEl = document.getElementById("handleStart");
const handleEndEl = document.getElementById("handleEnd");
const playerPlayPause = document.getElementById("playerPlayPause");
const playerBarTime = document.getElementById("playerBarTime");
const playerBarProgressContainer = document.getElementById("playerBarProgressContainer");
const playerBarProgressFill = document.getElementById("playerBarProgressFill");
const playerBarProgressDot = document.getElementById("playerBarProgressDot");

videoPlayer.addEventListener("loadedmetadata", () => {
  videoDuration = videoPlayer.duration || 0;
  selectionStart = 0;
  selectionEnd = 0;
  updateTimelineHandles();
  resizeCanvas();
});
videoPlayer.addEventListener("timeupdate", () => {
  const cTime = videoPlayer.currentTime;
  const dur = videoPlayer.duration || 0;
  playerBarTime.textContent = formatTime(cTime);
  const percent = dur ? (cTime / dur) * 100 : 0;
  playerBarProgressFill.style.width = percent + "%";
  const editingTag = timelineTagsData.find((t) => t.rangeEditing);
  if (editingTag && !draggingHandle) {
    if (cTime < editingTag.startSec) {
      selectionStart = cTime;
      selectionEnd = editingTag.endSec;
    } else if (cTime > editingTag.endSec) {
      selectionEnd = cTime;
      selectionStart = editingTag.startSec;
    } else {
      const distToStart = Math.abs(cTime - editingTag.startSec);
      const distToEnd = Math.abs(cTime - editingTag.endSec);
      if (distToStart < distToEnd) {
        selectionStart = cTime;
        selectionEnd = editingTag.endSec;
      } else {
        selectionEnd = cTime;
        selectionStart = editingTag.startSec;
      }
    }
    updateTimelineHandles();
  }
});
function togglePlayPause() {
  if (videoPlayer.paused) {
    videoPlayer.play();
    playerPlayPause.classList.remove("fa-play");
    playerPlayPause.classList.add("fa-pause");
  } else {
    videoPlayer.pause();
    playerPlayPause.classList.remove("fa-pause");
    playerPlayPause.classList.add("fa-play");
  }
}
playerPlayPause.addEventListener("click", togglePlayPause);
document.getElementById("playerStepBack").addEventListener("click", () => {
  videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
});
document.getElementById("playerStepForward").addEventListener("click", () => {
  videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 5);
});
document.getElementById("playerBackward").addEventListener("click", () => {
  videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 10);
});
document.getElementById("playerMute").addEventListener("click", (e) => {
  videoPlayer.muted = !videoPlayer.muted;
  if (videoPlayer.muted) {
    e.target.classList.remove("fa-volume-mute");
    e.target.classList.add("fa-volume-up");
  } else {
    e.target.classList.add("fa-volume-mute");
    e.target.classList.remove("fa-volume-up");
  }
});
const playerSpeed = document.getElementById("playerSpeed");
playerSpeed.addEventListener("click", () => {
  let newRate = 1;
  switch (videoPlayer.playbackRate) {
    case 1: newRate = 1.5; break;
    case 1.5: newRate = 2; break;
    case 2: newRate = 0.5; break;
    default: newRate = 1; break;
  }
  videoPlayer.playbackRate = newRate;
  playerSpeed.textContent = "x" + newRate;
});
videoContainer.addEventListener("click", (evt) => {
  if (currentTool === 'pointer' && zoomLevel === 1 && evt.target === videoPlayer) {
    togglePlayPause();
  }
});
document.addEventListener("keydown", (e) => {
  // Check if the active element is contentEditable.
  if (document.activeElement && document.activeElement.isContentEditable) {
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    togglePlayPause();
  } else if (e.code === "ArrowLeft") {
    e.preventDefault();
    videoPlayer.currentTime = Math.max(videoPlayer.currentTime - 5, 0);
  } else if (e.code === "ArrowRight") {
    e.preventDefault();
    videoPlayer.currentTime = Math.min(videoPlayer.currentTime + 5, videoPlayer.duration);
  } else if (e.code === "KeyM") {
    videoPlayer.muted = !videoPlayer.muted;
  }
});
playerBarProgressContainer.addEventListener("click", (e) => {
  const rect = playerBarProgressContainer.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const width = rect.width;
  const dur = videoPlayer.duration || 0;
  videoPlayer.currentTime = (clickX / width) * dur;
});

/********************************************
 * TIMELINE HANDLES (Video Editor)
 ********************************************/
function updateTimelineHandles() {
  const width = timelineContainer.offsetWidth;
  const startX = (selectionStart / videoDuration) * width;
  const endX = (selectionEnd / videoDuration) * width;
  handleStartEl.style.left = startX + "px";
  handleEndEl.style.left = endX + "px";
  selectionEl.style.left = Math.min(startX, endX) + "px";
  selectionEl.style.width = Math.abs(endX - startX) + "px";
  renderTimelineTags();
}
function onHandleMouseMove(e) {
  if (!draggingHandle) return;
  const rect = timelineContainer.getBoundingClientRect();
  let xPos = e.clientX - rect.left;
  xPos = Math.max(0, Math.min(timelineContainer.offsetWidth, xPos));
  const newSec = (xPos / timelineContainer.offsetWidth) * videoDuration;
  if (draggingHandle === "start") {
    selectionStart = Math.min(newSec, selectionEnd);
    videoPlayer.currentTime = selectionStart;
  } else {
    selectionEnd = Math.max(newSec, selectionStart);
    videoPlayer.currentTime = selectionEnd;
  }
  updateTimelineHandles();
}
function onHandleMouseUp() {
  draggingHandle = null;
  document.removeEventListener("mousemove", onHandleMouseMove);
  document.removeEventListener("mouseup", onHandleMouseUp);
}
handleStartEl.addEventListener("mousedown", () => {
  draggingHandle = "start";
  document.addEventListener("mousemove", onHandleMouseMove);
  document.addEventListener("mouseup", onHandleMouseUp);
});
handleEndEl.addEventListener("mousedown", () => {
  draggingHandle = "end";
  document.addEventListener("mousemove", onHandleMouseMove);
  document.addEventListener("mouseup", onHandleMouseUp);
});

/********************************************
 * DRAWING CANVAS RESIZE & COORDINATE MAPPING (Video Editor)
 ********************************************/
function resizeCanvas() {
  canvas.width = videoContainer.clientWidth;
  canvas.height = videoContainer.clientHeight;
  redrawCanvas();
}
window.addEventListener("resize", resizeCanvas);
videoPlayer.addEventListener("play", resizeCanvas);
videoPlayer.addEventListener("loadeddata", resizeCanvas);

/********************************************
 * ZOOM & PAN (Video Editor)
 ********************************************/
function updateZoomTransform() {
  videoPlayer.style.transform = `scale(${zoomLevel}) translate(${zoomOffsetX}px, ${zoomOffsetY}px)`;
  canvas.style.transform = `scale(${zoomLevel}) translate(${zoomOffsetX}px, ${zoomOffsetY}px)`;
}
function applyZoom() {
  updateZoomTransform();
}
videoContainer.addEventListener('mousedown', (e) => {
  if (currentTool === 'pointer' && zoomLevel > 1) {
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    e.preventDefault();
  }
});
videoContainer.addEventListener('mousemove', (e) => {
  if (isPanning) {
    const dx = e.clientX - panStartX;
    const dy = e.clientY - panStartY;
    zoomOffsetX += dx / zoomLevel;
    zoomOffsetY += dy / zoomLevel;
    panStartX = e.clientX;
    panStartY = e.clientY;
    updateZoomTransform();
  }
});
videoContainer.addEventListener('mouseup', () => { isPanning = false; });
videoContainer.addEventListener('mouseleave', () => { isPanning = false; });
videoContainer.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.ctrlKey) {
    zoomLevel -= e.deltaY * 0.001;
    zoomLevel = Math.max(0.5, Math.min(2, zoomLevel));
    updateZoomTransform();
  } else {
    zoomOffsetX -= e.deltaX / zoomLevel;
    zoomOffsetY -= e.deltaY / zoomLevel;
    updateZoomTransform();
  }
});
videoContainer.addEventListener('touchstart', (e) => {
  if (e.touches.length === 2) {
    e.preventDefault();
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    pinchStartDistance = Math.sqrt(dx * dx + dy * dy);
    pinchStartZoom = zoomLevel;
  }
});
videoContainer.addEventListener('touchmove', (e) => {
  if (e.touches.length === 2 && pinchStartDistance) {
    e.preventDefault();
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const newDistance = Math.sqrt(dx * dx + dy * dy);
    const scale = newDistance / pinchStartDistance;
    zoomLevel = pinchStartZoom * scale;
    updateZoomTransform();
  }
});
videoContainer.addEventListener('touchend', (e) => {
  if (e.touches.length < 2) {
    pinchStartDistance = null;
    pinchStartZoom = null;
  }
});
function resetView() {
  zoomLevel = 1;
  zoomOffsetX = 0;
  zoomOffsetY = 0;
  updateZoomTransform();
  showToast("View reset to normal");
}

/********************************************
 * TOOL BUTTONS & DRAWING (Video Editor)
 ********************************************/
function highlightActiveTool(toolId) {
  document.querySelectorAll('.nav-link').forEach(btn => { btn.classList.remove('active'); });
  const activeBtn = document.querySelector(`[data-btn-id="${toolId}"]`);
  if (activeBtn) { activeBtn.classList.add('active'); }
}
function setupToolButtons() {
  document.querySelector('[data-btn-id="mag-minus"]').addEventListener('click', () => {
    if (zoomLevel > 0.5) {
      zoomLevel -= 0.1;
      applyZoom();
    }
  });
  document.querySelector('[data-btn-id="mag-plus"]').addEventListener('click', () => {
    if (zoomLevel < 2) {
      zoomLevel += 0.1;
      applyZoom();
    }
  });
  document.querySelector('[data-btn-id="reset-view"]').addEventListener('click', resetView);
  document.querySelector('[data-btn-id="mouse-pointer"]').addEventListener('click', () => {
    currentTool = 'pointer';
    canvas.style.cursor = 'default';
    highlightActiveTool('mouse-pointer');
  });
  document.querySelector('[data-btn-id="pen"]').addEventListener('click', () => {
    currentTool = 'pen';
    canvas.style.cursor = 'crosshair';
    highlightActiveTool('pen');
  });
document.querySelector('[data-btn-id="vector-square"]').addEventListener('click', () => {
  currentTool = 'vectorSquare';
  canvas.style.cursor = 'crosshair';
  highlightActiveTool('vector-square');
  currentPolygonPoints = []; // clear previous points
});
document.querySelector('[data-btn-id="draw-polygon"]').addEventListener('click', () => {
  currentTool = 'polygon';
  canvas.style.cursor = 'crosshair';
  highlightActiveTool('draw-polygon');
  currentPolygonPoints = []; // clear previous points
});

  document.querySelector('[data-btn-id="comment"]').addEventListener('click', () => {
    currentTool = 'text';
    canvas.style.cursor = 'text';
    highlightActiveTool('comment');
  });
  document.querySelector('[data-btn-id="circle-notch"]').addEventListener('click', () => {
    currentTool = 'circle';
    canvas.style.cursor = 'crosshair';
    highlightActiveTool('circle-notch');
  });
  // Eraser button now clears all drawings
  document.querySelector('[data-btn-id="eraser"]').addEventListener('click', () => {
    drawnElements = [];
    homographyElements = [];
      document.querySelectorAll('.draggable-text').forEach(textElem => {
    textElem.remove();
  });
    redrawCanvas();
  });
  // Undo button now calls our combined undo function
  document.querySelector('[data-btn-id="undo"]').addEventListener('click', undoLast);
  document.querySelector('[data-btn-id="camera"]').addEventListener('click', takeScreenshot);
  // Set homography tool mode
  document.querySelector('[data-btn-id="set-homography"]').addEventListener('click', () => {
    currentTool = 'homography';
    canvas.style.cursor = 'crosshair';
    highlightActiveTool('set-homography');
  });
 //document.querySelector('[data-btn-id="floor-ring"]').addEventListener('click', () => {
 //   showToast("Floor Ring clicked (not fully implemented).");
 // });
}

/********************************************
 * CANVAS DRAWING HANDLERS (Integrated)
 ********************************************/
canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (currentTool === 'homography') {
    if(e.button === 2) return;
    // Check if clicking near an existing homography overlay’s ring to drag it
    for (let i = 0; i < homographyElements.length; i++) {
      const elem = homographyElements[i];
      const ring = elem.ringCenter;
      const dx = x - ring.x, dy = y - ring.y;
      if (Math.sqrt(dx*dx + dy*dy) < 20) {
        draggingHomographyIndex = i;
        homographyDragOffset = { x: dx, y: dy };
        return;
      }
    }
    homographyDrawing = true;
    currentHomographyBox = { startX: x, startY: y, endX: x, endY: y, timestamp: Date.now() };
    return;
  }
  if (currentTool === 'polygon' || currentTool === 'vectorSquare') {
    currentPolygonPoints.push({ x, y });
    redrawCanvas(); // Update the preview with the new point
    return;
  }
  // Other tools
  if (currentTool === 'pen') {
    isDrawing = true;
    lastX = x;
    lastY = y;
    drawnElements.push({
      type: 'pen',
      points: [{ x, y }],
      color: '#00ff9f',
      width: 3,
      timestamp: Date.now()
    });
  } else if (currentTool === 'text') {
    createTextInput(x, y);
  }
  else if (currentTool === 'square' || currentTool === 'circle') {
    isDrawing = true;
    startX = x;
    startY = y;
  }
});
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (currentTool === 'homography') {
    if (draggingHomographyIndex !== null) {
      const elem = homographyElements[draggingHomographyIndex];
      elem.ringCenter.x = x - homographyDragOffset.x;
      elem.ringCenter.y = y - homographyDragOffset.y;
      redrawCanvas();
      return;
    }
    if (homographyDrawing && currentHomographyBox) {
      currentHomographyBox.endX = x;
      currentHomographyBox.endY = y;
      redrawCanvas();
      ctx.save();
      ctx.strokeStyle = "yellow";
      ctx.lineWidth = 2;
      const minX = Math.min(currentHomographyBox.startX, currentHomographyBox.endX);
      const minY = Math.min(currentHomographyBox.startY, currentHomographyBox.endY);
      const width = Math.abs(currentHomographyBox.endX - currentHomographyBox.startX);
      const height = Math.abs(currentHomographyBox.endY - currentHomographyBox.startY);
      ctx.strokeRect(minX, minY, width, height);
      ctx.restore();
      return;
    }
  }

  if (currentTool === 'pen' && isDrawing) {
    const currentStroke = drawnElements[drawnElements.length - 1];
    currentStroke.points.push({ x, y });
    drawSegment(lastX, lastY, x, y, currentStroke.color, currentStroke.width);
    lastX = x;
    lastY = y;
  } else if ((currentTool === 'square' || currentTool === 'circle') && isDrawing) {
    redrawCanvas();
    drawShapePreview(x, y);
  }
});
canvas.addEventListener('mouseup', (e) => {
  if (currentTool === 'homography') {
    if (homographyDrawing && currentHomographyBox) {
      const box = currentHomographyBox;
      const minX = Math.min(box.startX, box.endX);
      const maxX = Math.max(box.startX, box.endX);
      const minY = Math.min(box.startY, box.endY);
      const maxY = Math.max(box.startY, box.endY);
      const ringCenter = { x: (minX + maxX) / 2, y: maxY };
      homographyElements.push({
        box: box,
        ringCenter: ringCenter,
        ringColor: defaultHomographyRingColor,
        spotlightColor: defaultHomographySpotlightColor,
        timestamp: Date.now()
      });
      homographyDrawing = false;
      currentHomographyBox = null;
      redrawCanvas();
      return;
    }
    if (draggingHomographyIndex !== null) {
      draggingHomographyIndex = null;
      return;
    }
  }

  if (currentTool === 'pen') {
    isDrawing = false;
  } else if (currentTool === 'square' || currentTool === 'circle') {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    if (currentTool === 'square') {
      drawnElements.push({
        type: 'square',
        x: Math.min(startX, x),
        y: Math.min(startY, y),
        width: Math.abs(x - startX),
        height: Math.abs(y - startY),
        color: '#00ff9f',
        lineWidth: 2,
        timestamp: Date.now()
      });
    } else {
      const radius = Math.sqrt(Math.pow(x - startX, 2) + Math.pow(y - startY, 2));
      drawnElements.push({
        type: 'circle',
        x: startX,
        y: startY,
        radius: radius,
        color: '#00ff9f',
        lineWidth: 2,
        timestamp: Date.now()
      });
    }
    isDrawing = false;
    redrawCanvas();
  }
});
canvas.addEventListener('mouseleave', () => {
  if (currentTool === 'pen') {
    isDrawing = false;
  }
});

// Context menu for homography overlays
canvas.addEventListener("contextmenu", (e) => {
  if (currentTool === 'homography') {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    for (let i = 0; i < homographyElements.length; i++) {
      const elem = homographyElements[i];
      const ring = elem.ringCenter;
      const dx = x - ring.x, dy = y - ring.y;
      if (Math.sqrt(dx*dx + dy*dy) < 20) {
        e.preventDefault();
        editingHomographyIndex = i;
        document.getElementById("homographyDropdownRingColor").value = elem.ringColor;
        document.getElementById("homographyDropdownSpotlightColor").value = elem.spotlightColor;
        const menu = document.getElementById("homographyContextMenu");
        menu.style.left = e.pageX + "px";
        menu.style.top = e.pageY + "px";
        menu.style.display = "block";
        return;
      }
    }
  }
});
document.getElementById("homographyDropdownSaveBtn").addEventListener("click", () => {
  if (editingHomographyIndex !== null) {
    const newRingColor = document.getElementById("homographyDropdownRingColor").value;
    const newSpotlightColor = document.getElementById("homographyDropdownSpotlightColor").value;
    homographyElements[editingHomographyIndex].ringColor = newRingColor;
    homographyElements[editingHomographyIndex].spotlightColor = newSpotlightColor;
    editingHomographyIndex = null;
    redrawCanvas();
    hideHomographyContextMenu();
  }
});
function hideHomographyContextMenu() {
  document.getElementById("homographyContextMenu").style.display = "none";
}
document.addEventListener("click", (e) => {
  const menu = document.getElementById("homographyContextMenu");
  if (menu && !menu.contains(e.target)) {
    hideHomographyContextMenu();
  }
});
canvas.addEventListener('dblclick', (e) => {
  if (currentTool === 'polygon') {
    if (currentPolygonPoints.length >= 3) {
      drawnElements.push({
        type: 'polygon',
        points: [...currentPolygonPoints],
        color: '#00ff9f',
        timestamp: Date.now()
      });
      currentPolygonPoints = []; // Reset for next shape
      redrawCanvas();
    }
  } else if (currentTool === 'vectorSquare') {
    if (currentPolygonPoints.length >= 2) {
      drawnElements.push({
        type: 'vectorSquare',
        points: [...currentPolygonPoints],
        color: '#00ff9f',
        timestamp: Date.now()
      });
      currentPolygonPoints = []; // Reset for next shape
      redrawCanvas();
    }
  }
});
/********************************************
 * DRAWING HELPER FUNCTIONS (General)
 ********************************************/
function createTextInput(x, y) {
  let initialText = '';
  // Create a new text element
  const textElem = document.createElement('div');
  textElem.className = 'draggable-text';
  textElem.contentEditable = true;
  textElem.innerText = initialText;
  // Set its initial position using pixels
  textElem.style.position = 'absolute';
  textElem.style.left = x + 'px';
  textElem.style.top = y + 'px';
  textElem.style.minWidth = '100px';
  textElem.style.minHeight = '20px';
  textElem.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  textElem.style.color = '#00ff9f';
  textElem.style.padding = '5px';
  textElem.style.border = '1px solid #00ff9f';
  textElem.style.outline = 'none';
  textElem.style.zIndex = '1000';

  // Variables for dragging
  let isDragging = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;
 document.querySelector('.video-container').appendChild(textElem);
  // Allow dragging when not in editing mode.
  textElem.addEventListener('mousedown', (e) => {
    // If not currently editable, start drag.
    if (textElem.contentEditable === "false") {
      isDragging = true;
      dragOffsetX = e.clientX - textElem.offsetLeft;
      dragOffsetY = e.clientY - textElem.offsetTop;
      // Prevent text selection during drag.
      e.preventDefault();
    }
  });

  // Update position on mouse move (attached to document for smooth dragging)
  function onMouseMove(e) {
    if (isDragging) {
      textElem.style.left = (e.clientX - dragOffsetX) + 'px';
      textElem.style.top = (e.clientY - dragOffsetY) + 'px';
    }
  }
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // When the element loses focus, lock it (but keep it on the screen)
  textElem.addEventListener('blur', () => {
    if (textElem.textContent.trim() !== '') {
      // Set contentEditable to false so that the text is locked and dragging can occur
      textElem.contentEditable = "false";
    }
  });

  // On double-click, re-enable editing for later adjustments
  textElem.addEventListener('dblclick', () => {
    textElem.contentEditable = "true";
    // Optionally, place the caret at the end after enabling editing
    setTimeout(() => textElem.focus(), 0);
  });

  // Append the text element to the container (make sure it sits above the canvas)
  document.querySelector('.video-container').appendChild(textElem);
  // Auto-focus the element so the user can immediately type
  setTimeout(() => { textElem.focus(); }, 0);

  return textElem;
}

function createTextInput_(x, y) {
  if (currentTextElement) { currentTextElement.remove(); }
  const textInput = document.createElement('div');
  textInput.contentEditable = true;
  textInput.style.position = 'absolute';
  textInput.style.left = (x / canvas.width) * canvas.clientWidth + 'px';
  textInput.style.top = (y / canvas.height) * canvas.clientHeight + 'px';
  textInput.style.minWidth = '100px';
  textInput.style.minHeight = '20px';
  textInput.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  textInput.style.color = '#00ff9f';
  textInput.style.padding = '5px';
  textInput.style.border = '1px solid #00ff9f';
  textInput.style.outline = 'none';
  textInput.style.zIndex = '1000';

  // Variables for dragging
  let isDragging = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  // When user presses down, start the drag if clicking on the textInput itself.
  textInput.addEventListener('mousedown', (e) => {
    // Prevent interfering with text selection if needed.
    isDragging = true;
    dragOffsetX = e.clientX - textInput.offsetLeft;
    dragOffsetY = e.clientY - textInput.offsetTop;
    // Stop propagation so the canvas doesn't immediately re-trigger other events.
    e.stopPropagation();
  });

  // Listen on document to allow smooth dragging even if the cursor moves outside the element.
  function onMouseMove(e) {
    if (!isDragging) return;
    textInput.style.left = (e.clientX - dragOffsetX) + 'px';
    textInput.style.top = (e.clientY - dragOffsetY) + 'px';
  }
  function onMouseUp() {
    isDragging = false;
  }
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);

  textInput.addEventListener('blur', () => {
    if (textInput.textContent.trim()) {
      // Save the final position from the element's style values.
      drawnElements.push({
        type: 'text',
        x: parseFloat(textInput.style.left),
        y: parseFloat(textInput.style.top),
        text: textInput.textContent,
        color: '#00ff9f',
        timestamp: Date.now()
      });
      redrawCanvas();
    }
    textInput.remove();
    currentTextElement = null;
    // Cleanup the event listeners when done.
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  });

  document.querySelector('.video-container').appendChild(textInput);
  currentTextElement = textInput;
  setTimeout(() => { textInput.focus(); }, 0);
}


function drawSegment(x1, y1, x2, y2, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
function drawShapePreview(x, y) {
  ctx.strokeStyle = '#00ff9f';
  ctx.lineWidth = 2;
  if (currentTool === 'square') {
    ctx.strokeRect(
      Math.min(startX, x),
      Math.min(startY, y),
      Math.abs(x - startX),
      Math.abs(y - startY)
    );
  } else if (currentTool === 'circle') {
    const radius = Math.sqrt(Math.pow(x - startX, 2) + Math.pow(y - startY, 2));
    ctx.beginPath();
    ctx.arc(startX, startY, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}
function redrawCanvas() {
  // Clear the entire canvas.
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw all finalized (stored) elements.
  drawnElements.forEach(element => {
    // Use the element's line width or default to 2.
    ctx.lineWidth = element.lineWidth || element.width || 2;
    // Use the element's color or default.
    ctx.strokeStyle = element.color || '#00ff9f';
    ctx.fillStyle = element.color || '#00ff9f';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (element.type === 'pen') {
      // Draw freehand pen strokes.
      ctx.beginPath();
      const pts = element.points;
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.stroke();
    } else if (element.type === 'square') {
      // Draw a rectangle.
      ctx.strokeRect(element.x, element.y, element.width, element.height);
    } else if (element.type === 'circle') {
      // Draw a circle.
      ctx.beginPath();
      ctx.arc(element.x, element.y, element.radius, 0, Math.PI * 2);
      ctx.stroke();
    } else if (element.type === 'polygon') {
      // Draw a finalized polygon.
      ctx.beginPath();
      ctx.moveTo(element.points[0].x, element.points[0].y);
      for (let i = 1; i < element.points.length; i++) {
        ctx.lineTo(element.points[i].x, element.points[i].y);
      }
      ctx.closePath();
      ctx.stroke();
    } else if (element.type === 'vectorSquare') {
      // Draw connecting lines between vertices.
      ctx.beginPath();
      ctx.moveTo(element.points[0].x, element.points[0].y);
      for (let i = 1; i < element.points.length; i++) {
        ctx.lineTo(element.points[i].x, element.points[i].y);
      }
      // Optionally, you could close the path if desired:
      // ctx.closePath();
      ctx.stroke();
      // Draw small dots at each vertex.
      element.points.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (element.type === 'text') {
      // Draw text.
      ctx.font = '16px Arial';
      ctx.fillText(element.text, element.x, element.y);
    }
  });

  // Draw the preview for an in-progress polygon or vector square.
  if ((currentTool === 'polygon' || currentTool === 'vectorSquare') && currentPolygonPoints.length) {
    ctx.beginPath();
    ctx.moveTo(currentPolygonPoints[0].x, currentPolygonPoints[0].y);
    for (let i = 1; i < currentPolygonPoints.length; i++) {
      ctx.lineTo(currentPolygonPoints[i].x, currentPolygonPoints[i].y);
    }
    ctx.stroke();
    // For vector square mode, also draw small circles at each vertex.
    if (currentTool === 'vectorSquare') {
      currentPolygonPoints.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  // Finally, draw any homography (halo) overlays.
  homographyElements.forEach(elem => {
    drawHaloAndSpotlight(elem.ringCenter, elem.ringColor, elem.spotlightColor);
  });
}


/********************************************
 * UNDO FUNCTION (Combined)
 ********************************************/
function undoLast() {
  let lastGeneral = drawnElements.length ? drawnElements[drawnElements.length - 1] : null;
  let lastHalo = homographyElements.length ? homographyElements[homographyElements.length - 1] : null;
  if (!lastGeneral && !lastHalo) return;
  if (lastGeneral && lastHalo) {
    if (lastGeneral.timestamp > lastHalo.timestamp) {
      drawnElements.pop();
    } else {
      homographyElements.pop();
    }
  } else if (lastGeneral) {
    drawnElements.pop();
  } else {
    homographyElements.pop();
  }
  redrawCanvas();
}

/********************************************
 * SCREENSHOT (Video Editor)
 ********************************************/
function takeScreenshot() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);
  const backupDrawnElements = drawnElements.slice();
  backupDrawnElements.forEach(element => {
    tempCtx.lineWidth = element.lineWidth || element.width || 2;
    tempCtx.strokeStyle = element.color || '#00ff9f';
    tempCtx.fillStyle = element.color || '#00ff9f';
    tempCtx.lineCap = 'round';
    tempCtx.lineJoin = 'round';
    if (element.type === 'pen') {
      tempCtx.beginPath();
      const pts = element.points;
      tempCtx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        tempCtx.lineTo(pts[i].x, pts[i].y);
      }
      tempCtx.stroke();
    } else if (element.type === 'square') {
      tempCtx.strokeRect(element.x, element.y, element.width, element.height);
    } else if (element.type === 'circle') {
      tempCtx.beginPath();
      tempCtx.arc(element.x, element.y, element.radius, 0, Math.PI * 2);
      tempCtx.stroke();
    } else if (element.type === 'polygon') {
      tempCtx.beginPath();
      tempCtx.moveTo(element.points[0].x, element.points[0].y);
      for (let i = 1; i < element.points.length; i++) {
        tempCtx.lineTo(element.points[i].x, element.points[i].y);
      }
      tempCtx.closePath();
      tempCtx.stroke();
    } else if (element.type === 'text') {
      tempCtx.font = '16px Arial';
      tempCtx.fillText(element.text, element.x, element.y);
    }
  });
  homographyElements.forEach(elem => {
    drawHaloAndSpotlight(elem.ringCenter, elem.ringColor, elem.spotlightColor);
  });
  tempCanvas.toBlob((blob) => {
    const link = document.createElement('a');
    link.download = `screenshot-${new Date().toISOString().slice(0, 10)}.png`;
    link.href = URL.createObjectURL(blob);
    link.click();
  }, 'image/png');
}

/********************************************
 * ACTION BUTTONS (Video Editor)
 ********************************************/
function detectAudioEvents() {
  if (!currentVideoPath) {
    showToast("Load a video first");        // nothing to analyse yet
    return;
  }
  sendSocketMessage("detectWhistles", {     // server action we just wired up
    videoPath: currentVideoPath             // absolute path the backend understands
    // modelPath: "/path/to/alt_model.tflite"  // optional override
  });
  showToast("Running whistle detector…");
}
function exportTags() {
  const checkboxes = document.querySelectorAll(".tag-select:checked");
  const selectedTags = [];
  checkboxes.forEach((checkbox) => {
    const parent = checkbox.closest(".tag-item");
    if (parent) {
      const textEl = parent.querySelector(".tag-display");
      selectedTags.push(textEl ? textEl.textContent : "Unknown");
    }
  });
  if (selectedTags.length === 0) {
    Swal.fire({
      background: '#2f2f2f',
      color: '#ffffff',
      icon: 'error',
      title: 'No Tags Selected',
      text: 'No tags selected.'
    });
  } else {
    Swal.fire({
      background: '#2f2f2f',
      color: '#ffffff',
      title: 'Selected Tags',
      html: selectedTags.join('<br>')
    });
  }
}
function trackPlayers() {
  const cb = document.querySelectorAll('.tag-select:checked');
  if (cb.length !== 1) {
    return Swal.fire({
      icon: 'error',
      title: 'Select one clip',
      text: 'Please pick exactly one tagged segment.'
    });
  }
  const item  = cb[0].closest('.tag-item');
  const start = parseFloat(item.dataset.start);
  const end   = parseFloat(item.dataset.end);
  showToast(`Masking from ${start}s to ${end}s`);
  sendSocketMessage('startSegmentation', {
    video:       currentVideoPath,
    checkpoint:  document.getElementById('seg_checkpoint').value,
    config:      document.getElementById('seg_config').value,
    fps:         parseInt(document.getElementById('seg_fps').value, 10),
    startSec:    start,
    endSec:      end
  });
}

/********************************************
 * INIT & HOMOGRAPHY TOOL MODE (Video Editor)
 ********************************************/
function init() {
  updateMainContentSize();
  setupToolButtons();
  // Default to pointer tool
  document.querySelector('[data-btn-id="mouse-pointer"]').click();
  const uploadLabel = document.getElementById("action-upload");
  const videoUploaderFirst = document.getElementById("videoUploaderFirst");
  uploadLabel.addEventListener("click", () => { videoUploaderFirst.click(); });
  videoUploaderFirst.addEventListener("change", () => {
    const file = videoUploaderFirst.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (file.type.startsWith("image/")) {
      videoPlayer.style.display = "none";
      const img = new Image();
      img.onload = () => {
        resizeCanvas();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = url;
    } else {
      videoPlayer.style.display = "block";
      videoPlayer.src = url;
      videoPlayer.load();
    }
  });
}
document.addEventListener('DOMContentLoaded', init);

/********************************************
 * HOMOGRAPHY DRAWING FUNCTIONS
 * (Integrated as a new tool mode)
 ********************************************/
function drawHaloAndSpotlight(center, ringColor, spotlightColor) {
  const cx = center.x;
  const cy = center.y;
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, 40, 15, 0, 0, 2 * Math.PI);
  ctx.strokeStyle = hexToRgba(ringColor, 0.8);
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(cx, cy, 28, 10, 0, 0, 2 * Math.PI);
  ctx.strokeStyle = hexToRgba(ringColor, 0.9);
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.restore();
  drawVerticalSpotlight(cx, cy, spotlightColor);
}
function drawVerticalSpotlight(cx, cy, spotlightColor) {
  let rgb = hexToRgb(spotlightColor);
  let gradient = ctx.createLinearGradient(0, 0, 0, cy);
  gradient.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.0)`);
  gradient.addColorStop(0.1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.15)`);
  gradient.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.0)`);
  const beamWidth = 80;
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.rect(cx - beamWidth/2, 0, beamWidth, cy);
  ctx.clip();
  ctx.fillRect(cx - beamWidth/2, 0, beamWidth, cy);
  ctx.restore();
}
function hexToRgb(hex) {
  hex = hex.replace(/^#/, '');
  if (hex.length === 3) {
    hex = hex.split('').map(c => c + c).join('');
  }
  const num = parseInt(hex, 16);
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}
function hexToRgba(hex, alpha) {
  const rgb = hexToRgb(hex);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}
// ─────────────────────────────────────────────────────────────────────────────
// Save original video‐editor HTML so we can restore it later
// ─────────────────────────────────────────────────────────────────────────────
const mainContentElem         = document.getElementById('mainContent');
const originalMainContentHTML = mainContentElem.innerHTML;
let   currentProjectId       = null;

// ─────────────────────────────────────────────────────────────────────────────
// ➊ showProjects(): hide editor & overlay projects.html in an iframe
// ─────────────────────────────────────────────────────────────────────────────
function showProjects() {
  // 1) hide the video editor
  document.getElementById('videoWrapper').style.display = 'none';

  // 2) allow absolute positioning on mainContent
  mainContentElem.style.position = 'relative';

  // 3) build the iframe
  const iframe = document.createElement('iframe');
  iframe.id  = 'projectsFrame';
  iframe.src = 'projects.html';
  Object.assign(iframe.style, {
    position: 'absolute',
    left:     '0',
    width:    '100%',
    border:   'none'
  });

  // 4) size it below the navbar, on load and on resize
  function resizeIframe() {
    const navH = document.querySelector('.navbar').offsetHeight;
    iframe.style.top    = `${navH}px`;
    iframe.style.height = `${window.innerHeight - navH}px`;
  }
  window.addEventListener('resize', resizeIframe);
  resizeIframe();

  // 5) attach it
  mainContentElem.appendChild(iframe);
}

// ─────────────────────────────────────────────────────────────────────────────
// ➋ onProjectSelected(): tear down iframe, restore editor & fetch videos
// ─────────────────────────────────────────────────────────────────────────────
function onProjectSelected(projectId) {
  currentProjectId = projectId;

  // 1) remove the iframe
  const iframe = document.getElementById('projectsFrame');
  if (iframe) iframe.remove();

  // 2) restore the original editor HTML
  mainContentElem.innerHTML = originalMainContentHTML;

  // 3) re‑run your init / layout to re‑hook everything
  init();              // sets up sidebars, tool buttons, canvas, etc.
  updateMainContentSize(); // corrects margins based on sidebars

  // 4) fetch that project’s videos.json
  const videoDB = `videos_${projectId}.json`;
  sendSocketMessage('fetchVideos', { projectId, videoDBPath: videoDB });

  // (Optionally) reinitialize your video editor (e.g. call init() or similar functions)
  //init(); // if you have an initialization function for the player view
}
