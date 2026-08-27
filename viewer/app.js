const state = {
  trace: null,
  traceBase: "",
  index: 0,
  playing: false,
  speed: 1,
  timer: null,
  preload: [],
};

const elements = {
  app: document.getElementById("app"),
  traceSelect: document.getElementById("trace-select"),
  traceKicker: document.getElementById("trace-kicker"),
  traceTitle: document.getElementById("trace-title"),
  runStatus: document.getElementById("run-status"),
  instruction: document.getElementById("instruction"),
  metricStep: document.getElementById("metric-step"),
  metricAction: document.getElementById("metric-action"),
  metricDistance: document.getElementById("metric-distance"),
  metricSuccess: document.getElementById("metric-success"),
  metricDuration: document.getElementById("metric-duration"),
  cameraFrame: document.getElementById("camera-frame"),
  cameraLoading: document.getElementById("camera-loading"),
  frameTime: document.getElementById("frame-time"),
  frameIndex: document.getElementById("frame-index"),
  navmeshImage: document.getElementById("navmesh-image"),
  mapStage: document.getElementById("map-stage"),
  routeMap: document.getElementById("route-map"),
  referenceRoute: document.getElementById("reference-route"),
  actualRoute: document.getElementById("actual-route"),
  completedRoute: document.getElementById("completed-route"),
  goalRadius: document.getElementById("goal-radius"),
  startMarker: document.getElementById("start-marker"),
  goalMarker: document.getElementById("goal-marker"),
  agentMarker: document.getElementById("agent-marker"),
  previousButton: document.getElementById("previous-button"),
  playButton: document.getElementById("play-button"),
  playIcon: document.getElementById("play-icon"),
  playLabel: document.getElementById("play-label"),
  nextButton: document.getElementById("next-button"),
  timeline: document.getElementById("timeline"),
  speedSelect: document.getElementById("speed-select"),
  actionStrip: document.getElementById("action-strip"),
  poseDetail: document.getElementById("pose-detail"),
  errorMessage: document.getElementById("error-message"),
};

const actionLabels = {
  start: "起点",
  forward: "前进",
  turn_left: "左转",
  turn_right: "右转",
};

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function pathPoints(points) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function setPlaying(playing) {
  state.playing = playing;
  elements.playIcon.textContent = playing ? "Ⅱ" : "▶";
  elements.playLabel.textContent = playing ? "暂停" : "播放";
  elements.playButton.setAttribute("aria-label", playing ? "暂停轨迹" : "播放轨迹");
  if (!playing && state.timer !== null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
}

function scheduleNext() {
  if (!state.playing || !state.trace) return;
  const frames = state.trace.frames;
  if (state.index >= frames.length - 1) {
    setPlaying(false);
    return;
  }
  const current = frames[state.index];
  const next = frames[state.index + 1];
  const recordedDelay = Math.max(0.3, Math.min(2.2, next.elapsed_s - current.elapsed_s));
  state.timer = window.setTimeout(() => {
    setIndex(state.index + 1);
    scheduleNext();
  }, (recordedDelay * 1000) / state.speed);
}

function togglePlay() {
  if (!state.trace) return;
  if (state.playing) {
    setPlaying(false);
    return;
  }
  if (state.index >= state.trace.frames.length - 1) setIndex(0);
  setPlaying(true);
  scheduleNext();
}

function setIndex(nextIndex) {
  if (!state.trace) return;
  const maximum = state.trace.frames.length - 1;
  state.index = Math.max(0, Math.min(maximum, Number(nextIndex)));
  renderFrame();
}

function renderFrame() {
  const trace = state.trace;
  const frame = trace.frames[state.index];
  const imageUrl = `${state.traceBase}/${frame.image}`;
  if (elements.cameraFrame.src !== new URL(imageUrl, window.location.href).href) {
    elements.cameraLoading.hidden = false;
    elements.cameraFrame.src = imageUrl;
  }
  elements.timeline.value = String(state.index);
  elements.metricStep.textContent = `${state.index} / ${trace.action_count}`;
  elements.metricAction.textContent = actionLabels[frame.action] || frame.action;
  elements.metricDistance.textContent = Number.isFinite(frame.distance_to_goal)
    ? `${frame.distance_to_goal.toFixed(2)} m`
    : "—";
  elements.frameTime.textContent = formatTime(frame.elapsed_s);
  elements.frameIndex.textContent = `FRAME ${String(state.index).padStart(3, "0")}`;
  elements.poseDetail.textContent = `x ${frame.world.x.toFixed(2)} · z ${frame.world.z.toFixed(2)} · yaw ${(
    (frame.yaw * 180) /
    Math.PI
  ).toFixed(1)}°`;

  const completed = trace.map.path.slice(0, state.index + 1);
  elements.completedRoute.setAttribute("points", pathPoints(completed));
  elements.agentMarker.setAttribute(
    "transform",
    `translate(${frame.map.x} ${frame.map.y}) rotate(${(
      (Math.atan2(frame.map.heading_y, frame.map.heading_x) * 180) /
        Math.PI +
      90
    ).toFixed(2)})`,
  );

  Array.from(elements.actionStrip.children).forEach((button, index) => {
    button.classList.toggle("is-complete", index + 1 <= state.index);
    button.classList.toggle("is-current", index + 1 === state.index);
  });
}

function buildActionStrip(trace) {
  elements.actionStrip.replaceChildren();
  elements.actionStrip.style.setProperty("--action-count", trace.action_count);
  trace.frames.slice(1).forEach((frame, actionIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-step";
    button.dataset.action = frame.action;
    button.setAttribute(
      "aria-label",
      `第 ${actionIndex + 1} 步，${actionLabels[frame.action] || frame.action}`,
    );
    button.title = `${actionIndex + 1}. ${actionLabels[frame.action] || frame.action}`;
    button.addEventListener("click", () => {
      setPlaying(false);
      setIndex(actionIndex + 1);
    });
    elements.actionStrip.append(button);
  });
}

function configureMap(trace) {
  const map = trace.map;
  elements.routeMap.setAttribute("viewBox", `0 0 ${map.width} ${map.height}`);
  elements.routeMap.setAttribute("preserveAspectRatio", "none");
  elements.mapStage.style.aspectRatio = `${map.width} / ${map.height}`;
  elements.navmeshImage.src = `${state.traceBase}/${map.image}`;
  elements.navmeshImage.alt = `${trace.scene_id} 的 Habitat navmesh`;
  elements.actualRoute.setAttribute("points", pathPoints(map.path));
  elements.referenceRoute.setAttribute("points", pathPoints(map.reference_path));
  const start = map.path[0];
  elements.startMarker.setAttribute("cx", start.x);
  elements.startMarker.setAttribute("cy", start.y);
  const goal = map.goals[0] || map.path[map.path.length - 1];
  elements.goalMarker.setAttribute("cx", goal.x);
  elements.goalMarker.setAttribute("cy", goal.y);
  elements.goalRadius.setAttribute("cx", goal.x);
  elements.goalRadius.setAttribute("cy", goal.y);
  elements.goalRadius.setAttribute("r", map.goal_radius_m / map.meters_per_pixel);
}

function preloadFrames(trace) {
  state.preload = trace.frames.map((frame) => {
    const image = new Image();
    image.src = `${state.traceBase}/${frame.image}`;
    return image;
  });
}

async function loadTrace(path) {
  setPlaying(false);
  elements.app.setAttribute("aria-busy", "true");
  elements.errorMessage.hidden = true;
  const response = await fetch(`data/${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`轨迹加载失败：HTTP ${response.status}`);
  const trace = await response.json();
  state.trace = trace;
  state.traceBase = `data/${path.replace(/\/[^/]+$/, "")}`;
  state.index = 0;

  elements.traceKicker.textContent = `${trace.model} · ${trace.benchmark}`;
  elements.traceTitle.textContent = trace.case_id;
  elements.instruction.textContent = trace.instruction;
  elements.runStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>${
    trace.metrics.sr > 0 ? "成功到达" : "未到达"
  }</span>`;
  elements.metricSuccess.textContent = `${Number(trace.metrics.sr).toFixed(0)} / ${Number(
    trace.metrics.spl,
  ).toFixed(2)}`;
  elements.metricDuration.textContent = formatTime(trace.duration_s);
  elements.timeline.max = String(trace.frames.length - 1);

  configureMap(trace);
  buildActionStrip(trace);
  preloadFrames(trace);
  renderFrame();
  elements.app.setAttribute("aria-busy", "false");
}

async function initialize() {
  try {
    const response = await fetch("data/index.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`轨迹索引加载失败：HTTP ${response.status}`);
    const index = await response.json();
    if (!index.traces.length) throw new Error("没有可用轨迹");
    index.traces.forEach((trace) => {
      const option = document.createElement("option");
      option.value = trace.path;
      option.textContent = trace.label;
      elements.traceSelect.append(option);
    });
    await loadTrace(index.traces[0].path);
  } catch (error) {
    elements.errorMessage.textContent = error instanceof Error ? error.message : String(error);
    elements.errorMessage.hidden = false;
    elements.app.setAttribute("aria-busy", "false");
  }
}

elements.cameraFrame.addEventListener("load", () => {
  elements.cameraLoading.hidden = true;
});
elements.playButton.addEventListener("click", togglePlay);
elements.previousButton.addEventListener("click", () => {
  setPlaying(false);
  setIndex(state.index - 1);
});
elements.nextButton.addEventListener("click", () => {
  setPlaying(false);
  setIndex(state.index + 1);
});
elements.timeline.addEventListener("input", (event) => {
  setPlaying(false);
  setIndex(event.target.value);
});
elements.speedSelect.addEventListener("change", (event) => {
  state.speed = Number(event.target.value);
  if (state.playing) {
    setPlaying(false);
    setPlaying(true);
    scheduleNext();
  }
});
elements.traceSelect.addEventListener("change", async (event) => {
  try {
    await loadTrace(event.target.value);
  } catch (error) {
    elements.errorMessage.textContent = error instanceof Error ? error.message : String(error);
    elements.errorMessage.hidden = false;
  }
});
document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    togglePlay();
  } else if (event.key === "ArrowLeft") {
    setPlaying(false);
    setIndex(state.index - 1);
  } else if (event.key === "ArrowRight") {
    setPlaying(false);
    setIndex(state.index + 1);
  }
});

initialize();
