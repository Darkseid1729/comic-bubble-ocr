import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "";
const bubbleTypes = ["all", "speech", "shout", "thought", "narration"];

const bubbleColors = {
  speech: "#0a7f7a",
  shout: "#e0632e",
  thought: "#6a7f2e",
  narration: "#244a6a",
  unknown: "#1f1a13"
};

const bubbleFills = {
  speech: "rgba(10, 127, 122, 0.12)",
  shout: "rgba(224, 99, 46, 0.14)",
  thought: "rgba(106, 127, 46, 0.12)",
  narration: "rgba(36, 74, 106, 0.12)",
  unknown: "rgba(31, 26, 19, 0.12)"
};

const buildUrl = (path) => {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
};

function OverlayImage({
  src,
  alt,
  bubbles,
  selectedId,
  showOverlay = true,
  size = "large",
  onOpenViewer
}) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  const handleLoad = (event) => {
    setImageSize({
      width: event.target.naturalWidth,
      height: event.target.naturalHeight
    });
  };

  return (
    <div className={`overlay-wrap ${size}`}>
      <img className="preview-image" src={src} alt={alt} onLoad={handleLoad} />
      <button
        className="image-open"
        type="button"
        onClick={() => onOpenViewer?.(src, alt)}
      >
        Open
      </button>
      {showOverlay && imageSize.width ? (
        <svg
          className="overlay"
          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {bubbles.map((bubble) => {
            const [x, y, w, h] = bubble.bbox || [];
            if ([x, y, w, h].some((value) => value == null)) {
              return null;
            }
            const stroke = bubbleColors[bubble.type] || bubbleColors.unknown;
            const fill =
              bubble.id === selectedId
                ? bubbleFills[bubble.type] || bubbleFills.unknown
                : "transparent";
            const isActive = bubble.id === selectedId;
            return (
              <rect
                key={`box-${bubble.id}`}
                className={isActive ? "bubble-box active" : "bubble-box"}
                x={x}
                y={y}
                width={w}
                height={h}
                rx={6}
                ry={6}
                stroke={stroke}
                fill={fill}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}

export default function App() {
  const [runData, setRunData] = useState(null);
  const [filterType, setFilterType] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [file, setFile] = useState(null);
  const [localPreview, setLocalPreview] = useState("");
  const [annotatedUrl, setAnnotatedUrl] = useState("");
  const [inputImageUrl, setInputImageUrl] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [lang, setLang] = useState("eng");
  const [minArea, setMinArea] = useState(500);
  const [useWatershed, setUseWatershed] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [translateEnabled, setTranslateEnabled] = useState(false);
  const [viewMode, setViewMode] = useState("annotated");
  const [activeTab, setActiveTab] = useState("workspace");
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerSrc, setViewerSrc] = useState("");
  const [viewerAlt, setViewerAlt] = useState("Image");
  const [viewerImageSize, setViewerImageSize] = useState({ width: 0, height: 0 });
  const [viewerCanvasSize, setViewerCanvasSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [fitMode, setFitMode] = useState("fit");
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  const dragRef = useRef({ active: false, startX: 0, startY: 0, originX: 0, originY: 0 });
  const viewerCanvasRef = useRef(null);

  const fileInputRef = useRef(null);

  const bubbles = runData?.bubbles || [];
  const metrics = runData?.metrics || {};
  const pipelineImages = runData?.artifacts?.pipeline || [];

  useEffect(() => {
    setSelectedId(bubbles[0]?.id ?? null);
  }, [bubbles]);

  useEffect(() => {
    if (!file) {
      setLocalPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setLocalPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const filtered = useMemo(() => {
    return bubbles.filter((bubble) => {
      const matchesType = filterType === "all" || bubble.type === filterType;
      const matchesText = bubble.text.toLowerCase().includes(searchTerm.toLowerCase());
      return matchesType && matchesText;
    });
  }, [bubbles, filterType, searchTerm]);

  const selected = useMemo(() => {
    return bubbles.find((bubble) => bubble.id === selectedId) || filtered[0];
  }, [bubbles, selectedId, filtered]);

  const orderedBubbles = useMemo(() => {
    return [...bubbles].sort((a, b) => (a.id ?? 0) - (b.id ?? 0));
  }, [bubbles]);

  const jsonPreview = useMemo(() => {
    return JSON.stringify(
      bubbles.map((bubble) => {
        const payload = {
          bubble_id: bubble.id,
          bubble_type: bubble.type,
          text: bubble.text,
          bbox: bubble.bbox,
          confidence: bubble.confidence
        };
        if (bubble.translated_text) {
          payload.translated_text = bubble.translated_text;
        }
        return payload;
      }),
      null,
      2
    );
  }, [bubbles]);

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStatus(`Selected: ${selectedFile.name}`);
      setError("");
    }
  };

  const handleRunOCR = async () => {
    if (!file) {
      setError("Upload an image first.");
      return;
    }

    setIsRunning(true);
    setStatus("Running OCR...");
    setError("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lang", lang);
    formData.append("min_area", String(minArea));
    formData.append("watershed", String(useWatershed));
    formData.append("debug", String(debugMode));
    formData.append("translate", String(translateEnabled));
    formData.append("target_lang", "EN");

    try {
      const endpoint = API_BASE ? `${API_BASE}/api/ocr` : "/api/ocr";
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      const normalized = {
        imageName: data.image_name || file.name,
        processing_time: data.processing_time,
        metrics: {
          bubbles: data.metrics?.bubbles ?? data.bubbles?.length ?? 0,
          avgConfidence: data.metrics?.avgConfidence ?? 0,
          chars: data.metrics?.chars ?? 0,
          wordAccuracy: data.metrics?.wordAccuracy ?? null,
          charAccuracy: data.metrics?.charAccuracy ?? null,
          iou: data.metrics?.iou ?? null,
          typeCounts: data.metrics?.typeCounts ?? {}
        },
        artifacts: data.artifacts || {},
        bubbles: data.bubbles || []
      };

      setRunData(normalized);
      setAnnotatedUrl(buildUrl(normalized.artifacts.annotated_image));
      setInputImageUrl(buildUrl(normalized.artifacts.input_image));
      setStatus("OCR complete.");
      setActiveTab("results");
    } catch (err) {
      setError(`OCR failed: ${err.message}`);
      setStatus("");
    } finally {
      setIsRunning(false);
    }
  };

  const handleShutdown = async () => {
    if (!window.confirm("Are you sure you want to close the application?")) return;
    
    try {
      const endpoint = API_BASE ? `${API_BASE}/api/shutdown` : "/api/shutdown";
      await fetch(endpoint, { method: "POST" });
      // The server will die, so we can just show a message or close the tab
      setStatus("Application closed.");
      setTimeout(() => {
        window.close();
        document.body.innerHTML = '<div style="background:#1a1a1a;color:#fff;height:100vh;display:flex;align-items:center;justify-content:center;font-family:sans-serif;"><h1>Application has been closed. You can now close this tab.</h1></div>';
      }, 500);
    } catch (err) {
      // Server might die before responding, which is fine
      window.close();
      document.body.innerHTML = '<div style="background:#1a1a1a;color:#fff;height:100vh;display:flex;align-items:center;justify-content:center;font-family:sans-serif;"><h1>Application has been closed. You can now close this tab.</h1></div>';
    }
  };

  const handleCopyJson = async () => {
    try {
      await navigator.clipboard.writeText(jsonPreview);
      setStatus("JSON copied to clipboard.");
      setError("");
    } catch (err) {
      setError("Unable to copy JSON. Your browser may block clipboard access.");
    }
  };

  const handleOpenResults = () => {
    const url = buildUrl(runData?.artifacts?.ocr_json);
    if (url) {
      window.open(url, "_blank", "noreferrer");
    }
  };

  const handleOpenAnnotated = () => {
    const url = buildUrl(runData?.artifacts?.annotated_image);
    if (url) {
      window.open(url, "_blank", "noreferrer");
    }
  };

  const handleOpenText = () => {
    const url = buildUrl(runData?.artifacts?.text_output);
    if (url) {
      window.open(url, "_blank", "noreferrer");
    }
  };

  const openViewer = (src, alt) => {
    if (!src) return;
    setViewerSrc(src);
    setViewerAlt(alt || "Image");
    setZoom(1);
    setFitMode("fit");
    setOffset({ x: 0, y: 0 });
    setViewerOpen(true);
  };

  const closeViewer = () => {
    setViewerOpen(false);
  };

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const zoomIn = () => {
    setZoom((prev) => clamp(Number((prev + 0.2).toFixed(2)), 0.5, 5));
  };

  const zoomOut = () => {
    setZoom((prev) => clamp(Number((prev - 0.2).toFixed(2)), 0.5, 5));
  };

  const resetView = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setFitMode("fit");
  };

  const setFit = (mode) => {
    setFitMode(mode);
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  const handleWheel = (event) => {
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.12 : -0.12;
    setZoom((prev) => clamp(Number((prev + delta).toFixed(2)), 0.5, 5));
  };

  const handleViewerImageLoad = (event) => {
    setViewerImageSize({
      width: event.target.naturalWidth,
      height: event.target.naturalHeight
    });
  };

  const handlePointerDown = (event) => {
    if (!viewerOpen) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: offset.x,
      originY: offset.y
    };
    setIsDragging(true);
  };

  const handlePointerMove = (event) => {
    if (!dragRef.current.active) return;
    const deltaX = event.clientX - dragRef.current.startX;
    const deltaY = event.clientY - dragRef.current.startY;
    setOffset({
      x: dragRef.current.originX + deltaX,
      y: dragRef.current.originY + deltaY
    });
  };

  const handlePointerUp = () => {
    dragRef.current.active = false;
    setIsDragging(false);
  };

  const avgConfidence = Math.round((metrics.avgConfidence ?? 0) * 100);
  const compareInput = inputImageUrl || localPreview;
  const compareAnnotated = annotatedUrl;

  useEffect(() => {
    if (!viewerOpen) return;
    const node = viewerCanvasRef.current;
    if (!node) return;

    const updateSize = () => {
      setViewerCanvasSize({
        width: node.clientWidth,
        height: node.clientHeight
      });
    };

    updateSize();

    let observer;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(updateSize);
      observer.observe(node);
    }

    window.addEventListener("resize", updateSize);
    return () => {
      window.removeEventListener("resize", updateSize);
      if (observer) observer.disconnect();
    };
  }, [viewerOpen]);

  const baseScale = useMemo(() => {
    if (!viewerCanvasSize.width || !viewerCanvasSize.height) {
      return 1;
    }
    if (!viewerImageSize.width || !viewerImageSize.height) {
      return 1;
    }
    const scaleX = viewerCanvasSize.width / viewerImageSize.width;
    const scaleY = viewerCanvasSize.height / viewerImageSize.height;
    return fitMode === "fill" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
  }, [fitMode, viewerCanvasSize, viewerImageSize]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◉</span>
          <div>
            <p className="brand-title">Bubble OCR Studio</p>
            <p className="brand-subtitle">Speech Bubble-Aware OCR</p>
          </div>
        </div>
        <div className="header-actions">
          <div className="metric-pill">
            <span>Bubbles</span>
            <strong>{metrics.bubbles ?? bubbles.length}</strong>
          </div>
          <div className="metric-pill">
            <span>Avg conf</span>
            <strong>{avgConfidence}%</strong>
          </div>
          <button className="btn primary" onClick={handleRunOCR} disabled={isRunning}>
            {isRunning ? "Running..." : "Run OCR"}
          </button>
          <button className="btn danger" onClick={handleShutdown} title="Exit Application">
            Close App
          </button>
        </div>
      </header>

      <div className="tabs">
        <button
          className={activeTab === "workspace" ? "active" : ""}
          onClick={() => setActiveTab("workspace")}
        >
          Workspace
        </button>
        <button
          className={activeTab === "pipeline" ? "active" : ""}
          onClick={() => setActiveTab("pipeline")}
        >
          Pipeline
        </button>
        <button
          className={activeTab === "results" ? "active" : ""}
          onClick={() => setActiveTab("results")}
        >
          Results
        </button>
      </div>

      <main>
        {activeTab === "workspace" ? (
          <section className="workspace">
            <div className="panel upload-panel">
              <div className="panel-header">
                <p>Input</p>
                <div className="chip">{runData?.imageName || "No image"}</div>
              </div>
              <div className="panel-canvas">
                {localPreview ? (
                  <img className="preview-image" src={localPreview} alt="Input preview" />
                ) : (
                  <>
                    <div className="canvas-label">Drop a panel image here</div>
                    <div className="canvas-hint">PNG, JPG, WEBP</div>
                  </>
                )}
              </div>
              <div className="panel-footer">
                <input
                  ref={fileInputRef}
                  className="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                />
                <button
                  className="btn ghost"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Upload
                </button>
                <button className="btn primary" onClick={handleRunOCR} disabled={isRunning}>
                  {isRunning ? "Running..." : "Run OCR"}
                </button>
              </div>

              <div className="settings">
                <div className="field">
                  <label htmlFor="lang">Language</label>
                  <select id="lang" value={lang} onChange={(e) => setLang(e.target.value)}>
                    <option value="eng">English</option>
                    <option value="jpn">Japanese</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="minArea">Min bubble area</label>
                  <input
                    id="minArea"
                    type="number"
                    min="100"
                    step="50"
                    value={minArea}
                    onChange={(e) => setMinArea(Number(e.target.value))}
                  />
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={translateEnabled}
                    onChange={(e) => setTranslateEnabled(e.target.checked)}
                  />
                  <span>Translate to English</span>
                </label>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={debugMode}
                    onChange={(e) => setDebugMode(e.target.checked)}
                  />
                  <span>Save pipeline stages</span>
                </label>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={useWatershed}
                    onChange={(e) => setUseWatershed(e.target.checked)}
                  />
                  <span>Use watershed split</span>
                </label>
              </div>
              {status ? <p className="status">{status}</p> : null}
              {error ? <p className="status error">{error}</p> : null}
            </div>

            <div className="panel preview-panel">
              <div className="panel-header">
                <p>Preview</p>
                <div className="segment">
                  {[
                    { key: "raw", label: "Raw" },
                    { key: "annotated", label: "Annotated" },
                    { key: "compare", label: "Compare" }
                  ].map((item) => (
                    <button
                      key={item.key}
                      className={viewMode === item.key ? "active" : ""}
                      onClick={() => setViewMode(item.key)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="preview-stack">
                {viewMode === "raw" ? (
                  compareInput ? (
                    <OverlayImage
                      src={compareInput}
                      alt="Raw input"
                      bubbles={bubbles}
                      selectedId={selectedId}
                      showOverlay
                      size="large"
                      onOpenViewer={openViewer}
                    />
                  ) : (
                    <div className="stage-empty">Run OCR to see the raw image.</div>
                  )
                ) : null}
                {viewMode === "annotated" ? (
                  compareAnnotated ? (
                    <OverlayImage
                      src={compareAnnotated}
                      alt="Annotated"
                      bubbles={bubbles}
                      selectedId={selectedId}
                      showOverlay
                      size="large"
                      onOpenViewer={openViewer}
                    />
                  ) : (
                    <div className="stage-empty">Run OCR to see annotated output.</div>
                  )
                ) : null}
                {viewMode === "compare" ? (
                  compareInput && compareAnnotated ? (
                    <div className="compare-grid">
                      <div>
                        <p className="muted">Raw</p>
                        <OverlayImage
                          src={compareInput}
                          alt="Raw input"
                          bubbles={bubbles}
                          selectedId={selectedId}
                          showOverlay={false}
                          size="small"
                          onOpenViewer={openViewer}
                        />
                      </div>
                      <div>
                        <p className="muted">Annotated</p>
                        <OverlayImage
                          src={compareAnnotated}
                          alt="Annotated"
                          bubbles={bubbles}
                          selectedId={selectedId}
                          showOverlay
                          size="small"
                          onOpenViewer={openViewer}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="stage-empty">Run OCR to compare outputs.</div>
                  )
                ) : null}
              </div>
            </div>

            <div className="panel inspector">
              <div className="panel-header">
                <p>Bubble Inspector</p>
                <div className="pill">{filtered.length} items</div>
              </div>
              <div className="filters">
                {bubbleTypes.map((type) => (
                  <button
                    key={type}
                    className={`chip ${filterType === type ? "active" : ""}`}
                    onClick={() => setFilterType(type)}
                  >
                    {type}
                  </button>
                ))}
                <input
                  className="search"
                  placeholder="Search text..."
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                />
              </div>

              <div className="bubble-list">
                {filtered.map((bubble) => (
                  <button
                    key={bubble.id}
                    className={`bubble-item ${selected?.id === bubble.id ? "active" : ""}`}
                    onClick={() => setSelectedId(bubble.id)}
                  >
                    <span className={`badge ${bubble.type}`}>{bubble.type}</span>
                    <div>
                      <p className="bubble-text">{bubble.text}</p>
                      <p className="bubble-meta">
                        #{bubble.id} - {bubble.bbox.join(", ")}
                      </p>
                    </div>
                    <div className="confidence">
                      <span>{Math.round((bubble.confidence ?? 0) * 100)}%</span>
                      <div className="bar">
                        <div
                          className="bar-fill"
                          style={{ width: `${(bubble.confidence ?? 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              {selected ? (
                <div className="detail">
                  <div>
                    <p className="muted">Selected bubble</p>
                    <h3>
                      #{selected.id} {selected.type}
                    </h3>
                  </div>
                  <p className="detail-text">{selected.text}</p>
                  <div className="detail-grid">
                    <div>
                      <p className="muted">BBox</p>
                      <strong>{selected.bbox.join(", ")}</strong>
                    </div>
                    <div>
                      <p className="muted">Confidence</p>
                      <strong>{Math.round((selected.confidence ?? 0) * 100)}%</strong>
                    </div>
                    <div>
                      <p className="muted">Type</p>
                      <strong>{selected.type}</strong>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        ) : null}

        {activeTab === "pipeline" ? (
          <section className="pipeline">
            <div className="panel">
              <div className="panel-header">
                <p>Pipeline Stages</p>
                <span className="chip">Debug images</span>
              </div>
              <div className="pipeline-gallery">
                {pipelineImages.length ? (
                  pipelineImages.map((stage) => (
                    <div key={stage.title} className="stage-card">
                      <div className="stage-title">{stage.title}</div>
                      <div className="stage-media">
                        <img className="stage-image" src={buildUrl(stage.image)} alt={stage.title} />
                        <button
                          className="image-open"
                          type="button"
                          onClick={() => openViewer(buildUrl(stage.image), stage.title)}
                        >
                          Open
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="stage-empty">
                    Enable "Save pipeline stages" and run OCR to populate this view.
                  </div>
                )}
              </div>
            </div>
          </section>
        ) : null}

        {activeTab === "results" ? (
          <section className="results">
            <div className="results-grid">
              <div className="results-col">
                <div className="panel stats">
                  <div className="panel-header">
                    <p>Run Summary</p>
                    <span className="chip">{runData?.imageName || "No run"}</span>
                  </div>
                  <div className="metric-grid">
                    <div>
                      <p className="muted">Bubbles</p>
                      <h3>{metrics.bubbles ?? 0}</h3>
                    </div>
                    <div>
                      <p className="muted">Characters</p>
                      <h3>{metrics.chars ?? 0}</h3>
                    </div>
                    <div>
                      <p className="muted">Avg confidence</p>
                      <h3>{avgConfidence}%</h3>
                    </div>
                    <div>
                      <p className="muted">Latency</p>
                      <h3>{runData?.processing_time == null ? "--" : `${runData.processing_time.toFixed(2)}s`}</h3>
                    </div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <p>Reading Order</p>
                    <span className="chip">Dialogue</span>
                  </div>
                  <div className="dialogue-list compact">
                    {orderedBubbles.length ? (
                      orderedBubbles.map((bubble) => (
                        <div key={bubble.id} className="dialogue-item">
                          <div className="dialogue-index">#{bubble.id}</div>
                          <div>
                            <div className={`badge ${bubble.type}`}>{bubble.type}</div>
                            <p className="dialogue-text">{bubble.text}</p>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="stage-empty">Run OCR to build the dialogue timeline.</div>
                    )}
                  </div>
                </div>
              </div>

              <div className="results-col">
                <div className="panel json">
                  <div className="panel-header">
                    <p>JSON Preview</p>
                    <button className="btn ghost" onClick={handleCopyJson}>
                      Copy JSON
                    </button>
                  </div>
                  <pre>{jsonPreview}</pre>
                </div>

                <div className="export-bar">
                  <button className="btn ghost" onClick={handleOpenAnnotated}>
                    Open Annotated Image
                  </button>
                  <button className="btn ghost" onClick={handleOpenResults}>
                    Open JSON
                  </button>
                  <button className="btn primary" onClick={handleOpenText}>
                    Open Extracted Text
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : null}
      </main>

      {viewerOpen ? (
        <div className="viewer-backdrop" onClick={closeViewer}>
          <div className="viewer" onClick={(event) => event.stopPropagation()}>
            <div className="viewer-toolbar">
              <p className="viewer-title">{viewerAlt}</p>
              <div className="viewer-actions">
                <button
                  className={fitMode === "fit" ? "viewer-btn active" : "viewer-btn"}
                  type="button"
                  onClick={() => setFit("fit")}
                >
                  Fit
                </button>
                <button
                  className={fitMode === "fill" ? "viewer-btn active" : "viewer-btn"}
                  type="button"
                  onClick={() => setFit("fill")}
                >
                  Fill
                </button>
                <button className="viewer-btn" type="button" onClick={zoomOut}>
                  -
                </button>
                <button className="viewer-btn" type="button" onClick={zoomIn}>
                  +
                </button>
                <button className="viewer-btn" type="button" onClick={resetView}>
                  Reset
                </button>
                <button className="viewer-btn" type="button" onClick={closeViewer}>
                  Close
                </button>
              </div>
            </div>
            <div
              className={`viewer-canvas ${isDragging ? "dragging" : ""}`}
              ref={viewerCanvasRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
              onWheel={handleWheel}
              onDoubleClick={resetView}
            >
              <img
                className="viewer-image"
                src={viewerSrc}
                alt={viewerAlt}
                style={{
                  transform: `translate(-50%, -50%) translate(${offset.x}px, ${offset.y}px) scale(${zoom * baseScale})`
                }}
                draggable={false}
                onLoad={handleViewerImageLoad}
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
