import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Camera,
  CloudDownload,
  FileImage,
  Film,
  Gauge,
  ImageUp,
  Moon,
  Play,
  RefreshCcw,
  Server,
  SlidersHorizontal,
  Square,
  Sun,
  Upload
} from "lucide-react";
import {
  detectImage,
  detectVideo,
  Detection,
  getModels,
  ImageDetectionResponse,
  ModelInfo,
  syncModel,
  VideoDetectionResponse
} from "./api";
import "./styles.css";

type Mode = "imagem" | "video" | "webcam";
type Theme = "light" | "dark";

const COLAB_MODEL_LABELS: Record<string, string> = {
  "yolo-caneta": "YOLOv8n Caneta",
  "yolo-maca": "YOLOv8n Maçã",
  "ssdlite-caneta": "SSDLite320 Caneta",
  "ssdlite-maca": "SSDLite320 Maçã"
};

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [mode, setMode] = useState<Mode>("imagem");
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());
  const [confidence, setConfidence] = useState(0.35);
  const [iou, setIou] = useState(0.45);
  const [imageResult, setImageResult] = useState<ImageDetectionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncLoading, setSyncLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  async function loadModels() {
    try {
      const items = await getModels();
      setModels(items);
      setSelectedModel((current) => {
        const colabItems = items.filter((item) => item.id in COLAB_MODEL_LABELS);
        if (colabItems.some((item) => item.id === current)) return current;
        return colabItems[0]?.id ?? "";
      });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao carregar modelos.");
    }
  }

  const colabModels = useMemo(
    () => models.filter((model) => model.id in COLAB_MODEL_LABELS),
    [models]
  );

  const activeModel = useMemo(
    () => colabModels.find((model) => model.id === selectedModel),
    [colabModels, selectedModel]
  );

  async function runImage(file: File) {
    setLoading(true);
    setMessage("");
    setVideoResult(null);
    try {
      if (!activeModel) throw new Error("Nenhum modelo do Colab selecionado.");
      const result = await detectImage("/api/detect/image", file, file.name, selectedModel, confidence, iou);
      setImageResult(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao processar imagem.");
    } finally {
      setLoading(false);
    }
  }

  async function runVideo(file: File) {
    setLoading(true);
    setMessage("");
    setImageResult(null);
    try {
      if (!activeModel) throw new Error("Nenhum modelo do Colab selecionado.");
      const result = await detectVideo(file, selectedModel, confidence, iou);
      setVideoResult(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao processar vídeo.");
    } finally {
      setLoading(false);
    }
  }

  async function runFrame(blob: Blob) {
    setLoading(true);
    setMessage("");
    setVideoResult(null);
    try {
      if (!activeModel) throw new Error("Nenhum modelo do Colab selecionado.");
      const result = await detectImage("/api/detect/frame", blob, "webcam.jpg", selectedModel, confidence, iou);
      setImageResult(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao processar frame.");
    } finally {
      setLoading(false);
    }
  }

  async function syncSelectedModel() {
    setSyncLoading(true);
    setMessage("");
    try {
      if (!activeModel) throw new Error("Nenhum modelo do Colab selecionado.");
      const response = await syncModel(selectedModel);
      setMessage(response.status);
      await loadModels();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao sincronizar modelo.");
    } finally {
      setSyncLoading(false);
    }
  }

  return (
    <main className="shell" aria-busy={loading}>
      <section className="topbar" aria-labelledby="page-title">
        <div>
          <span className="eyebrow">Projeto PDI</span>
          <h1 id="page-title">Laboratório de Detecção de Objetos</h1>
        </div>
        <div className="statusPill" aria-label="Backend configurado com FastAPI">
          <Server size={18} aria-hidden="true" />
          API FastAPI
        </div>
        <button
          type="button"
          className="themeToggle"
          aria-label={theme === "dark" ? "Ativar tema claro" : "Ativar tema escuro"}
          onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? <Sun size={18} aria-hidden="true" /> : <Moon size={18} aria-hidden="true" />}
          {theme === "dark" ? "Claro" : "Escuro"}
        </button>
      </section>

      <section className="layout">
        <aside className="panel controls">
          <h2>Configuração</h2>

          <div className="field">
            <span id="model-select-label">Modelo</span>
            {colabModels.length ? (
              <select
                id="model-select"
                aria-labelledby="model-select-label"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {colabModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {COLAB_MODEL_LABELS[model.id]}
                  </option>
                ))}
              </select>
            ) : (
              <div className="modelEmpty" role="status">
                Nenhum modelo final do Colab encontrado.
              </div>
            )}
          </div>

          {activeModel && (
            <div className={`modelState ${activeModel.available ? "ready" : "demo"}`} role="status">
              <strong>{activeModel.family}</strong>
              <span>{activeModel.status}</span>
            </div>
          )}

          <button
            type="button"
            className="secondary syncButton"
            disabled={!activeModel?.source_configured || syncLoading || loading}
            onClick={syncSelectedModel}
          >
            {syncLoading ? (
              <span className="spinner" aria-hidden="true" />
            ) : (
              <CloudDownload size={18} aria-hidden="true" />
            )}
            {syncLoading ? "Sincronizando" : "Sincronizar do Drive"}
          </button>

          <div className="segmented" role="group" aria-label="Tipo de teste">
            <button
              type="button"
              className={mode === "imagem" ? "active" : ""}
              aria-pressed={mode === "imagem"}
              onClick={() => setMode("imagem")}
            >
              <FileImage size={18} aria-hidden="true" />
              Imagem
            </button>
            <button
              type="button"
              className={mode === "video" ? "active" : ""}
              aria-pressed={mode === "video"}
              onClick={() => setMode("video")}
            >
              <Film size={18} aria-hidden="true" />
              Vídeo
            </button>
            <button
              type="button"
              className={mode === "webcam" ? "active" : ""}
              aria-pressed={mode === "webcam"}
              onClick={() => setMode("webcam")}
            >
              <Camera size={18} aria-hidden="true" />
              Webcam
            </button>
          </div>

          <Slider
            icon={<Gauge size={18} aria-hidden="true" />}
            label="Confiança"
            value={confidence}
            min={0.05}
            max={0.95}
            step={0.05}
            onChange={setConfidence}
          />
          <Slider
            icon={<SlidersHorizontal size={18} aria-hidden="true" />}
            label="IoU"
            value={iou}
            min={0.1}
            max={0.9}
            step={0.05}
            onChange={setIou}
          />
        </aside>

        <section className="workspace">
          <div className="panel testPanel">
            {mode === "imagem" && <ImageTester disabled={!activeModel} loading={loading} onSubmit={runImage} />}
            {mode === "video" && <VideoTester disabled={!activeModel} loading={loading} onSubmit={runVideo} />}
            {mode === "webcam" && (
              <WebcamTester
                disabled={!activeModel}
                loading={loading}
                onFrame={runFrame}
                modelId={selectedModel}
                confidence={confidence}
                iou={iou}
                onError={setMessage}
              />
            )}
          </div>

          {message && (
            <div className="alert" role="alert">
              {message}
            </div>
          )}

          <ResultPanel imageResult={imageResult} videoResult={videoResult} loading={loading} />
        </section>
      </section>
    </main>
  );
}

function getInitialTheme(): Theme {
  const storedTheme = localStorage.getItem("theme");
  if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function Slider(props: {
  icon: React.ReactNode;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider">
      <span>
        {props.icon}
        {props.label}
        <strong>{props.value.toFixed(2)}</strong>
      </span>
      <input
        aria-label={props.label}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ImageTester({
  disabled,
  loading,
  onSubmit
}: {
  disabled: boolean;
  loading: boolean;
  onSubmit: (file: File) => void;
}) {
  return (
    <Uploader
      accept="image/*"
      icon={<ImageUp size={22} />}
      title="Teste com imagem"
      action="Processar imagem"
      disabled={disabled}
      loading={loading}
      onSubmit={onSubmit}
    />
  );
}

function VideoTester({
  disabled,
  loading,
  onSubmit
}: {
  disabled: boolean;
  loading: boolean;
  onSubmit: (file: File) => void;
}) {
  return (
    <Uploader
      accept="video/*"
      icon={<Film size={22} />}
      title="Teste com vídeo"
      action="Processar vídeo"
      disabled={disabled}
      loading={loading}
      onSubmit={onSubmit}
    />
  );
}

function Uploader(props: {
  accept: string;
  icon: React.ReactNode;
  title: string;
  action: string;
  disabled: boolean;
  loading: boolean;
  onSubmit: (file: File) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function selectFile(nextFile: File | undefined) {
    if (!nextFile) return;
    setFile(nextFile);
  }

  return (
    <div className="uploader">
      <div className="panelTitle">
        <span aria-hidden="true">{props.icon}</span>
        <h2>{props.title}</h2>
      </div>
      <div
        className="dropzone"
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          selectFile(event.dataTransfer.files[0]);
        }}
      >
        <Upload size={30} aria-hidden="true" />
        <span>{file ? file.name : "Arraste um arquivo ou selecione manualmente"}</span>
        <button type="button" className="secondary fileButton">
          Selecionar arquivo
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={props.accept}
          onChange={(event) => selectFile(event.target.files?.[0])}
        />
      </div>
      <button
        type="button"
        className="primary"
        disabled={!file || props.disabled || props.loading}
        onClick={() => file && props.onSubmit(file)}
      >
        {props.loading ? <span className="spinner" aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
        {props.loading ? "Processando" : props.action}
      </button>
    </div>
  );
}

function WebcamTester({
  disabled,
  loading,
  onFrame,
  modelId,
  confidence,
  iou,
  onError
}: {
  disabled: boolean;
  loading: boolean;
  onFrame: (blob: Blob) => void;
  modelId: string;
  confidence: number;
  iou: number;
  onError: (message: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveIntervalRef = useRef<number | null>(null);
  const processingRef = useRef(false);
  const [active, setActive] = useState(false);
  const [live, setLive] = useState(false);
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const [liveSize, setLiveSize] = useState({ width: 0, height: 0 });

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;
    setActive(true);
  }

  function stop() {
    stopLive();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setActive(false);
    setLiveDetections([]);
  }

  function capture(sendFrame = onFrame) {
    const video = videoRef.current;
    if (!video) return;
    if (!video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => blob && sendFrame(blob), "image/jpeg", 0.92);
  }

  function startLive() {
    if (!active || liveIntervalRef.current !== null) return;
    setLive(true);

    const sendLiveFrame = async (blob: Blob) => {
      if (processingRef.current) return;
      processingRef.current = true;
      try {
        const result = await detectImage("/api/detect/frame", blob, "webcam.jpg", modelId, confidence, iou);
        setLiveDetections(result.detections);
        setLiveSize({ width: result.width, height: result.height });
      } catch (error) {
        onError(error instanceof Error ? error.message : "Falha ao processar webcam ao vivo.");
        stopLive();
      } finally {
        processingRef.current = false;
      }
    };

    capture(sendLiveFrame);
    liveIntervalRef.current = window.setInterval(() => capture(sendLiveFrame), 1000);
  }

  function stopLive() {
    if (liveIntervalRef.current !== null) {
      window.clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    processingRef.current = false;
    setLive(false);
  }

  return (
    <div className="webcam">
      <div className="panelTitle">
        <Camera size={22} aria-hidden="true" />
        <h2>Teste com webcam</h2>
      </div>
      <div className="cameraStage">
        <video ref={videoRef} className="cameraPreview" autoPlay muted playsInline aria-label="Prévia da webcam" />
        {live && <LiveOverlay detections={liveDetections} size={liveSize} />}
      </div>
      <div className="buttonRow">
        {!active ? (
          <button type="button" className="secondary" disabled={disabled} onClick={start}>
            <Camera size={18} aria-hidden="true" />
            Ativar
          </button>
        ) : (
          <button type="button" className="secondary" onClick={stop}>
            <Square size={18} aria-hidden="true" />
            Parar
          </button>
        )}
        <button type="button" className="primary" disabled={!active || disabled || loading} onClick={() => capture()}>
          {loading ? <span className="spinner" aria-hidden="true" /> : <RefreshCcw size={18} aria-hidden="true" />}
          {loading ? "Processando" : "Capturar frame"}
        </button>
        {!live ? (
          <button type="button" className="primary" disabled={!active || disabled} onClick={startLive}>
            <Play size={18} aria-hidden="true" />
            Ao vivo 1 FPS
          </button>
        ) : (
          <button type="button" className="secondary" onClick={stopLive}>
            <Square size={18} aria-hidden="true" />
            Parar ao vivo
          </button>
        )}
      </div>
    </div>
  );
}

function LiveOverlay({ detections, size }: { detections: Detection[]; size: { width: number; height: number } }) {
  if (!size.width || !size.height) return null;

  return (
    <svg className="liveOverlay" viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none">
      {detections.map((detection, index) => {
        const [x1, y1, x2, y2] = detection.box;
        const width = Math.max(x2 - x1, 0);
        const height = Math.max(y2 - y1, 0);
        const label = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
        const labelWidth = Math.max(label.length * 9 + 10, 46);
        const labelY = y1 > 22 ? y1 - 22 : y1 + 2;
        const textY = labelY + 16;

        return (
          <g key={`${detection.label}-${index}`}>
            <rect x={x1} y={y1} width={width} height={height} />
            <rect className="labelBg" x={x1} y={labelY} width={labelWidth} height={20} />
            <text x={x1 + 5} y={textY}>{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ResultPanel({
  imageResult,
  videoResult,
  loading
}: {
  imageResult: ImageDetectionResponse | null;
  videoResult: VideoDetectionResponse | null;
  loading: boolean;
}) {
  const detections: Detection[] = imageResult?.detections ?? [];

  return (
    <section className="panel resultPanel">
      <div className="panelTitle">
        <Gauge size={22} aria-hidden="true" />
        <h2>Resultado</h2>
      </div>

      {loading && (
        <div className="empty" role="status">
          <span className="spinner large" aria-hidden="true" />
          Processando arquivo...
        </div>
      )}

      {!loading && imageResult && (
        <div className="resultGrid">
          <img className="preview" src={imageResult.annotated_image} alt="Resultado anotado" />
          <DetectionTable detections={detections} />
        </div>
      )}

      {!loading && videoResult && (
        <div className="videoResult">
          <video className="preview" src={videoResult.annotated_video} controls aria-label="Vídeo anotado" />
          <div className="metrics">
            <span>{videoResult.frames_processed} frames</span>
            <span>{videoResult.detections_total} detecções</span>
          </div>
        </div>
      )}

      {!loading && !imageResult && !videoResult && <div className="empty">Nenhum teste executado.</div>}
    </section>
  );
}

function DetectionTable({ detections }: { detections: Detection[] }) {
  if (!detections.length) return <div className="empty small">Nenhuma detecção acima do limiar.</div>;

  return (
    <table>
      <thead>
        <tr>
          <th>Classe</th>
          <th>Confiança</th>
          <th>Box</th>
        </tr>
      </thead>
      <tbody>
        {detections.map((detection, index) => (
          <tr key={`${detection.label}-${index}`}>
            <td>{detection.label}</td>
            <td>{(detection.confidence * 100).toFixed(1)}%</td>
            <td>{detection.box.map((value) => Math.round(value)).join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
