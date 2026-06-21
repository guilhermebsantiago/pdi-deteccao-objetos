import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Camera,
  FileImage,
  Film,
  Gauge,
  ImageUp,
  Play,
  RefreshCcw,
  Server,
  SlidersHorizontal,
  Square,
  Upload
} from "lucide-react";
import {
  detectImage,
  detectVideo,
  Detection,
  getModels,
  ImageDetectionResponse,
  ModelInfo,
  VideoDetectionResponse
} from "./api";
import "./styles.css";

type Mode = "imagem" | "video" | "webcam";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("yolo");
  const [mode, setMode] = useState<Mode>("imagem");
  const [confidence, setConfidence] = useState(0.35);
  const [iou, setIou] = useState(0.45);
  const [imageResult, setImageResult] = useState<ImageDetectionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    getModels()
      .then((items) => {
        setModels(items);
        if (items[0]) setSelectedModel(items[0].id);
      })
      .catch((error: Error) => setMessage(error.message));
  }, []);

  const activeModel = useMemo(
    () => models.find((model) => model.id === selectedModel),
    [models, selectedModel]
  );

  async function runImage(file: File) {
    setLoading(true);
    setMessage("");
    setVideoResult(null);
    try {
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
      const result = await detectImage("/api/detect/frame", blob, "webcam.jpg", selectedModel, confidence, iou);
      setImageResult(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Falha ao processar frame.");
    } finally {
      setLoading(false);
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
      </section>

      <section className="layout">
        <aside className="panel controls">
          <h2>Configuração</h2>

          <label className="field" htmlFor="model-select">
            <span>Modelo</span>
            <select id="model-select" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          {activeModel && (
            <div className={`modelState ${activeModel.available ? "ready" : "demo"}`} role="status">
              <strong>{activeModel.family}</strong>
              <span>{activeModel.status}</span>
            </div>
          )}

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
            {mode === "imagem" && <ImageTester loading={loading} onSubmit={runImage} />}
            {mode === "video" && <VideoTester loading={loading} onSubmit={runVideo} />}
            {mode === "webcam" && <WebcamTester loading={loading} onFrame={runFrame} />}
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

function ImageTester({ loading, onSubmit }: { loading: boolean; onSubmit: (file: File) => void }) {
  return (
    <Uploader
      accept="image/*"
      icon={<ImageUp size={22} />}
      title="Teste com imagem"
      action="Processar imagem"
      loading={loading}
      onSubmit={onSubmit}
    />
  );
}

function VideoTester({ loading, onSubmit }: { loading: boolean; onSubmit: (file: File) => void }) {
  return (
    <Uploader
      accept="video/*"
      icon={<Film size={22} />}
      title="Teste com vídeo"
      action="Processar vídeo"
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
  loading: boolean;
  onSubmit: (file: File) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const inputId = `upload-${props.accept.replace(/[^a-z]/gi, "")}`;

  return (
    <div className="uploader">
      <div className="panelTitle">
        <span aria-hidden="true">{props.icon}</span>
        <h2>{props.title}</h2>
      </div>
      <label className="dropzone" htmlFor={inputId}>
        <Upload size={30} aria-hidden="true" />
        <span>{file ? file.name : "Selecionar arquivo"}</span>
        <input
          id={inputId}
          type="file"
          accept={props.accept}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
      </label>
      <button
        type="button"
        className="primary"
        disabled={!file || props.loading}
        onClick={() => file && props.onSubmit(file)}
      >
        {props.loading ? <span className="spinner" aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
        {props.loading ? "Processando" : props.action}
      </button>
    </div>
  );
}

function WebcamTester({ loading, onFrame }: { loading: boolean; onFrame: (blob: Blob) => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;
    setActive(true);
  }

  function stop() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setActive(false);
  }

  function capture() {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => blob && onFrame(blob), "image/jpeg", 0.92);
  }

  return (
    <div className="webcam">
      <div className="panelTitle">
        <Camera size={22} aria-hidden="true" />
        <h2>Teste com webcam</h2>
      </div>
      <video ref={videoRef} className="cameraPreview" autoPlay muted playsInline aria-label="Prévia da webcam" />
      <div className="buttonRow">
        {!active ? (
          <button type="button" className="secondary" onClick={start}>
            <Camera size={18} aria-hidden="true" />
            Ativar
          </button>
        ) : (
          <button type="button" className="secondary" onClick={stop}>
            <Square size={18} aria-hidden="true" />
            Parar
          </button>
        )}
        <button type="button" className="primary" disabled={!active || loading} onClick={capture}>
          {loading ? <span className="spinner" aria-hidden="true" /> : <RefreshCcw size={18} aria-hidden="true" />}
          {loading ? "Processando" : "Capturar frame"}
        </button>
      </div>
    </div>
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
