export type ModelInfo = {
  id: string;
  name: string;
  family: string;
  weights_path: string;
  available: boolean;
  status: string;
};

export type Detection = {
  label: string;
  confidence: number;
  box: number[];
};

export type ImageDetectionResponse = {
  model_id: string;
  filename: string;
  width: number;
  height: number;
  detections: Detection[];
  annotated_image: string;
};

export type VideoDetectionResponse = {
  model_id: string;
  filename: string;
  frames_processed: number;
  detections_total: number;
  annotated_video: string;
  content_type: string;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "Falha ao consultar API.");
  }
  return response.json() as Promise<T>;
}

export async function getModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/api/models");
}

export async function detectImage(
  endpoint: "/api/detect/image" | "/api/detect/frame",
  file: File | Blob,
  filename: string,
  modelId: string,
  confidence: number,
  iou: number
): Promise<ImageDetectionResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  form.append("model_id", modelId);
  form.append("confidence", String(confidence));
  form.append("iou", String(iou));
  return request<ImageDetectionResponse>(endpoint, { method: "POST", body: form });
}

export async function detectVideo(
  file: File,
  modelId: string,
  confidence: number,
  iou: number
): Promise<VideoDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("confidence", String(confidence));
  form.append("iou", String(iou));
  return request<VideoDetectionResponse>("/api/detect/video", { method: "POST", body: form });
}
