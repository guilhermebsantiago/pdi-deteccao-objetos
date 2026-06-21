# Detector de Objetos - PDI

Monorepo com frontend e backend para testar os modelos treinados no Google Colab.

O projeto não treina os modelos. Ele consome os pesos exportados dos notebooks:

- YOLOv8n
- SSDLite320 MobileNetV3

## Estrutura

```text
backend/   API FastAPI para inferência
frontend/  interface web para imagem, vídeo e webcam
```

## Pesos vindos do Colab

Copie os arquivos gerados no Colab para:

```text
backend/storage/models/yolo/best.pt
backend/storage/models/ssdlite/model.pth
```

Os pesos não são versionados porque podem ser grandes. O backend indica no endpoint `/api/models` se cada arquivo foi encontrado.

Também é possível sincronizar os pesos diretamente de links públicos do Google Drive. Configure no `backend/.env`:

```env
YOLO_SOURCE_URL=https://drive.google.com/file/d/ID_DO_ARQUIVO/view?usp=sharing
SSDLITE_SOURCE_URL=https://drive.google.com/file/d/ID_DO_ARQUIVO/view?usp=sharing
```

Depois use o botão **Sincronizar do Drive** no frontend ou chame:

```text
POST /api/models/yolo/sync
POST /api/models/ssdlite/sync
```

## Rodar backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Rodar frontend

```bash
cd frontend
npm install
npm run dev
```

Abra `http://localhost:5173`.

## Fluxos de teste

- Imagem: envia uma imagem para a API e mostra as detecções.
- Vídeo: envia um vídeo, processa os frames no backend e exibe o vídeo anotado.
- Webcam: captura frames no navegador e envia para a API usando o mesmo endpoint de inferência por imagem.

## Observação sobre os formatos

O YOLO usa diretamente o arquivo `.pt` da Ultralytics.

Para o SSDLite, o backend aceita:

- checkpoint com modelo completo salvo por `torch.save(model, path)`;
- checkpoint com `state_dict` compatível com `torchvision.models.detection.ssdlite320_mobilenet_v3_large`;
- arquivo ONNX, se configurado no `.env`.

Se o formato exportado pelo Colab for diferente, o ponto de ajuste fica em `backend/app/services/detectors.py`.
