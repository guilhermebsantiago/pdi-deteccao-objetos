# Backend

API FastAPI para consumir os pesos exportados pelos Colabs.

## Instalar

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Rodar

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Contrato dos modelos

O backend le `backend/.env` mesmo quando o processo e iniciado fora da pasta `backend`.

Principais variaveis:

```env
YOLO_CANETA_WEIGHTS=storage/models/yolo/caneta.pt
YOLO_CANETA_SOURCE_URL=https://drive.google.com/drive/folders/ID_DA_PASTA
YOLO_CANETA_SOURCE_FILE=yolo_caneta_best.pt

YOLO_MACA_WEIGHTS=storage/models/yolo/maca.pt
YOLO_MACA_SOURCE_URL=https://drive.google.com/drive/folders/ID_DA_PASTA
YOLO_MACA_SOURCE_FILE=yolo_maca_best.pt

SSDLITE_CANETA_WEIGHTS=storage/models/ssdlite/caneta.pth
SSDLITE_CANETA_SOURCE_URL=https://drive.google.com/drive/folders/ID_DA_PASTA
SSDLITE_CANETA_SOURCE_FILE=ssdlite_caneta_best.pth

SSDLITE_MACA_WEIGHTS=storage/models/ssdlite/maca.pth
SSDLITE_MACA_SOURCE_URL=https://drive.google.com/drive/folders/ID_DA_PASTA
SSDLITE_MACA_SOURCE_FILE=ssdlite_maca_best.pth
```

Se `SOURCE_URL` for uma pasta do Drive, o backend baixa a pasta para um diretorio temporario e procura o arquivo definido em `SOURCE_FILE`.

## Modelos cadastrados

- `yolo-caneta`
- `yolo-maca`
- `yolo`
- `ssdlite-caneta`
- `ssdlite-maca`
- `ssdlite`

O fluxo principal usa os modelos separados por base. `yolo` e `ssdlite` ficam como compatibilidade.

## Endpoints

```text
GET  /api/health
GET  /api/models
POST /api/models/{model_id}/sync
POST /api/detect/image
POST /api/detect/frame
POST /api/detect/video
```

Os endpoints de deteccao recebem `multipart/form-data` com:

- `file`
- `model_id`
- `confidence`
- `iou`

## Sincronizacao

Exemplos:

```text
POST /api/models/yolo-caneta/sync
POST /api/models/yolo-maca/sync
POST /api/models/ssdlite-caneta/sync
POST /api/models/ssdlite-maca/sync
```

Apos baixar o peso, o cache do modelo em memoria e limpo para a proxima inferencia carregar o arquivo novo.
