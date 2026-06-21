# Backend

API em FastAPI para consumir os pesos exportados pelo Colab.

## Contrato dos modelos

O backend lê os caminhos do `.env`:

```env
YOLO_WEIGHTS=storage/models/yolo/best.pt
SSDLITE_WEIGHTS=storage/models/ssdlite/model.pth
```

O YOLO usa o loader da Ultralytics.

O SSDLite usa PyTorch/TorchVision e aceita checkpoint com modelo completo ou `state_dict`.

## Endpoints

- `GET /api/health`
- `GET /api/models`
- `POST /api/detect/image`
- `POST /api/detect/frame`
- `POST /api/detect/video`

Os endpoints de detecção recebem `multipart/form-data` com:

- `file`
- `model_id`
- `confidence`
- `iou`
