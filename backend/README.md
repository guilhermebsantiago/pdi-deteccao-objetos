# Backend

API em FastAPI para consumir os pesos exportados pelo Colab.

## Contrato dos modelos

O backend lê os caminhos do `.env`:

```env
YOLO_WEIGHTS=storage/models/yolo/best.pt
YOLO_SOURCE_URL=https://drive.google.com/file/d/ID_DO_ARQUIVO/view?usp=sharing
SSDLITE_WEIGHTS=storage/models/ssdlite/model.pth
SSDLITE_SOURCE_URL=https://drive.google.com/file/d/ID_DO_ARQUIVO/view?usp=sharing
```

O YOLO usa o loader da Ultralytics.

O SSDLite usa PyTorch/TorchVision e aceita checkpoint com modelo completo ou `state_dict`.

## Endpoints

- `GET /api/health`
- `GET /api/models`
- `POST /api/models/{model_id}/sync`
- `POST /api/detect/image`
- `POST /api/detect/frame`
- `POST /api/detect/video`

Os endpoints de detecção recebem `multipart/form-data` com:

- `file`
- `model_id`
- `confidence`
- `iou`

## Sincronizar pesos do Colab

No Colab, salve o arquivo final no Google Drive e gere um link compartilhável público. A API baixa esse arquivo para o caminho configurado em `YOLO_WEIGHTS` ou `SSDLITE_WEIGHTS`.

O backend usa os pesos locais após o download. Se um modelo já estava carregado em memória, o cache dele é limpo após a sincronização.
