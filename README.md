# Detector de Objetos - PDI

Monorepo com frontend React e backend FastAPI para testar os modelos treinados nos Colabs do trabalho de PDI.

O projeto nao treina os modelos. Ele consome os pesos finais exportados pelos notebooks:

- YOLOv8n
- SSDLite320 MobileNetV3

## Estrutura

```text
backend/   API FastAPI para inferencia e sincronizacao dos pesos
frontend/  interface web para testar imagem, video e webcam
```

## Requisitos

- Python 3.12 recomendado. O Python 3.13 pode falhar ao instalar `torch==2.5.1`.
- Node.js 20 ou superior.
- Acesso aos links publicos das pastas do Google Drive com os modelos.

## Configurar backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite `backend/.env` se precisar trocar os links das pastas. A configuracao padrao espera estes nomes dentro da pasta de modelos:

```text
ssdlite_caneta_best.pth
ssdlite_maca_best.pth
yolo_caneta_best.pt
yolo_maca_best.pt
```

Os caminhos locais padrao dos pesos sao:

```text
backend/storage/models/ssdlite/caneta.pth
backend/storage/models/ssdlite/maca.pth
backend/storage/models/yolo/caneta.pt
backend/storage/models/yolo/maca.pt
```

Inicie a API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

A API fica em `http://127.0.0.1:8000`.

## Configurar frontend

Em outro terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Abra `http://127.0.0.1:5173`.

## Sincronizar modelos do Drive

No frontend, selecione o modelo e clique em **Sincronizar do Drive**.

Modelos esperados:

- `YOLOv8n Caneta treinado no Colab`
- `YOLOv8n Maca treinado no Colab`
- `SSDLite320 Caneta treinado no Colab`
- `SSDLite320 Maca treinado no Colab`

Tambem e possivel chamar a API diretamente:

```text
POST /api/models/yolo-caneta/sync
POST /api/models/yolo-maca/sync
POST /api/models/ssdlite-caneta/sync
POST /api/models/ssdlite-maca/sync
```

O endpoint baixa a pasta do Drive em uma area temporaria, encontra o arquivo esperado pelo nome e substitui o peso local. Se o modelo ja estava carregado em memoria, o cache dele e limpo apos o sync.

## Rodar testes na interface

A tela permite testar:

- Imagem: envia uma imagem para a API e mostra as caixas detectadas.
- Video: envia um video e recebe um video anotado.
- Webcam: captura frame manualmente ou roda inferencia ao vivo em 1 FPS com overlay.

Os controles de `Confianca` e `IoU` sao enviados para os endpoints de inferencia.

## Endpoints principais

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

## Observacoes

- Os pesos nao sao versionados no Git.
- Se os pesos ainda nao foram sincronizados, o backend usa modo demonstracao quando `DEMO_MODE=true`.
- Os modelos SSDLite por base usam duas classes: `background + objeto`.
- O modelo `ssdlite` geral e o `yolo` geral ficam cadastrados por compatibilidade, mas o fluxo principal usa os modelos separados por base.

## Organização do projeto

A divisão de tasks do projeto está disponível em: [pdi-deteccao-objetos](https://github.com/users/Jenniiu/projects/2/views/1)
