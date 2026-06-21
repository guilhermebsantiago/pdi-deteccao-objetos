from pathlib import Path
from urllib.request import urlretrieve


def download_model(source_url: str, destination: Path) -> None:
    if not source_url.strip():
        raise ValueError("URL de origem do modelo não configurada.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")

    if "drive.google.com" in source_url:
        import gdown

        output = gdown.download(source_url, str(temporary_path), quiet=False, fuzzy=True)
        if output is None:
            raise RuntimeError("Falha ao baixar arquivo do Google Drive.")
    else:
        urlretrieve(source_url, temporary_path)

    temporary_path.replace(destination)
