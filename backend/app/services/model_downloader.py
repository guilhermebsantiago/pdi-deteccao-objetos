from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve


def download_model(source_url: str, destination: Path, source_file: str = "") -> None:
    if not source_url.strip():
        raise ValueError("URL de origem do modelo não configurada.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")

    if "drive.google.com" in source_url:
        import gdown

        if "/folders/" in source_url:
            expected_name = source_file.strip() or destination.name
            with tempfile.TemporaryDirectory(prefix="pdi-drive-model-") as temp_dir:
                output = gdown.download_folder(source_url, output=temp_dir, quiet=False, use_cookies=False)
                if not output:
                    raise RuntimeError("Falha ao baixar pasta do Google Drive.")

                matches = list(Path(temp_dir).rglob(expected_name))
                if not matches:
                    raise FileNotFoundError(f"Arquivo '{expected_name}' não encontrado na pasta do Google Drive.")

                shutil.copyfile(matches[0], temporary_path)
        else:
            output = gdown.download(source_url, str(temporary_path), quiet=False, fuzzy=True)
            if output is None:
                raise RuntimeError("Falha ao baixar arquivo do Google Drive.")
    else:
        urlretrieve(source_url, temporary_path)

    temporary_path.replace(destination)
