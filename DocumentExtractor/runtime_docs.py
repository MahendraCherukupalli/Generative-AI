import os
from pathlib import Path
from typing import List, Tuple
import logging

try:
    from .config import UPLOADS_DIR
    from .utils import MemoryManager
except Exception:
    from config import UPLOADS_DIR
    from utils import MemoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def handle_uploaded_documents(uploaded_files, memory_manager: MemoryManager) -> Tuple[bool, List[str]]:
    uploads_path = Path(UPLOADS_DIR)
    uploads_path.mkdir(parents=True, exist_ok=True)

    processed_paths: List[Path] = []
    skipped: List[str] = []
    MAX_MB = 30
    allowed = {".pdf", ".docx"}

    for up in uploaded_files:
        if up is None:
            continue
        suffix = Path(up.name).suffix.lower()
        up.seek(0, 2)
        size_mb = up.tell() / (1024 * 1024)
        up.seek(0)
        if suffix not in allowed:
            skipped.append(f"{up.name} (unsupported)")
            continue
        if size_mb > MAX_MB:
            skipped.append(f"{up.name} (>{MAX_MB}MB)")
            continue
        file_path = uploads_path / up.name
        try:
            with open(file_path, "wb") as f:
                f.write(up.read())
            processed_paths.append(file_path)
        except Exception as e:
            logging.error(f"Failed saving upload {up.name}: {e}")

    if not processed_paths:
        return False, skipped

    ok = memory_manager.add_documents(processed_paths)

    # cleanup
    for p in processed_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    return ok, skipped


