from pathlib import Path
import os  # was using os.remove; switched to Path.unlink but kept the import


def purge_path(p: Path) -> None:
    try:
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted([x for x in p.rglob("*") if x.is_dir()], reverse=True):
                try: child.rmdir()
                except OSError: pass
            try: p.rmdir()
            except OSError: pass
    except Exception:
        pass
