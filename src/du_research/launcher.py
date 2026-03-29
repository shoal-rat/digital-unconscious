"""Windows launcher for Digital Unconscious.

Creates a windowless Python process that runs the tray icon.
Avoids VBS (encoding issues with Unicode paths) — uses pythonw.exe directly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_pythonw() -> Path:
    """Find pythonw.exe (Python without console window)."""
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    if pythonw.exists():
        return pythonw
    return Path(sys.executable)


def _pyw_path(target_dir: Path | None = None) -> Path:
    if target_dir is None:
        target_dir = Path.home() / ".du"
    target_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parents[1]
    pyw = target_dir / "du_tray.pyw"
    pyw.write_text(
        f'import sys\n'
        f'sys.path.insert(0, r"{src_dir}")\n'
        f'from du_research.cli import main\n'
        f'main(["tray"])\n',
        encoding="utf-8",
    )
    return pyw


def create_launcher_script(target_dir: Path | None = None) -> Path:
    """Create the .pyw launcher and return its path."""
    return _pyw_path(target_dir)


def launch_windowless() -> bool:
    """Launch the tray app as a windowless process via pythonw.exe."""
    pyw = _pyw_path()
    pythonw = _find_pythonw()
    try:
        subprocess.Popen(
            [str(pythonw), str(pyw)],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def create_desktop_shortcut() -> Path | None:
    """Create a desktop shortcut."""
    try:
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            return None
        pyw = _pyw_path()
        pythonw = _find_pythonw()
        shortcut_path = desktop / "Digital Unconscious.lnk"
        # Use PowerShell to create .lnk
        ps_cmd = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{pythonw}"; '
            f'$s.Arguments = """{pyw}"""; '
            f'$s.Description = "Digital Unconscious"; '
            f'$s.Save()'
        )
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, timeout=10)
        if shortcut_path.exists():
            return shortcut_path
    except Exception:
        pass
    return None
