"""Windows launcher for Digital Unconscious.

Creates a windowless Python process that runs the tray icon.
This avoids the console window that blocks icon visibility on Windows.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def create_launcher_script(target_dir: Path | None = None) -> Path:
    """Create launcher scripts that start du without a console window."""
    if target_dir is None:
        target_dir = Path.home() / ".du"
    target_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    # Use pythonw.exe if available (no console window)
    pythonw = Path(python_exe).parent / "pythonw.exe"
    if not pythonw.exists():
        pythonw = Path(python_exe)

    # Find the src directory for dev installs
    src_dir = Path(__file__).resolve().parents[1]

    # Create a .pyw file that launches the tray
    pyw_path = target_dir / "du_tray.pyw"
    pyw_path.write_text(
        f'import sys\n'
        f'sys.path.insert(0, r"{src_dir}")\n'
        f'from du_research.cli import main\n'
        f'main(["tray"])\n',
        encoding="utf-8",
    )

    # Create a .vbs launcher (truly invisible on Windows — no flash)
    vbs_path = target_dir / "Digital Unconscious.vbs"
    # VBS needs doubled quotes for paths with spaces
    cmd_line = f'"{pythonw}" "{pyw_path}"'
    vbs_content = f'Set ws = CreateObject("Wscript.Shell")\nws.Run "{cmd_line}", 0, False\n'
    # Escape inner quotes for VBS
    vbs_content = (
        'Set ws = CreateObject("Wscript.Shell")\n'
        f'ws.Run chr(34) & "{pythonw}" & chr(34) & " " & chr(34) & "{pyw_path}" & chr(34), 0, False\n'
    )
    vbs_path.write_text(vbs_content, encoding="utf-8")

    # Create a .bat for users who prefer command line
    bat_path = target_dir / "du.bat"
    bat_path.write_text(
        f'@echo off\nstart /b "" "{pythonw}" "{pyw_path}"\n',
        encoding="utf-8",
    )

    return vbs_path


def launch_windowless() -> bool:
    """Launch the tray app as a windowless process. Returns True on success."""
    vbs_path = create_launcher_script()
    try:
        subprocess.Popen(
            ["wscript.exe", str(vbs_path)],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return True
    except Exception:
        return False


def create_desktop_shortcut() -> Path | None:
    """Create a desktop shortcut to the launcher."""
    try:
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            return None
        vbs_path = create_launcher_script()
        shortcut_path = desktop / "Digital Unconscious.lnk"
        ps_cmd = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "wscript.exe"; '
            f'$s.Arguments = """{vbs_path}"""; '
            f'$s.Description = "Digital Unconscious — AI Research Companion"; '
            f'$s.Save()'
        )
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            timeout=10,
        )
        if shortcut_path.exists():
            return shortcut_path
    except Exception:
        pass
    return None
