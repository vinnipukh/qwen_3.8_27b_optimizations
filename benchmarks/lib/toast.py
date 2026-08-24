"""Windows Toast notification sender via PowerShell interop (D2-15).

Dispatches alerts for guard trips, thermal events, and session completion
using raw WinRT XML with strict XML entity escaping (no BurntToast dependency).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import xml.sax.saxutils as saxutils


def build_toast_xml(title: str, body: str) -> str:
    """Build XML-escaped toast notification payload."""
    esc_title = saxutils.escape(title)
    esc_body = saxutils.escape(body)
    return (
        "<toast>\n"
        "  <visual>\n"
        '    <binding template="ToastGeneric">\n'
        f"      <text>{esc_title}</text>\n"
        f"      <text>{esc_body}</text>\n"
        "    </binding>\n"
        "  </visual>\n"
        "</toast>"
    )


def send(title: str, body: str) -> bool:
    """Send Windows toast notification using powershell.exe with -File parameter form."""
    xml_content = build_toast_xml(title, body)

    # PowerShell script to load WinRT Toast Notification and show XML
    ps_script = (
        "param([string]$XmlText)\n"
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null\n"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null\n"
        "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument\n"
        "$xml.LoadXml($XmlText)\n"
        "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml\n"
        "$appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'\n"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)\n"
    )

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as tf:
            tf.write(ps_script)
            tf_path = tf.name

        res = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", tf_path,
                "-XmlText", xml_content,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False
    finally:
        if "tf_path" in locals() and os.path.exists(tf_path):
            try:
                os.unlink(tf_path)
            except OSError:
                pass


def send_summary(ok_count: int, failed_count: int) -> bool:
    """Format and send D2-15 end-of-session summary ping."""
    title = "Benchmark Session Complete"
    body = f"{ok_count} OK / {failed_count} FAILED"
    return send(title, body)
