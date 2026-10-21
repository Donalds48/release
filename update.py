"""Auto-Update beim Start: vergleicht die eigenen Dateien mit version.json auf dem Server
und lädt geänderte nach. Ohne Internet oder ohne UPDATE_URL passiert einfach nichts.

Exit-Code 2 = Pakete haben sich geändert -> Start.bat führt setup.ps1 erneut aus.
"""
import hashlib
import json
import os
import sys
import urllib.request

UPDATE_URL = ""  # z. B. https://raw.githubusercontent.com/<name>/LiveTranslator/main/

BASE = os.path.dirname(os.path.abspath(__file__))
SETUP_FILES = {"requirements.txt", "requirements-gpu.txt", "setup.ps1"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "LiveTranslator-Updater",
                                               "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local_sha(name):
    try:
        with open(os.path.join(BASE, name), "rb") as f:
            return sha(f.read())
    except OSError:
        return None


def main():
    if not UPDATE_URL:
        return 0
    base = UPDATE_URL.rstrip("/") + "/"
    try:
        remote = json.loads(get(base + "version.json"))
    except Exception as e:  # noqa: BLE001
        print(f"Update-Prüfung übersprungen ({e})")
        return 0

    changed = [n for n, h in remote["files"].items() if local_sha(n) != h]
    if not changed:
        return 0
    print(f"Update auf Version {remote['version']}: {', '.join(changed)}")
    if remote.get("notes"):
        print("  " + remote["notes"])

    # erst alles laden und prüfen, dann schreiben -> kein halbes Update bei Abbruch
    new = {}
    try:
        for n in changed:
            data = get(base + n)
            if sha(data) != remote["files"][n]:
                raise ValueError(f"{n}: Prüfsumme stimmt nicht")
            new[n] = data
    except Exception as e:  # noqa: BLE001
        print(f"Update fehlgeschlagen ({e}) – starte alte Version")
        return 0

    for n, data in new.items():
        # Start.bat läuft gerade -> wird von Start.bat selbst beim nächsten Start ausgetauscht
        path = os.path.join(BASE, n + (".new" if n == "Start.bat" else ""))
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    print("Update installiert.")

    if SETUP_FILES & set(changed):
        try:
            os.remove(os.path.join(BASE, ".venv", ".installed"))
        except OSError:
            pass
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
