"""Live-Übersetzer: PC-Audio (Discord, YouTube, Spiele) -> deutsche Untertitel + Sprachausgabe.

Pipeline: WASAPI-Loopback -> Segmentierung -> Whisper (Erkennung) -> Sprecher-Erkennung
          -> Argos/CTranslate2 (EN->DE) -> Overlay + Piper-TTS
"""
import json
import os
import queue
import re
import site
import string
import sys
import threading
import time
import tkinter as tk

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
SR = 16000

DEFAULTS = {
    "whisper_model": "medium",
    "whisper_device": "cuda",
    "cpu_whisper_model": "small",  # ohne NVIDIA-GPU wäre medium zu langsam
    "cpu_live_model": "base",
    "output_device_contains": "",
    "source_app": "Discord",
    "source_exclude": False,
    "tts_enabled": False,
    "tts_speak_name": False,
    "tts_speed": 1.1,
    "show_original": True,
    "speaker_threshold": 0.5,
    "speaker_new_threshold": 0.4,
    "speaker_merge_threshold": 0.65,
    "speaker_debug": False,
    "live_model": "small",
    "live_interval": 1.0,
    "silence_end_sec": 0.45,
    "max_segment_sec": 8,
    "min_volume": 0.006,
    "overlay_lines": 6,
    "overlay_width": 900,
    "font_size": 15,
    "hotkey_reply": "f7",
    "hotkey_tts": "f8",
    "hotkey_overlay": "f9",
}
try:
    CFG = {**DEFAULTS, **json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))}
except (OSError, ValueError):
    CFG = dict(DEFAULTS)


def save_config():
    with open(os.path.join(BASE, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)


def add_cuda_dlls():
    # cuBLAS/cuDNN kommen aus den nvidia-*-Pip-Paketen, Windows findet sie sonst nicht
    for sp in site.getsitepackages():
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(sp, "nvidia", sub, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


add_cuda_dlls()
if not os.path.exists(os.path.join(BASE, "config.json")):
    save_config()

import ctranslate2  # noqa: E402
import pyaudiowpatch as pyaudio  # noqa: E402
import sentencepiece as spm  # noqa: E402
import sherpa_onnx  # noqa: E402
import sounddevice as sd  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from piper import PiperVoice, SynthesisConfig  # noqa: E402

import proc_loopback  # noqa: E402

# Typische Whisper-Halluzinationen bei Stille/Musik
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "bye",
    "untertitel im auftrag des zdf", "untertitel der amara.org-community",
    "vielen dank", "please subscribe", "subtitles by the amara.org community",
}

ui_queue = queue.Queue()  # (kind, payload) an das Overlay
segment_queue = queue.Queue(maxsize=8)
tts_queue = queue.Queue()
live_event = threading.Event()
models_ready = threading.Event()
state = {"tts": CFG["tts_enabled"], "tts_busy_until": 0.0, "running": True, "final_seg": 0, "source_ver": 0}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def status(msg):
    log(msg)
    ui_queue.put(("status", msg))


def http_get(url):  # argos-net.com lehnt Requests ohne User-Agent ab (403)
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LiveTranslator"})
    return urllib.request.urlopen(req, timeout=60)


# Sprecher-Stimmmodell und deutsche TTS-Stimme; Whisper und Argos laden sich selbst nach
MODEL_FILES = {
    "speaker.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                    "speaker-recongition-models/nemo_en_titanet_small.onnx",
    "de_voice.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                     "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "de_voice.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                          "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
}


def ensure_models():
    os.makedirs(MODELS, exist_ok=True)
    for name, url in MODEL_FILES.items():
        path = os.path.join(MODELS, name)
        if os.path.exists(path):
            continue
        with http_get(url) as resp, open(path + ".part", "wb") as f:
            total, done = int(resp.headers.get("Content-Length") or 0), 0
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    ui_queue.put(("status", f"Lade {name} … {done * 100 // total} %"))
        os.replace(path + ".part", path)
        log(f"{name} geladen")
    models_ready.set()


# ---------------------------------------------------------------- Audio-Aufnahme

def open_source(raw):
    """Startet die Aufnahme laut CFG source_app/source_exclude. Liefert 16 kHz Mono in `raw`.
    -> Funktion zum Stoppen"""
    app, exclude = CFG.get("source_app", ""), CFG.get("source_exclude", False)
    if app:
        pid = proc_loopback.root_pid(app)
        if pid is None:
            status(f"{app} läuft nicht – höre vorerst auf alles")
        else:
            try:
                cap = proc_loopback.ProcessCapture(pid, raw.put, exclude=exclude)
                cap.start()
                status(f"Höre auf: {'alles außer ' if exclude else 'nur '}{app}")
                return cap.stop
            except Exception as e:  # noqa: BLE001
                status(f"Prozess-Aufnahme für {app} fehlgeschlagen ({e}) – höre auf alles")

    p = pyaudio.PyAudio()
    dev = p.get_default_wasapi_loopback()
    rate, ch = int(dev["defaultSampleRate"]), dev["maxInputChannels"]
    log(f"Höre auf: {dev['name']} ({rate} Hz, {ch} ch)")
    if not app:
        status("Höre auf: alles (gesamter PC-Ton)")

    def cb(data, frames, info, flags):
        x = np.frombuffer(data, dtype=np.float32).reshape(-1, ch).mean(axis=1)
        if rate != SR:
            n = int(len(x) * SR / rate)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        raw.put(x)
        return (None, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                    input_device_index=dev["index"], frames_per_buffer=int(rate * 0.03),
                    stream_callback=cb)
    stream.start_stream()

    def stop():
        stream.stop_stream()
        stream.close()
        p.terminate()
    return stop


def capture_thread():
    raw = queue.Queue()
    src_ver = state["source_ver"]
    stop_source = open_source(raw)

    buf, speaking, silence, noise = [], False, 0.0, CFG["min_volume"]
    seg_no, last_live = 0, 0.0
    while state["running"]:
        if state["source_ver"] != src_ver:  # Quelle im Overlay gewechselt
            stop_source()
            src_ver = state["source_ver"]
            raw = queue.Queue()
            stop_source = open_source(raw)
            buf, speaking, silence = [], False, 0.0
        try:
            x = raw.get(timeout=0.1)
            data, dur = x, len(x) / SR
        except queue.Empty:
            # WASAPI liefert nichts, wenn gar nichts abgespielt wird -> zählt als Stille
            data, dur = None, 0.1

        # Eigene Sprachausgabe nicht wieder aufnehmen
        if time.time() < state["tts_busy_until"]:
            buf, speaking, silence = [], False, 0.0
            continue

        rms = float(np.sqrt(np.mean(x ** 2))) if data is not None and len(x) else 0.0
        if not speaking:
            noise = 0.98 * noise + 0.02 * min(rms, noise * 4)
        thr = max(CFG["min_volume"], noise * 3)

        if rms > thr:
            if not speaking:
                speaking = True
                buf = []
                seg_no += 1
                last_live = time.time()
            silence = 0.0
        elif speaking:
            silence += dur

        if speaking:
            if data is not None:
                buf.append(x)
            total = sum(len(b) for b in buf) / SR
            # Live-Vorschau: während jemand noch redet, regelmäßig den bisherigen Satz schicken
            if (CFG["live_model"] and total >= 1.0 and silence < 0.2
                    and time.time() - last_live >= CFG["live_interval"]):
                state["live_job"] = (seg_no, np.concatenate(buf))
                live_event.set()
                last_live = time.time()
            if silence >= CFG["silence_end_sec"] or total >= CFG["max_segment_sec"]:
                if total - silence >= 0.4:
                    try:
                        segment_queue.put_nowait((seg_no, np.concatenate(buf)))
                    except queue.Full:
                        log("Rückstau – Segment verworfen")
                buf, speaking, silence = [], False, 0.0


# ---------------------------------------------------------------- Sprecher

SPEAKERS_FILE = os.path.join(BASE, "speakers.json")


class Speakers:
    """Online-Sprecher-Clustering über Stimm-Embeddings.

    Jeder Abschnitt wird in überlappende Fenster zerlegt, damit Sprecherwechsel ohne Pause
    erkannt werden. Sprecher haben stabile IDs; zusammengeführte IDs werden über `alias` aufgelöst.
    Benannte Sprecher werden in speakers.json gespeichert und beim nächsten Start wiedererkannt.
    """
    WIN, HOP, MIN = 1.5, 0.75, 0.8  # Sekunden
    MAX_EXEMPLARS = 40

    def __init__(self):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=os.path.join(MODELS, "speaker.onnx"), num_threads=2, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.lock = threading.Lock()
        self.reset()
        self.load()

    # -- Verwaltung
    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.spk = {}      # id -> {"name", "named", "anchors": [emb], "ex": [(line_id, emb)]}
            self.alias = {}    # zusammengeführte id -> id
            self.next_id = 0
            self.last = None
            self.letter = 0

    def resolve(self, sid):
        while sid in self.alias:
            sid = self.alias[sid]
        return sid

    def name(self, sid):
        s = self.spk.get(self.resolve(sid))
        return s["name"] if s else "?"

    def ids(self):
        return list(self.spk)

    def rename(self, sid, name):
        with self.lock:
            s = self.spk[self.resolve(sid)]
            s["name"], s["named"] = name, True
        self.save()

    def _new(self, embs=()):
        sid = self.next_id
        self.next_id += 1
        n = self.letter
        self.letter += 1
        name = "Sprecher " + (string.ascii_uppercase[n] if n < 26 else str(n + 1))
        self.spk[sid] = {"name": name, "named": False, "anchors": [], "ex": [(-1, e) for e in embs]}
        return sid

    def load(self):
        try:
            data = json.load(open(SPEAKERS_FILE, encoding="utf-8"))
        except (OSError, ValueError):
            return
        for d in data:
            sid = self._new()
            e = np.array(d["emb"], np.float32)
            self.spk[sid].update(name=d["name"], named=True, anchors=[e / np.linalg.norm(e)])
        self.letter = 0
        log(f"Bekannte Sprecher geladen: {', '.join(d['name'] for d in data)}")

    def save(self):
        with self.lock:
            out = [{"name": s["name"], "emb": self.centroid(sid).tolist()}
                   for sid, s in self.spk.items() if s["named"] and self.centroid(sid) is not None]
        with open(SPEAKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f)

    # -- Embeddings
    def embed(self, audio):
        s = self.ex.create_stream()
        s.accept_waveform(SR, audio)
        s.input_finished()
        e = np.array(self.ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def windows(self, audio):
        """-> Liste (start_sec, end_sec, emb)"""
        dur = len(audio) / SR
        if dur < self.MIN:
            return []
        if dur <= self.WIN + self.HOP / 2:
            return [(0.0, dur, self.embed(audio))]
        out, t = [], 0.0
        while t + self.WIN <= dur + 1e-6:
            a, b = int(t * SR), int((t + self.WIN) * SR)
            out.append((t, t + self.WIN, self.embed(audio[a:b])))
            t += self.HOP
        if dur - (t - self.HOP + self.WIN) > self.HOP / 2:  # Rest am Ende
            out.append((dur - self.WIN, dur, self.embed(audio[-int(self.WIN * SR):])))
        return out

    def centroid(self, sid):
        s = self.spk[sid]
        embs = s["anchors"] * 3 + [e for _, e in s["ex"]]
        if not embs:
            return None
        c = np.mean(embs, axis=0)
        return c / (np.linalg.norm(c) + 1e-9)

    def best(self, e):
        best, sim = None, -1.0
        for sid in self.spk:
            c = self.centroid(sid)
            if c is not None:
                s = float(np.dot(e, c))
                if s > sim:
                    best, sim = sid, s
        return best, sim

    # -- Zuordnung
    def label(self, wins):
        """Ordnet jedem Fenster eine Sprecher-ID zu (legt bei Bedarf neue Sprecher an)."""
        with self.lock:
            if not wins:
                if self.last is None or self.last not in self.spk:
                    self.last = self._new()
                return []
            t_new = CFG["speaker_new_threshold"]
            labels, sims = [], []
            for _, _, e in wins:
                sid, s = self.best(e)
                labels.append(sid if s >= t_new else None)
                sims.append(s)

            # Läufe unbekannter Fenster -> neuer Sprecher (einzelne Ausreißer -> Nachbar)
            k = 0
            while k < len(labels):
                if labels[k] is not None:
                    k += 1
                    continue
                j = k
                while j < len(labels) and labels[j] is None:
                    j += 1
                run = [wins[r][2] for r in range(k, j)]
                if len(run) >= 2 or len(wins) == 1 or not self.spk:
                    m = np.mean(run, axis=0)
                    m /= np.linalg.norm(m) + 1e-9
                    sid, s = self.best(m)
                    if sid is None or s < CFG["speaker_threshold"]:
                        # Sprecher aus zu kurzen Schnipseln haben noch keine Stimme -> zuerst die füllen
                        empty = [i for i, sp in self.spk.items() if not sp["ex"] and not sp["anchors"]]
                        sid = empty[-1] if empty else self._new()
                    for r in range(k, j):
                        labels[r] = sid
                k = j
            for k in range(len(labels)):
                if labels[k] is None:
                    near = [labels[r] for r in (k - 1, k + 1) if 0 <= r < len(labels) and labels[r] is not None]
                    labels[k] = near[0] if near else self.best(wins[k][2])[0]

            if CFG["speaker_debug"]:
                log("  Fenster: " + " ".join(f"{self.spk[l]['name'][-1]}:{s:.2f}" for l, s in zip(labels, sims)))
            self.last = labels[-1]
            return labels

    def commit(self, line_id, sid, embs):
        with self.lock:
            ex = self.spk[sid]["ex"]
            ex.extend((line_id, e) for e in embs)
            del ex[:-self.MAX_EXEMPLARS]
            self._merge()

    def _merge(self):
        ids = list(self.spk)
        for a in ids:
            for b in ids:
                if a >= b or a not in self.spk or b not in self.spk:
                    continue
                sa, sb = self.spk[a], self.spk[b]
                if sa["named"] and sb["named"]:
                    continue
                ca, cb = self.centroid(a), self.centroid(b)
                if ca is None or cb is None or float(np.dot(ca, cb)) < CFG["speaker_merge_threshold"]:
                    continue
                keep, drop = (b, a) if sb["named"] and not sa["named"] else (a, b)
                self.spk[keep]["ex"] = (self.spk[keep]["ex"] + self.spk[drop]["ex"])[-self.MAX_EXEMPLARS:]
                self.spk[keep]["anchors"] += self.spk[drop]["anchors"]
                del self.spk[drop]
                self.alias[drop] = keep
                log(f"Sprecher zusammengeführt: {drop} -> {self.spk[keep]['name']}")

    def correct(self, line_id, old_sid, target):
        """Nutzer-Korrektur: Zeile gehörte zu `target` (None = neuer Sprecher). -> neue ID"""
        with self.lock:
            old = self.spk.get(self.resolve(old_sid))
            moved = []
            if old:
                moved = [e for lid, e in old["ex"] if lid == line_id]
                old["ex"] = [(lid, e) for lid, e in old["ex"] if lid != line_id]
            sid = self._new() if target is None else self.resolve(target)
            # Korrekturen zählen stärker als automatische Zuordnungen
            self.spk[sid]["anchors"] += moved
            return sid


# ---------------------------------------------------------------- Übersetzung EN->DE

ARGOS_INDEX = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"


class Translator:
    """Argos-Modelle direkt über CTranslate2. Fehlende Sprachpaare werden nachgeladen,
    nicht vorhandene Paare laufen über Englisch (z. B. de -> en -> fr)."""

    def __init__(self):
        self.pairs = {}
        self.lock = threading.Lock()
        self.index = None

    def _download(self, a, b):
        import shutil
        import zipfile
        get = http_get

        if self.index is None:
            self.index = json.load(get(ARGOS_INDEX))
        pkg = next((p for p in self.index if p["from_code"] == a and p["to_code"] == b), None)
        if not pkg:
            return False
        status(f"Lade Sprachpaket {a}→{b} …")
        tmp = os.path.join(MODELS, f"_{a}_{b}")
        zpath = tmp + ".zip"
        with get(pkg["links"][0]) as resp, open(zpath, "wb") as f:
            shutil.copyfileobj(resp, f)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        os.remove(zpath)
        inner = os.path.join(tmp, os.listdir(tmp)[0])
        shutil.rmtree(os.path.join(inner, "stanza"), ignore_errors=True)
        shutil.move(inner, os.path.join(MODELS, f"{a}_{b}"))
        shutil.rmtree(tmp, ignore_errors=True)
        status(f"Sprachpaket {a}→{b} bereit")
        return True

    def _pair(self, a, b):
        key = (a, b)
        if key not in self.pairs:
            d = os.path.join(MODELS, f"{a}_{b}")
            if not os.path.isdir(d):
                try:
                    if not self._download(a, b):
                        self.pairs[key] = None
                        return None
                except Exception as e:
                    log(f"Download {a}->{b} fehlgeschlagen: {e}")
                    return None
            sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "sentencepiece.model"))
            tr = ctranslate2.Translator(os.path.join(d, "model"), device="cpu", compute_type="int8")
            self.pairs[key] = (sp, tr)
        return self.pairs[key]

    def _run(self, text, pair):
        sp, tr = pair
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        if not sents:
            return ""
        res = tr.translate_batch([sp.encode(s, out_type=str) for s in sents], beam_size=2)
        # wie Argos selbst: nicht alle Modelle teilen das Vokabular des SentencePiece-Modells
        return " ".join("".join(r.hypotheses[0]).replace("▁", " ").strip() for r in res)

    def translate(self, text, src, dst):
        if src == dst:
            return text
        with self.lock:
            p = self._pair(src, dst)
            if p:
                return self._run(text, p)
            if "en" in (src, dst):
                return None
            p1, p2 = self._pair(src, "en"), self._pair("en", dst)
            if p1 and p2:
                return self._run(self._run(text, p1), p2)
            return None

    def __call__(self, text):
        return self.translate(text, "en", "de")


# ---------------------------------------------------------------- Verarbeitung

def load_whisper(live=False):
    name = CFG["live_model" if live else "whisper_model"]
    if CFG["whisper_device"] == "cuda":
        for ct in ("int8", "float32"):  # ältere Karten können kein int8
            try:
                m = WhisperModel(name, device="cuda", compute_type=ct)
                m.transcribe(np.zeros(SR, np.float32), language="en")  # CUDA-DLLs prüfen
                return m, "GPU"
            except Exception as e:
                err = e
        log(f"GPU nicht nutzbar ({err}) – nehme CPU")
    name = CFG["cpu_live_model" if live else "cpu_whisper_model"] or name
    return WhisperModel(name, device="cpu", compute_type="int8"), f"CPU, {name}"


def clean(text):
    t = text.strip()
    if t.lower().strip(" .!?,") in HALLUCINATIONS or len(t.strip(" .!?,-")) < 2:
        return ""
    return t


def process_thread():
    try:
        ensure_models()
    except Exception as e:  # noqa: BLE001
        status(f"Modell-Download fehlgeschlagen ({e}) – Internet prüfen und neu starten")
        return
    status("Lade Whisper (beim ersten Start wird es heruntergeladen) …")
    whisper, where = load_whisper()
    status("Lade Übersetzer und Sprecher-Erkennung …")
    translate = Translator()
    translate("Hello")  # en->de laden
    state["translator"] = translate
    speakers = Speakers()
    ui_queue.put(("speakers", speakers))
    status(f"Bereit ({where}) – warte auf Sprache …")

    while state["running"]:
        item = segment_queue.get()
        if item is None:
            break
        seg_no, audio = item
        process_segment(whisper, translate, speakers, audio)
        # endgültige Zeile ist da -> Live-Vorschau dieses Abschnitts verwerfen
        state["final_seg"] = seg_no
        ui_queue.put(("live_done", seg_no))


def process_segment(whisper, translate, speakers, audio):
    t0 = time.time()
    lang, prob, _ = whisper.detect_language(audio)
    task = "transcribe" if lang in ("en", "de") else "translate"
    if prob > 0.7:
        state["cur_lang"] = lang
        if lang != "de":
            state["last_lang"] = lang
            ui_queue.put(("lang", lang))

    # Wort-Zeitstempel kosten Zeit und werden nur gebraucht, wenn mehrere Sprecher im Abschnitt sind
    wins = speakers.windows(audio)
    labels = speakers.label(wins)
    multi = len(set(labels)) > 1
    segs, info = whisper.transcribe(audio, language=lang, task=task, beam_size=2, word_timestamps=multi,
                                    vad_filter=True, condition_on_previous_text=False)
    segs = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
    if multi:
        words = [w for s in segs for w in (s.words or [])]
        if not clean("".join(w.word for w in words)):
            return
        groups = split_by_speaker(words, wins, labels, speakers.last)
    else:
        text = " ".join(s.text.strip() for s in segs)
        if not clean(text):
            return
        sid = labels[0] if labels else speakers.last
        groups = [(sid, text, [e for _, _, e in wins])]

    for sid, text, embs in groups:
        text = clean(text)
        if not text:
            continue
        if lang == "de":
            german, original = text, None
        else:
            german = translate(text)
            original = text if task == "transcribe" else f"[{lang}] {text}"
        line_id = next(LINE_IDS)
        speakers.commit(line_id, sid, embs)
        log(f"{speakers.name(sid)} [{lang} {prob:.0%}] {text}  ->  {german}  ({time.time() - t0:.1f}s)")
        ui_queue.put(("line", {"id": line_id, "spk": sid, "de": german, "orig": original}))
        if state["tts"] and lang != "de":
            tts_queue.put((sid, german))


def live_thread():
    """Schnelle Vorschau mit kleinem Modell, solange jemand noch spricht."""
    model, _ = load_whisper(live=True)
    while state.get("translator") is None and state["running"]:
        time.sleep(0.2)
    tr = state.get("translator")
    while state["running"]:
        live_event.wait()
        live_event.clear()
        job = state.pop("live_job", None)
        if job is None:
            continue
        seg_no, audio = job
        if seg_no <= state["final_seg"]:
            continue
        lang = state.get("cur_lang")
        task = "transcribe" if lang in (None, "en", "de") else "translate"
        segs, info = model.transcribe(audio, language=lang, task=task, beam_size=1, vad_filter=False,
                                      condition_on_previous_text=False, without_timestamps=True)
        text = clean(" ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6))
        if not text or seg_no <= state["final_seg"]:
            continue
        lang = lang or info.language
        german = text if lang == "de" else tr(text)
        ui_queue.put(("live", {"seg": seg_no, "de": german}))


LINE_IDS = __import__("itertools").count()
reply_queue = queue.Queue()


def reply_thread():
    """Deine deutsche Antwort -> Gesprächssprache, zum Ablesen ins Overlay + Zwischenablage."""
    while state["running"]:
        item = reply_queue.get()
        if item is None:
            break
        text, dst = item
        tr = state.get("translator")
        if tr is None:
            status("Übersetzer lädt noch …")
            continue
        out = tr.translate(text, "de", dst)
        if out is None:
            status(f"Keine Übersetzung de→{dst} verfügbar")
            continue
        log(f"Du [de→{dst}] {text}  ->  {out}")
        ui_queue.put(("line", {"id": next(LINE_IDS), "spk": None, "me": dst, "de": out, "orig": text}))


def split_by_speaker(words, wins, labels, fallback):
    """Wörter anhand der Fenster-Labels in Sprecher-Gruppen teilen -> [(sid, text, embs)]"""
    if not wins:
        return [(fallback, "".join(w.word for w in words), [])]

    def at(t):
        return min(range(len(wins)), key=lambda k: abs((wins[k][0] + wins[k][1]) / 2 - t))

    groups = []  # [sid, [words]]
    for w in words:
        sid = labels[at((w.start + w.end) / 2)]
        if groups and groups[-1][0] == sid:
            groups[-1][1].append(w)
        else:
            groups.append([sid, [w]])

    # Mini-Gruppen (< 3 Wörter und < 0.8 s) sind fast immer Fehlzuordnungen -> Nachbar
    merged = []
    for g in groups:
        short = len(g[1]) < 3 and g[1][-1].end - g[1][0].start < 0.8
        if merged and (short or merged[-1][0] == g[0]):
            merged[-1][1].extend(g[1])
        else:
            merged.append(g)
    if len(merged) > 1 and len(merged[0][1]) < 3 and merged[0][1][-1].end - merged[0][1][0].start < 0.8:
        merged[1][1][:0] = merged.pop(0)[1]

    out = []
    for sid, ws in merged:
        a, b = ws[0].start, ws[-1].end
        embs = [e for (s, t, e), l in zip(wins, labels) if l == sid and t > a and s < b]
        out.append((sid, "".join(w.word for w in ws), embs))
    return out


def tts_thread(voice_ready):
    models_ready.wait()
    voice = PiperVoice.load(os.path.join(MODELS, "de_voice.onnx"))
    out = None
    if CFG["output_device_contains"]:
        for i, d in enumerate(sd.query_devices()):
            if CFG["output_device_contains"].lower() in d["name"].lower() and d["max_output_channels"] > 0:
                out = i
                break
    syn = SynthesisConfig(length_scale=1.0 / CFG["tts_speed"])
    voice_ready.set()
    while state["running"]:
        item = tts_queue.get()
        if item is None:
            break
        while tts_queue.qsize() > 2:  # lieber Sätze auslassen als hinterherhinken
            item = tts_queue.get()
        spk, text = item
        if not state["tts"]:
            continue
        if CFG["tts_speak_name"]:
            text = f"{ui_speaker_name(spk)}: {text}"
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text, syn)])
        rate = voice.config.sample_rate
        # nur sperren, wenn die eigene Stimme mit aufgenommen würde
        only_app = CFG.get("source_app") and not CFG.get("source_exclude")
        if out is None and not only_app:
            state["tts_busy_until"] = time.time() + len(audio) / rate + 0.4
        sd.play(audio, rate, device=out)
        sd.wait()


_speakers_ref = {}


def ui_speaker_name(sid):
    sp = _speakers_ref.get("s")
    return sp.name(sid) if sp else "?"


# ---------------------------------------------------------------- Overlay

def hotkey_thread(binds):
    """Globale Hotkeys über Win32 RegisterHotKey (die keyboard-Lib bekommt hier keine Events).
    WM_HOTKEY erlaubt dem Prozess außerdem, sich in den Vordergrund zu holen."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    actions = {}
    for i, (key, action) in enumerate(binds.items(), 1):
        k = key.strip().lower()
        if re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", k):
            vk = 0x6F + int(k[1:])
        elif len(k) == 1 and k.isalnum():
            vk = ord(k.upper())
        else:
            log(f"Hotkey '{key}' nicht unterstützt (nur F1–F24, A–Z, 0–9)")
            continue
        if u.RegisterHotKey(None, i, 0x4000, vk):  # MOD_NOREPEAT
            actions[i] = action
        else:
            log(f"Hotkey {key.upper()} ist schon von einem anderen Programm belegt")
    msg = wintypes.MSG()
    while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == 0x0312 and msg.wParam in actions:
            ui_queue.put((actions[msg.wParam], None))


PLACEHOLDER ="Deine Antwort auf Deutsch … (Enter)"
REPLY_LANGS = ["en", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk", "tr", "ar", "ja", "ko", "zh"]
COLORS = ["#4FC3F7", "#FFB74D", "#81C784", "#F06292", "#BA68C8", "#FFF176", "#4DB6AC", "#E57373"]


class Overlay:
    def __init__(self):
        self.root = r = tk.Tk()
        r.title("Live-Übersetzer")
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.attributes("-alpha", 0.88)
        r.configure(bg="#111")
        w = CFG["overlay_width"]
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        h = CFG["overlay_lines"] * 52 + 70
        r.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - 110}")

        fs = CFG["font_size"]
        self.status = tk.Label(r, text="Starte …", fg="#888", bg="#111", anchor="w",
                               font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=8, pady=(4, 0))

        # Eingabe: deine Antwort auf Deutsch -> Gesprächssprache
        bar = tk.Frame(r, bg="#111")
        bar.pack(side="bottom", fill="x", padx=8, pady=(0, 6))
        self.lang_var = tk.StringVar(value="Auto (en)")
        langs = ["Auto"] + REPLY_LANGS
        opt = tk.OptionMenu(bar, self.lang_var, *langs, command=self._pick_lang)
        opt.configure(bg="#222", fg="#ccc", activebackground="#333", bd=0, highlightthickness=0,
                      font=("Segoe UI", 9), width=9)
        opt.pack(side="right", padx=(6, 0))
        self.entry = tk.Entry(bar, bg="#2b2b2b", fg="#fff", insertbackground="#fff", insertwidth=2, bd=0,
                              font=("Segoe UI", fs - 2), relief="flat", selectbackground="#3399ff",
                              selectforeground="#fff")
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.insert(0, PLACEHOLDER)
        self.entry.configure(fg="#666")
        self.entry.bind("<Return>", self._send)
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<Button-1>", lambda e: self._focus_entry())
        self.lang_auto = True
        self.reply_lang = "en"

        self.text = tk.Text(r, bg="#111", fg="#fff", bd=0, wrap="word", highlightthickness=0,
                            font=("Segoe UI", fs), padx=8, pady=4, cursor="arrow")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("orig", foreground="#888", font=("Segoe UI", fs - 4, "italic"))
        self.text.tag_configure("live", foreground="#bbb", font=("Segoe UI", fs, "italic"))
        self.text.tag_configure("me",foreground="#fff", font=("Segoe UI", fs, "bold"))
        self.text.tag_configure("me_text", foreground="#000", background="#FFE082",
                                font=("Segoe UI", fs + 1, "bold"))
        for i, c in enumerate(COLORS):
            self.text.tag_configure(f"s{i}", foreground=c, font=("Segoe UI", fs, "bold"))
        self.text.configure(state="disabled")

        self.speakers = None
        self.show_orig = tk.BooleanVar(value=CFG["show_original"])
        self.tts_var = tk.BooleanVar(value=state["tts"])
        self.lines = []
        self.live, self.final_seg = None, 0

        for wdg in (r, self.text, self.status):
            wdg.bind("<ButtonPress-1>", self._start_move)
            wdg.bind("<B1-Motion>", self._move)
            wdg.bind("<Button-3>", self._menu)
        self.text.bind("<Double-Button-1>", self._dbl_rename)
        grip = tk.Label(r, text="◢", fg="#555", bg="#111", cursor="size_nw_se")
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.bind("<B1-Motion>", self._resize)

        self._hotkeys()
        r.after(100, self._poll)

    def _clear_placeholder(self, e=None):
        if self.entry.get() == PLACEHOLDER:
            self.entry.delete(0, "end")
            self.entry.configure(fg="#fff")

    def _pick_lang(self, val):
        self.lang_auto = val == "Auto"
        if self.lang_auto:
            self.lang_var.set(f"Auto ({self.reply_lang})")
        else:
            self.reply_lang = val

    def _send(self, e=None):
        text = self.entry.get().strip()
        if not text or text == PLACEHOLDER:
            return
        self.entry.delete(0, "end")
        reply_queue.put((text, self.reply_lang))
        self.status.configure(text=f"Übersetze nach {self.reply_lang} …")

    def _copy(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _activate(self, win=None):
        """Rahmenlose Tk-Fenster werden von Windows beim Klick nicht aktiviert -> Tastatur
        ginge ans Spiel. Alt-Tipp hebt die Vordergrund-Sperre auf, dann SetForegroundWindow."""
        import ctypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = int((win or self.root).wm_frame(), 16)
        fg = u.GetForegroundWindow()
        if fg == hwnd:
            return
        # an den Eingabe-Thread des Vordergrundfensters hängen, dann darf man den Fokus übernehmen
        fg_thread = u.GetWindowThreadProcessId(fg, None)
        me = k.GetCurrentThreadId()
        attached = fg_thread and fg_thread != me and u.AttachThreadInput(me, fg_thread, True)
        u.keybd_event(0x12, 0, 0, 0)
        u.keybd_event(0x12, 0, 2, 0)
        u.BringWindowToTop(hwnd)
        u.SetForegroundWindow(hwnd)
        u.SetFocus(hwnd)
        if attached:
            u.AttachThreadInput(me, fg_thread, False)

    def _focus_entry(self):
        self.root.deiconify()
        self._activate()
        self.root.focus_force()
        self.entry.focus_force()
        self._clear_placeholder()

    def _hotkeys(self):
        binds = {CFG["hotkey_reply"]: "focus_entry", CFG["hotkey_tts"]: "toggle_tts",
                 CFG["hotkey_overlay"]: "toggle_overlay"}
        threading.Thread(target=hotkey_thread, args=(binds,), daemon=True).start()

    def _start_move(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _resize(self, e):
        w = max(300, e.x_root - self.root.winfo_x())
        h = max(80, e.y_root - self.root.winfo_y())
        self.root.geometry(f"{w}x{h}")

    def _line_at(self, e):
        if e.widget is not self.text:
            return None
        for tag in self.text.tag_names(f"@{e.x},{e.y}"):
            if tag.startswith("L"):
                lid = int(tag[1:])
                return next((l for l in self.lines if l["id"] == lid), None)
        return None

    def _dbl_rename(self, e):
        line = self._line_at(e)
        if line and not line.get("me") and self.speakers:
            self._rename(self.speakers.resolve(line["spk"]))
        return "break"

    def _menu(self, e):
        m = tk.Menu(self.root, tearoff=0)
        line = self._line_at(e)
        if line and line.get("me"):
            m.add_command(label="Kopieren", command=lambda: self._copy(line["de"]))
            m.add_separator()
        elif line and self.speakers:
            cur = self.speakers.resolve(line["spk"])
            sub = tk.Menu(m, tearoff=0)
            for sid in self.speakers.ids():
                if sid != cur:
                    sub.add_command(label=self.speakers.name(sid),
                                    command=lambda sid=sid: self._correct(line, sid))
            sub.add_separator()
            sub.add_command(label="Neuer Sprecher …", command=lambda: self._correct(line, None))
            m.add_cascade(label=f"Diese Zeile war nicht {self.speakers.name(cur)} → ", menu=sub)
            m.add_command(label=f"{self.speakers.name(cur)} umbenennen …", command=lambda: self._rename(cur))
            m.add_separator()
        m.add_checkbutton(label=f"Sprachausgabe ({CFG['hotkey_tts'].upper()})", variable=self.tts_var,
                          command=lambda: self._set_tts(self.tts_var.get()))
        m.add_checkbutton(label="Original anzeigen", variable=self.show_orig, command=self._redraw)
        m.add_cascade(label="Audioquelle", menu=self._source_menu(m))
        if self.speakers and self.speakers.ids():
            sub = tk.Menu(m, tearoff=0)
            for sid in self.speakers.ids():
                sub.add_command(label=self.speakers.name(sid), command=lambda sid=sid: self._rename(sid))
            m.add_cascade(label="Sprecher umbenennen", menu=sub)
            m.add_command(label="Sprecher zurücksetzen", command=self._reset_speakers)
        m.add_command(label="Text leeren", command=lambda: (self.lines.clear(), self._redraw()))
        m.add_separator()
        m.add_command(label="Beenden", command=self.quit)
        m.tk_popup(e.x_root, e.y_root)

    def _source_menu(self, parent):
        sub = tk.Menu(parent, tearoff=0)
        cur = (CFG.get("source_app", ""), CFG.get("source_exclude", False))
        try:
            apps = proc_loopback.audio_apps()
        except Exception as e:  # noqa: BLE001
            log(f"Audio-Programme nicht lesbar: {e}")
            apps = []
        apps = [a for a in apps if a.lower() != "python"]
        if cur[0] and cur[0] not in apps:
            apps.append(cur[0])
        self._src_var = tk.StringVar(value=f"{cur[0]}|{int(cur[1])}")
        sub.add_radiobutton(label="Alles (gesamter PC-Ton)", variable=self._src_var, value="|0",
                            command=lambda: self._set_source("", False))
        sub.add_separator()
        for a in apps:
            sub.add_radiobutton(label=f"Nur {a}", variable=self._src_var, value=f"{a}|0",
                                command=lambda a=a: self._set_source(a, False))
        sub.add_separator()
        for a in apps:
            sub.add_radiobutton(label=f"Alles außer {a}", variable=self._src_var, value=f"{a}|1",
                                command=lambda a=a: self._set_source(a, True))
        return sub

    def _set_source(self, app, exclude):
        CFG["source_app"], CFG["source_exclude"] = app, exclude
        save_config()
        state["source_ver"] += 1

    def _set_tts(self, on):
        state["tts"] = on
        self.tts_var.set(on)
        self.status.configure(text=f"Sprachausgabe {'an' if on else 'aus'}")

    def _rename(self, sid):
        # simpledialog landet hinter dem Topmost-Overlay -> eigenes Fenster, selbst topmost
        d = tk.Toplevel(self.root)
        d.title("Sprecher umbenennen")
        d.configure(bg="#1b1b1b")
        d.attributes("-topmost", True)
        d.resizable(False, False)
        tk.Label(d, text=f"Neuer Name für {self.speakers.name(sid)}:", fg="#ddd", bg="#1b1b1b",
                 font=("Segoe UI", 11)).pack(padx=14, pady=(12, 6), anchor="w")
        e = tk.Entry(d, font=("Segoe UI", 13), width=24, bg="#ffffff", fg="#000000",
                     insertbackground="#000000", insertwidth=2, relief="flat",
                     selectbackground="#3399ff", selectforeground="#ffffff")
        e.pack(padx=14, ipady=4)
        e.insert(0, self.speakers.name(sid))
        e.select_range(0, "end")
        e.icursor("end")

        def ok(_=None):
            n = e.get().strip()
            if n:
                self.speakers.rename(sid, n)
                self._redraw()
            d.destroy()

        row = tk.Frame(d, bg="#1b1b1b")
        row.pack(fill="x", padx=14, pady=12)
        tk.Button(row, text="Abbrechen", command=d.destroy, width=10).pack(side="right")
        tk.Button(row, text="OK", command=ok, width=10).pack(side="right", padx=6)
        e.bind("<Return>", ok)
        d.bind("<Escape>", lambda _: d.destroy())
        d.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - d.winfo_width()) // 2
        y = max(0, self.root.winfo_y() - d.winfo_height() - 10)
        d.geometry(f"+{x}+{y}")
        d.lift()
        d.after(50, lambda: (self._activate(d), d.focus_force(), e.focus_force()))

    def _correct(self, line, target):
        line["spk"] = self.speakers.correct(line["id"], line["spk"], target)
        if target is None:
            self._rename(line["spk"])
        self._redraw()

    def _reset_speakers(self):
        self.speakers.reset()
        try:
            os.remove(SPEAKERS_FILE)
        except OSError:
            pass
        self.lines.clear()
        self._redraw()

    def _poll(self):
        try:
            while True:
                kind, val = ui_queue.get_nowait()
                if kind == "status":
                    self.status.configure(text=val)
                elif kind == "speakers":
                    self.speakers = val
                    _speakers_ref["s"] = val
                elif kind == "line":
                    self.lines = (self.lines + [val])[-CFG["overlay_lines"] * 3:]
                    self._redraw()
                    if val.get("me"):
                        self._copy(val["de"])
                        self.status.configure(text="Übersetzt – auch in der Zwischenablage (Strg+V)")
                elif kind == "live":
                    if val["seg"] > self.final_seg:
                        self.live = val
                        self._redraw()
                elif kind == "live_done":
                    self.final_seg = max(self.final_seg, val)
                    if self.live and self.live["seg"] <= val:
                        self.live = None
                        self._redraw()
                elif kind == "lang":
                    if self.lang_auto and val in REPLY_LANGS:
                        self.reply_lang = val
                        self.lang_var.set(f"Auto ({val})")
                elif kind == "focus_entry":
                    self._focus_entry()
                elif kind == "toggle_tts":
                    self._set_tts(not state["tts"])
                elif kind == "toggle_overlay":
                    if self.root.state() == "withdrawn":
                        self.root.deiconify()
                    else:
                        self.root.withdraw()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _redraw(self):
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for l in self.lines:
            if l.get("me"):
                t.insert("end", f"Du → {l['me'].upper()}: ", "me")
                t.insert("end", " " + l["de"] + " ", "me_text")
                t.insert("end", "\n    " + l["orig"] + "\n", "orig")
                continue
            sid =self.speakers.resolve(l["spk"]) if self.speakers else 0
            tag = f"L{l['id']}"
            t.insert("end", ui_speaker_name(sid) + ": ", (f"s{sid % len(COLORS)}", tag))
            t.insert("end", l["de"] + "\n", tag)
            if l["orig"] and self.show_orig.get():
                t.insert("end", "    " + l["orig"] + "\n", ("orig", tag))
        if self.live:
            t.insert("end", "▶ " + self.live["de"] + "\n", "live")
        t.see("end")
        t.configure(state="disabled")

    def quit(self):
        if self.speakers:
            self.speakers.save()
        state["running"] = False
        segment_queue.put(None)
        tts_queue.put(None)
        self.root.destroy()


def main():
    ov = Overlay()
    voice_ready = threading.Event()
    threading.Thread(target=tts_thread, args=(voice_ready,), daemon=True).start()
    threading.Thread(target=process_thread, daemon=True).start()
    threading.Thread(target=reply_thread, daemon=True).start()
    if CFG["live_model"]:
        threading.Thread(target=live_thread, daemon=True).start()
    threading.Thread(target=capture_thread, daemon=True).start()
    ov.root.mainloop()
    os._exit(0)


if __name__ == "__main__":
    main()
