"""Ton eines einzelnen Programms aufnehmen (Windows 10 2004+ / 11, WASAPI Process Loopback).

Nutzt ActivateAudioInterfaceAsync mit VAD\\Process_Loopback und liefert 16 kHz Mono float32.
"""
import ctypes
import threading
from ctypes import POINTER, Structure, byref, c_int, c_int64, c_long, c_uint32, c_uint64, c_ushort, c_ulong, c_void_p
from ctypes import wintypes

import comtypes
import numpy as np
import psutil
from comtypes import COMMETHOD, GUID, HRESULT, COMObject, IUnknown

SR = 16000


class WAVEFORMATEX(Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(Structure):
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("blob", BLOB)]


class ACTIVATION_PARAMS(Structure):
    # AUDIOCLIENT_ACTIVATION_PARAMS mit AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS
    _fields_ = [("ActivationType", c_int), ("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", c_int)]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(c_void_p), "data"), (["out"], POINTER(c_uint32), "frames"),
                  (["out"], POINTER(wintypes.DWORD), "flags"), (["out"], POINTER(c_uint64), "pos"),
                  (["out"], POINTER(c_uint64), "qpc")),
        COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32, "frames")),
        COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32), "frames")),
    ]


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize", (["in"], c_int, "mode"), (["in"], wintypes.DWORD, "flags"),
                  (["in"], c_int64, "duration"), (["in"], c_int64, "period"), (["in"], POINTER(WAVEFORMATEX), "fmt"),
                  (["in"], c_void_p, "session")),
        COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32), "n")),
        COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(c_int64), "l")),
        COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32), "p")),
        COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], c_int, "m"), (["in"], POINTER(WAVEFORMATEX), "f"),
                  (["out"], POINTER(c_void_p), "c")),
        COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(c_void_p), "f")),
        COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(c_int64), "d"), (["out"], POINTER(c_int64), "m")),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle", (["in"], wintypes.HANDLE, "h")),
        COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"), (["out"], POINTER(POINTER(IUnknown)), "ppv")),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(c_long), "hr"),
                  (["out"], POINTER(POINTER(IUnknown)), "iface")),
    ]


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted", (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "op")),
    ]


class IAgileObject(IUnknown):
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class _Handler(COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.done = threading.Event()
        self.hr, self.iface, self.error = None, None, None

    def ActivateCompleted(self, op):
        try:
            self.hr, self.iface = op.GetActivateResult()
        except Exception as e:  # noqa: BLE001 – Fehler an den wartenden Thread weiterreichen
            self.error = e
        self.done.set()
        return 0


_ActivateAsync = ctypes.windll.Mmdevapi.ActivateAudioInterfaceAsync
_ActivateAsync.argtypes = [wintypes.LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT),
                           POINTER(IActivateAudioInterfaceCompletionHandler),
                           POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]
_ActivateAsync.restype = HRESULT

LOOPBACK, EVENTCALLBACK, AUTOCONVERT, SRC_QUALITY = 0x00020000, 0x00040000, 0x80000000, 0x08000000
SILENT = 0x2


def audio_apps():
    """Programme, die gerade eine Audio-Session haben -> sortierte Namen ohne .exe"""
    from pycaw.pycaw import AudioUtilities
    names = set()
    for s in AudioUtilities.GetAllSessions():
        try:
            if s.Process:
                names.add(s.Process.name().removesuffix(".exe"))
        except psutil.Error:
            pass
    return sorted(names, key=str.lower)


def root_pid(app):
    """Oberster Prozess mit diesem Namen (Discord & Browser bestehen aus vielen Prozessen)."""
    exe = app.lower() + ".exe"
    procs = [p for p in psutil.process_iter(["name", "ppid"]) if (p.info["name"] or "").lower() == exe]
    pids = {p.pid for p in procs}
    roots = [p for p in procs if p.info["ppid"] not in pids]
    return (roots or procs)[0].pid if procs else None


class ProcessCapture:
    """Liefert per Callback 16 kHz Mono float32-Blöcke vom Ton des Prozessbaums `pid`."""

    def __init__(self, pid, on_audio, exclude=False):
        self.pid, self.on_audio, self.exclude = pid, on_audio, exclude
        self.running = False
        self.error = None
        self.ready = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ready.wait(5)
        if self.error:
            raise self.error

    def stop(self):
        self.running = False

    def _run(self):
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        try:
            client = self._activate()
            fmt = WAVEFORMATEX(1, 1, SR, SR * 2, 2, 16, 0)  # PCM16 mono 16 kHz, Windows konvertiert
            client.Initialize(0, LOOPBACK | EVENTCALLBACK | AUTOCONVERT | SRC_QUALITY, 2_000_000, 0, byref(fmt), None)
            k = ctypes.windll.kernel32
            ev = k.CreateEventW(None, False, False, None)
            client.SetEventHandle(ev)
            cap = client.GetService(byref(IAudioCaptureClient._iid_)).QueryInterface(IAudioCaptureClient)
            client.Start()
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        try:
            while self.running:
                k.WaitForSingleObject(ev, 100)
                while self.running and cap.GetNextPacketSize():
                    data, frames, flags, _, _ = cap.GetBuffer()
                    if frames:
                        if flags & SILENT or not data:
                            x = np.zeros(frames, np.float32)
                        else:
                            x = np.frombuffer(ctypes.string_at(data, frames * 2), np.int16).astype(np.float32) / 32768
                        self.on_audio(x)
                    cap.ReleaseBuffer(frames)
        finally:
            client.Stop()
            k.CloseHandle(ev)

    def _activate(self):
        params = ACTIVATION_PARAMS(1, self.pid, 1 if self.exclude else 0)
        pv = PROPVARIANT()
        pv.vt = 65  # VT_BLOB
        pv.blob.cbSize = ctypes.sizeof(params)
        pv.blob.pBlobData = ctypes.cast(byref(params), c_void_p)
        handler = _Handler()
        op = POINTER(IActivateAudioInterfaceAsyncOperation)()
        _ActivateAsync("VAD\\Process_Loopback", byref(IAudioClient._iid_), byref(pv),
                       handler.QueryInterface(IActivateAudioInterfaceCompletionHandler), byref(op))
        if not handler.done.wait(5):
            raise TimeoutError("Windows hat die Prozess-Aufnahme nicht gestartet")
        if handler.error:
            raise handler.error
        if handler.hr != 0:
            raise OSError(f"Prozess-Aufnahme fehlgeschlagen (HRESULT 0x{handler.hr & 0xFFFFFFFF:08X})")
        return handler.iface.QueryInterface(IAudioClient)
