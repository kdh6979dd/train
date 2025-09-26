import sys, os, subprocess, tempfile, signal, configparser, threading, json

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if sys.platform == "win32" and os.environ.get("QT_DPI_FIX") != "1":
    os.environ.update(
        {
            "QT_DPI_FIX": "1",
            "QT_QPA_PLATFORM": "windows:dpiawareness=0",
            "QT_ENABLE_HIGHDPI_SCALING": "0",
            "QT_AUTO_SCREEN_SCALE_FACTOR": "0",
            "QT_LOGGING_RULES": "qt.qpa.window=false",
            "QT_FONT_DPI": "96",
        }
    )
    os.execv(sys.executable, [sys.executable, sys.argv[0]] + sys.argv[1:])
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

STATE = "config.ini"
cfg = configparser.ConfigParser()
if os.path.exists(STATE):
    try:
        cfg.read(STATE, encoding="utf-8")
    except:
        pass
if not os.path.exists(STATE) or os.path.getsize(STATE) == 0:
    cfg["TRAIN"] = {
        "mode": "Pretrained",
        "w_pre": "yolo12n.pt",
        "w_resume": "",
        "data_yaml": "",
        "epochs": "50",
        "imgsz": "640",
        "batch": "auto",
        "workers": "16",
        "cache": "False",
        "augment": "True",
        "degrees": "10",
        "flipud": "0.0",
        "device": "0",
        "amp": "True",
        "profile": "False",
        "wandb_off": "True",
    }
    with open(STATE, "w", encoding="utf-8") as f:
        cfg.write(f)


def g(s, k, d):
    try:
        v = cfg.get(s, k)
        return v if v not in ("", None) else str(d)
    except:
        return str(d)


def gi(s, k, d):
    try:
        return cfg.getint(s, k)
    except:
        return int(d)


def gf(s, k, d):
    try:
        return cfg.getfloat(s, k)
    except:
        return float(d)


def gb(s, k, d):
    try:
        return cfg.getboolean(s, k)
    except:
        return bool(d)


def setv(sec, key, val):
    if not cfg.has_section(sec):
        cfg.add_section(sec)
    cfg.set(sec, key, str(val))
    with open(STATE, "w", encoding="utf-8") as f:
        cfg.write(f)


DEFAULTS = dict(
    epochs=int(g("TRAIN", "epochs", 50)),
    imgsz=int(g("TRAIN", "imgsz", 640)),
    batch=g("TRAIN", "batch", "auto"),
    workers=int(g("TRAIN", "workers", 16)),
    cache=gb("TRAIN", "cache", False),
    augment=gb("TRAIN", "augment", True),
    degrees=gi("TRAIN", "degrees", 10),
    flipud=gf("TRAIN", "flipud", 0.0),
    device=g("TRAIN", "device", "0"),
    amp=gb("TRAIN", "amp", True),
    profile=gb("TRAIN", "profile", False),
    wandb_off=gb("TRAIN", "wandb_off", True),
)


def pretrained():
    return [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov10n.pt",
        "yolov10s.pt",
        "yolov10m.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo12n.pt",
        "yolo12s.pt",
        "yolo12m.pt",
    ]


def _parse_yaml(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        d = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
        return d


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _count_images(entry):
    import os, glob

    n = 0
    if entry is None:
        return 0
    if isinstance(entry, (list, tuple)):
        for e in entry:
            n += _count_images(e)
            return n
    p = str(entry).strip().strip('"').strip("'")
    if not p:
        return 0
    p = os.path.normpath(p)
    if os.path.isdir(p):
        for r, _, files in os.walk(p):
            for f in files:
                if os.path.splitext(f)[1].lower() in _IMG_EXT:
                    n += 1
    elif os.path.isfile(p) and p.lower().endswith(".txt"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if os.path.splitext(line.strip())[1].lower() in _IMG_EXT:
                        n += 1
        except:
            pass
    else:
        for ext in _IMG_EXT:
            n += len(glob.glob(p.replace("*", "*") + f"/**/*{ext}", recursive=True))
    return n


def worker_main(arg_path):
    import multiprocessing as mp, json, os, tempfile

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import cv2

        cv2.setNumThreads(0)
        try:
            cv2.ocl.setUseOpenCL(False)
        except:
            pass
    except:
        pass
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if sys.platform == "win32":
        mp.freeze_support()

    with open(arg_path, "r", encoding="utf-8") as f:
        a = json.load(f)
    params = a.get("params") or {}
    for k, v in params.items():
        DEFAULTS[k] = v

    try:
        import yaml
    except:
        yaml = None

    def _strip_parent(p):
        s = str(p).strip().strip('"').strip("'")
        while s.startswith("../"):
            s = s[3:]
        return s

    def _best_path(yaml_dir, rel):
        rel = str(rel).strip().strip('"').strip("'")
        cands = []
        p1 = os.path.normpath(os.path.join(yaml_dir, rel))
        cands.append(p1)
        p2 = os.path.normpath(os.path.join(yaml_dir, _strip_parent(rel)))
        if p2 != p1:
            cands.append(p2)
        best = cands[0]
        best_n = _count_images(best)
        for c in cands[1:]:
            n = _count_images(c)
            if n > best_n:
                best, best_n = c, n
        return best

    yaml_path = os.path.normpath(a["data_yaml"])
    yaml_dir = os.path.dirname(yaml_path)
    d_raw = _parse_yaml(yaml_path) or {}
    if "val" not in d_raw and "valid" in d_raw:
        d_raw["val"] = d_raw.pop("valid")
    t_abs = _best_path(yaml_dir, d_raw.get("train"))
    v_abs = _best_path(yaml_dir, d_raw.get("val"))
    s_abs = _best_path(yaml_dir, d_raw.get("test")) if d_raw.get("test") else None
    d_fix = dict(d_raw)
    d_fix["train"] = t_abs
    d_fix["val"] = v_abs
    if s_abs:
        d_fix["test"] = s_abs
    tmp_yaml = os.path.join(tempfile.gettempdir(), "data_abs.yaml")
    if yaml:
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(d_fix, f, sort_keys=False, allow_unicode=True)
    else:
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            f.write(f"train: {t_abs}\nval: {v_abs}\n")
            if s_abs:
                f.write(f"test: {s_abs}\n")
            if "nc" in d_fix:
                f.write(f"nc: {d_fix['nc']}\n")
            if "names" in d_fix:
                ns = d_fix["names"]
                if isinstance(ns, (list, tuple)):
                    f.write("names:\n")
                    for x in ns:
                        f.write(f"  - {x}\n")
                else:
                    f.write(f"names: {ns}\n")
    data_yaml_abs = tmp_yaml
    d = _parse_yaml(data_yaml_abs)
    train_p, val_p = d.get("train"), d.get("val")
    nt = _count_images(train_p)
    nv = _count_images(val_p)
    print(f"[precheck] train={train_p}", flush=True)
    print(f"[precheck] val={val_p}", flush=True)
    print(f"[precheck] images: train={nt}, val={nv}", flush=True)
    if nt == 0 or nv == 0:
        print("[error] No images found. Fix data.yaml paths.", flush=True)
        sys.exit(3)

    if DEFAULTS["wandb_off"]:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        try:
            os.environ.pop("WANDB_DISABLED")
        except KeyError:
            pass

    print("[load] import ultralytics", flush=True)
    from ultralytics import YOLO

    print("[load] import ok", flush=True)
    import torch, torch.multiprocessing as tmp

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except:
        pass
    try:
        tmp.set_sharing_strategy("file_system")
    except:
        pass

    print(f"[load] weights={a['w']}", flush=True)
    m = YOLO(a["w"])
    print("[load] model ready", flush=True)

    w = int(DEFAULTS["workers"])

    kw = dict(
        device=DEFAULTS["device"]
        if DEFAULTS["device"] == "cpu"
        else int(DEFAULTS["device"]),
        batch=-1 if DEFAULTS["batch"] == "auto" else int(DEFAULTS["batch"]),
        resume=(a["mode"] == "Resume"),
        workers=w,
        deterministic=False,
    )
    if a["mode"] != "Resume":
        kw.update(
            data=data_yaml_abs,
            epochs=int(DEFAULTS["epochs"]),
            imgsz=int(DEFAULTS["imgsz"]),
            profile=bool(DEFAULTS["profile"]),
            cache=bool(DEFAULTS["cache"]),
            amp=bool(DEFAULTS["amp"]),
            augment=bool(DEFAULTS["augment"]),
        )
        if DEFAULTS["augment"]:
            kw.update(
                degrees=int(DEFAULTS["degrees"]), flipud=float(DEFAULTS["flipud"])
            )

    print(f"[train] starting with workers={w}", flush=True)
    m.train(**kw)


if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
    worker_main(sys.argv[2])
    sys.exit(0)

from PySide6.QtCore import QObject, Signal, QSize
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
)

QSS = """
QMainWindow{background:#0b0d0f;}
QGroupBox{border:1px solid #1a1d21;border-radius:12px;margin-top:14px;background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0c0f12,stop:1 #090b0d);}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:2px 8px;color:#d6dbe6;font-weight:600;}
QLabel{color:#cbd2de;}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#111418;color:#eef2f8;border:1px solid #2a2f35;border-radius:10px;padding:8px 12px;min-height:36px;}
QComboBox::drop-down{width:22px;border:0;}
QComboBox QAbstractItemView{background:#0f1216;color:#e9eef7;border:1px solid #2a2f35;selection-background-color:#2f6aff;}
QPlainTextEdit{background:#0a0c0f;color:#e9eef7;border:1px solid #1a1d21;border-radius:12px;padding:10px;}
QPushButton{background:#181b20;color:#e9eef7;border:1px solid #2a2f35;border-radius:10px;padding:10px 16px;}
QPushButton:hover{background:#1b1f25;}
QPushButton#primary{background:#2f6aff;border:0;color:#fff;}
QPushButton#danger{background:#cf3f44;border:0;color:#fff;}
QPushButton#ghost{background:#13161a;color:#d6dbe6;border:1px solid #2a2f35;}
QCheckBox{color:#cbd2de;spacing:8px;}
"""


class Em(QObject):
    line = Signal(str)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Trainer")
        self.setMinimumSize(QSize(1100, 740))
        self.p = None
        self.reader = None
        self.log_path = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        box1 = QGroupBox("Source")
        h1 = QVBoxLayout()
        r0 = QHBoxLayout()
        r0.setSpacing(8)
        self.mode = QComboBox()
        self.mode.addItems(["Pretrained", "Resume"])
        self.mode.setCurrentText(g("TRAIN", "mode", "Pretrained"))
        self.mode.currentTextChanged.connect(lambda v: setv("TRAIN", "mode", v))
        r0.addWidget(QLabel("Mode"))
        r0.addWidget(self.mode)
        r0.addStretch(1)
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        self.w_pre = QComboBox()
        self.w_pre.addItems(pretrained())
        self.w_pre.setCurrentText(g("TRAIN", "w_pre", "yolo12n.pt"))
        self.w_pre.currentTextChanged.connect(lambda v: setv("TRAIN", "w_pre", v))
        r1.addWidget(QLabel("Weights (Pretrained)"))
        r1.addWidget(self.w_pre)
        r1.addStretch(1)
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        self.w_res = QLineEdit(g("TRAIN", "w_resume", ""))
        self.w_res.textChanged.connect(lambda v: setv("TRAIN", "w_resume", v))
        b_res = QPushButton("Browse")
        b_res.setObjectName("ghost")
        b_res.clicked.connect(self.pick_resume)
        r2.addWidget(QLabel("Weights (Resume)"))
        r2.addWidget(self.w_res)
        r2.addWidget(b_res)
        h1.addLayout(r0)
        h1.addLayout(r1)
        h1.addLayout(r2)
        box1.setLayout(h1)

        box2 = QGroupBox("Dataset")
        r_yaml = QHBoxLayout()
        r_yaml.setSpacing(6)
        self.data_yaml = QLineEdit(g("TRAIN", "data_yaml", ""))
        self.data_yaml.textChanged.connect(lambda v: setv("TRAIN", "data_yaml", v))
        b_yaml = QPushButton("Browse")
        b_yaml.setObjectName("ghost")
        b_yaml.clicked.connect(self.pick_yaml)
        r_yaml.addWidget(QLabel("data.yaml"))
        r_yaml.addWidget(self.data_yaml)
        r_yaml.addWidget(b_yaml)
        v2 = QVBoxLayout()
        v2.addLayout(r_yaml)
        box2.setLayout(v2)

        box3 = QGroupBox("Parameters")
        p = QVBoxLayout()
        p.setSpacing(6)
        rA = QHBoxLayout()
        rA.setSpacing(8)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(DEFAULTS["epochs"])
        self.epochs.valueChanged.connect(lambda v: setv("TRAIN", "epochs", v))
        self.imgsz = QComboBox()
        self.imgsz.addItems(["1280", "640", "320", "160"])
        self.imgsz.setCurrentText(str(DEFAULTS["imgsz"]))
        self.imgsz.currentTextChanged.connect(lambda v: setv("TRAIN", "imgsz", v))
        rA.addWidget(QLabel("epochs"))
        rA.addWidget(self.epochs)
        rA.addWidget(QLabel("imgsz"))
        rA.addWidget(self.imgsz)
        rA.addStretch(1)
        rB = QHBoxLayout()
        rB.setSpacing(8)
        self.device = QComboBox()
        self.device.addItems(["cpu", "0", "1", "2", "3", "4", "5"])
        self.device.setCurrentText(DEFAULTS["device"])
        self.device.currentTextChanged.connect(lambda v: setv("TRAIN", "device", v))
        self.batch = QComboBox()
        self.batch.addItems(["auto", "4", "8", "16", "32", "64", "128", "256"])
        self.batch.setCurrentText(DEFAULTS["batch"])
        self.batch.currentTextChanged.connect(lambda v: setv("TRAIN", "batch", v))
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setValue(DEFAULTS["workers"])
        self.workers.valueChanged.connect(lambda v: setv("TRAIN", "workers", v))
        rB.addWidget(QLabel("device"))
        rB.addWidget(self.device)
        rB.addWidget(QLabel("batch"))
        rB.addWidget(self.batch)
        rB.addWidget(QLabel("workers"))
        rB.addWidget(self.workers)
        rB.addStretch(1)
        rC = QHBoxLayout()
        rC.setSpacing(12)
        self.cache = QCheckBox("cache")
        self.cache.setChecked(DEFAULTS["cache"])
        self.cache.toggled.connect(lambda v: setv("TRAIN", "cache", v))
        self.augment = QCheckBox("augment")
        self.augment.setChecked(DEFAULTS["augment"])
        self.augment.toggled.connect(lambda v: setv("TRAIN", "augment", v))
        self.amp = QCheckBox("AMP")
        self.amp.setChecked(DEFAULTS["amp"])
        self.amp.toggled.connect(lambda v: setv("TRAIN", "amp", v))
        self.profile = QCheckBox("profile")
        self.profile.setChecked(DEFAULTS["profile"])
        self.profile.toggled.connect(lambda v: setv("TRAIN", "profile", v))
        self.wandb = QCheckBox("WANDB off")
        self.wandb.setChecked(DEFAULTS["wandb_off"])
        self.wandb.toggled.connect(lambda v: setv("TRAIN", "wandb_off", v))
        rC.addWidget(self.cache)
        rC.addWidget(self.augment)
        rC.addWidget(self.amp)
        rC.addWidget(self.profile)
        rC.addWidget(self.wandb)
        rC.addStretch(1)
        rD = QHBoxLayout()
        rD.setSpacing(8)
        self.degrees = QSpinBox()
        self.degrees.setRange(-180, 180)
        self.degrees.setSingleStep(5)
        self.degrees.setValue(DEFAULTS["degrees"])
        self.degrees.valueChanged.connect(lambda v: setv("TRAIN", "degrees", v))
        self.flipud = QDoubleSpinBox()
        self.flipud.setRange(0.0, 1.0)
        self.flipud.setSingleStep(0.05)
        self.flipud.setValue(DEFAULTS["flipud"])
        self.flipud.valueChanged.connect(lambda v: setv("TRAIN", "flipud", v))
        rD.addWidget(QLabel("degrees"))
        rD.addWidget(self.degrees)
        rD.addWidget(QLabel("flipud"))
        rD.addWidget(self.flipud)
        rD.addStretch(1)
        p.addLayout(rA)
        p.addLayout(rB)
        p.addLayout(rC)
        p.addLayout(rD)
        box3.setLayout(p)

        box5 = QGroupBox("Run")
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_log = QPushButton("Open log")
        self.btn_log.setObjectName("ghost")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_log.clicked.connect(self.open_log)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_log)
        bar.addStretch(1)
        box5.setLayout(bar)

        box6 = QGroupBox("Logs")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setPointSize(10)
        self.log.setFont(mono)
        v = QVBoxLayout()
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(self.log)
        box6.setLayout(v)

        layout.addWidget(box1)
        layout.addWidget(box2)
        layout.addWidget(box3)
        layout.addWidget(box5)
        layout.addWidget(box6, 1)

        self.em = Em()
        self.em.line.connect(self.log.appendPlainText)
        self.mode.currentTextChanged.connect(self.toggle_mode)
        self.toggle_mode()
        self.setStyleSheet(QSS)
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#0b0d0f"))
        pal.setColor(QPalette.Base, QColor("#0a0c0f"))
        pal.setColor(QPalette.Text, QColor("#eef2f8"))
        pal.setColor(QPalette.ButtonText, QColor("#eef2f8"))
        pal.setColor(QPalette.Highlight, QColor("#2f6aff"))
        self.setPalette(pal)

    def toggle_mode(self):
        m = self.mode.currentText()
        self.w_pre.setEnabled(m == "Pretrained")
        self.w_res.setEnabled(m == "Resume")

    def pick_resume(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select last.pt", "", "PyTorch Weights (*.pt)"
        )
        if p:
            self.w_res.setText(p)

    def pick_yaml(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML (*.yaml *.yml)"
        )
        if p:
            self.data_yaml.setText(p)

    def start(self):
        if getattr(self, "p", None):
            return
        mode = self.mode.currentText()
        w = (
            self.w_pre.currentText()
            if mode == "Pretrained"
            else self.w_res.text().strip()
        )
        if not w or (mode != "Pretrained" and not os.path.isfile(w)):
            return
        data_yaml = self.data_yaml.text().strip()
        if not os.path.isfile(data_yaml):
            return
        DEFAULTS.update(
            epochs=self.epochs.value(),
            imgsz=int(self.imgsz.currentText()),
            batch=self.batch.currentText(),
            workers=self.workers.value(),
            cache=self.cache.isChecked(),
            augment=self.augment.isChecked(),
            degrees=self.degrees.value(),
            flipud=self.flipud.value(),
            device=self.device.currentText(),
            amp=self.amp.isChecked(),
            profile=self.profile.isChecked(),
            wandb_off=self.wandb.isChecked(),
        )
        for k, v in DEFAULTS.items():
            setv("TRAIN", k, v)
        a = dict(mode=mode, w=w, data_yaml=data_yaml, params=DEFAULTS)
        args_path = os.path.join(tempfile.gettempdir(), "yolo_worker_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(a, f)
        cmd = [sys.executable, os.path.abspath(sys.argv[0]), "--worker", args_path]
        flags = 0x08000000
        si = None
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", "train.log")
        lf = open(self.log_path, "a", encoding="utf-8")
        self.p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            creationflags=flags,
            startupinfo=si,
        )

        def reader():
            import re

            ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
            buf = []
            r = self.p.stdout
            while True:
                ch = r.read(1)
                if not ch:
                    break
                if ch in ("\n", "\r"):
                    s = "".join(buf).strip()
                    if s:
                        s = ansi.sub("", s)
                        lf.write(s + "\n")
                        lf.flush()
                        self.em.line.emit(s)
                    buf = []
                else:
                    buf.append(ch)
            code = self.p.wait()
            self.em.line.emit(f"[exit {code}]")
            lf.close()

        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        self.em.line.emit("started")

    def stop(self):
        if not getattr(self, "p", None):
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.p.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        except:
            pass
        self.p = None
        self.em.line.emit("stopped")

    def open_log(self):
        if self.log_path and os.path.exists(self.log_path):
            try:
                os.startfile(self.log_path)
            except:
                pass

    def closeEvent(self, e):
        try:
            self.stop()
        except:
            pass
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = Main()
    m.show()
    sys.exit(app.exec())
