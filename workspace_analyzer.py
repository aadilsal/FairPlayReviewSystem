import os
from pathlib import Path

root = Path(r"d:\Projects\FairPlayReviewSystem")

def list_py_and_configs():
    exts = ("*.py","requirements.txt","pyproject.toml","data.yaml","*.yaml")
    found = []
    for ext in exts:
        for p in root.rglob(ext):
            found.append(str(p))
    for p in sorted(found):
        print(p)

def dataset_summary():
    ds = root / "cricket_ball_data"
    if not ds.exists():
        print("cricket_ball_data not found at", ds)
        return
    for sub in ("train","valid","test"):
        d = ds / sub
        if not d.exists():
            print(f"{sub}/ not found")
            continue
        imgs = [p for p in d.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
        txts = [p for p in d.rglob("*.txt")]
        print(f"{sub}: images={len(imgs)}, labels={len(txts)}")
        print("  sample images:")
        for s in imgs[:10]:
            print("    ", s)
        print("  sample labels:")
        for s in txts[:10]:
            print("    ", s)
    dy = ds / "data.yaml"
    if dy.exists():
        print("---- data.yaml ----")
        print(dy.read_text())
        print("-------------------")
    else:
        print("data.yaml not found in cricket_ball_data/")

def find_weights():
    exts = (".pt",".pth",".weights")
    found = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            found.append((str(p), size))
    found.sort(key=lambda x: -x[1])
    if not found:
        print("No weight files found (.pt/.pth/.weights)")
        return
    print("Found weight files:")
    for p,s in found:
        print(p, f"{s/1024/1024:.2f} MB")

def find_yolo_usage():
    keywords = ["yolo","ultralytics","YOLO","yolov","yolov8","torch","cv2","openvino"]
    pyfiles = list(root.rglob('*.py'))
    matches = []
    for f in pyfiles:
        try:
            txt = f.read_text(errors='ignore')
        except Exception:
            continue
        for kw in keywords:
            if kw in txt:
                matches.append((str(f), kw))
                break
    if matches:
        print("Files mentioning YOLO/ultralytics/torch/cv2:")
        for p,kw in matches:
            print(" ", p)
    else:
        print("No python files mentioning common YOLO/ultralytics/torch keywords found.")

def show_requirements_and_python():
    req = root / 'requirements.txt'
    if req.exists():
        print('---- requirements.txt ----')
        print(req.read_text())
        print('--------------------------')
    else:
        print('requirements.txt not found at project root')
    # Try to show pip_freeze.txt if exists
    pf = root / 'pip_freeze.txt'
    if pf.exists():
        print('---- pip_freeze.txt ----')
        print(pf.read_text()[:10000])
        print('-------------------------')
    else:
        print('pip_freeze.txt not found')
    # Python version
    try:
        import platform
        print('Python version (platform):', platform.python_version())
    except Exception:
        pass

if __name__ == '__main__':
    print('=== Python/config files ===')
    list_py_and_configs()
    print('\n=== cricket_ball_data summary ===')
    dataset_summary()
    print('\n=== Weight files ===')
    find_weights()
    print('\n=== YOLO / ML usage ===')
    find_yolo_usage()
    print('\n=== Requirements / Python ===')
    show_requirements_and_python()
