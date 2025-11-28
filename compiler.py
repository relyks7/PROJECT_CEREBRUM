import os
import shutil
import subprocess
SRC_DIR = "metal"
OUT_DIR = "kernels"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)
def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)
for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        if not fname.endswith(".metal"):
            continue
        src_path = os.path.join(root, fname)
        name = os.path.splitext(fname)[0]
        air_path = os.path.join(OUT_DIR, f"{name}.air")
        lib_path = os.path.join(OUT_DIR, f"{name}.metallib")
        run([
            "xcrun", "-sdk", "macosx",
            "metal", src_path,
            "-o", air_path
        ])
        run([
            "xcrun", "-sdk", "macosx",
            "metallib", air_path,
            "-o", lib_path
        ])
print("[+] Compiled kernels.")