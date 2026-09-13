# Zero 2 W bench - Osprey mono vision stack

CM0 is the same BCM2837 silicon as the Pi Zero 2 W (Raspberry Pi's own
guidance: test on Zero 2 W before building the CM0 product). This bench
measures the real rate of the depth net + VIO frontend on your hardware.

## Run it (needs 64-bit Raspberry Pi OS - onnxruntime has no 32-bit wheels)

    sudo apt update && sudo apt install -y python3-pip git
    pip3 install --user --break-system-packages onnxruntime numpy
    # (older OS: drop --break-system-packages; or use a venv)
    git clone --branch auto-researcher https://github.com/Eykam/DroneStudio.git
    cd DroneStudio/autoresearch/bench/zero2w
    python3 bench.py

Takes ~3-5 min on a Zero 2 W. Paste the full printed output back.

## What it reports
- VisNet depth+seg inference: median/p90 ms and fps at 1 and 4 threads (128x96)
- VIO frontend (Shi-Tomasi + LK, the repo's own vis_frontend.py): median ms
- Peak RSS of the process
- Combined net+VIO Hz (the number to compare against the 1-10 Hz slow loop)
