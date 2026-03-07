# 🚀 VisionAid Deployment Guide (Raspberry Pi 5)

This guide helps you move the project from your Mac to your Raspberry Pi 5 using an SD card or external SSD.

## Architecture
<img width="681" height="448" alt="Screenshot 2026-03-07 at 5 39 54 PM" src="https://github.com/user-attachments/assets/4b8aa216-da86-499c-bec5-3d2cf8ec61dd" />



## 1. Prepare your OS
Use **Raspberry Pi Imager** to flash **Raspberry Pi OS (64-bit) Bookworm** onto your card/drive.
- **IMPORTANT**: In the "OS Customization" settings, enable **SSH** and set up your **Wi-Fi** so you can connect to it easily.

## 2. Copy the Files
Since Mac cannot natively write to the Linux (ext4) partition of the SD card, the best ways to get the code over are:

### Option A: Via Network (Recommended)
Once the Pi is booted and connected to your Wi-Fi:
```bash
# From your Mac terminal
scp -r ~/Documents/HACK_AI pi@raspberrypi.local:~/VisionAid
```

### Option B: Via GitHub (Best for updates)
1. Create a new repository on [GitHub](https://github.com/new).
2. On your **Mac**, push the code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
3. On your **Pi**, clone it:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git ~/VisionAid
   ```

### Option C: Via the 'boot' partition (The "Sneakernet" way)
1. Plug your flashed SD card into your Mac.
2. Open the volume named `bootfs`.
3. Create a folder called `VisionAid` and copy all these files into it.
4. Plug the card into your Pi and boot it.
5. Once logged in, move the folder:
   ```bash
   mv /boot/firmware/VisionAid ~/VisionAid
   ```

## 3. Setup on the Pi
Run these commands once you are on the Pi terminal:
```bash
cd ~/VisionAid
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install ultralytics opencv-python-headless numpy
```

## 4. Hardware Wiring
- **Camera**: Connect to CSI port (Ribbon cable).
- **Buzzer (+)**: GPIO 18 (PWM).
- **Buzzer (-)**: GND.

## 5. Launch
```bash
python main.py
```
