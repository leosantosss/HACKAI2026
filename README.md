# Vizzion: AI-Powered Spatial Awareness for the Visually Impaired

![Vizzion Workflow](assets/workflow.png)

Vizzion is a real-time assistive navigation system that translates the visual world into intuitive haptic feedback. By combining state-of-the-art semantic segmentation with intelligent temporal tracking, Vizzion identifies safe paths, structural hazards, and rapidly approaching threats to empower independent mobility.

## 🚀 Key Features

*   **Pixel-Level Scene Understanding:** Utilizes a fine-tuned **SegFormer-B0** model to distinguish between sidewalks, roads, curbs, and stairs with high precision.
*   **Intelligent Approach Tracking:** Implements an **Exponential Moving Average (EMA)** growth algorithm to detect "object growth," alerting users only when a threat (like a car or person) is actively moving toward them.
*   **Stateful Hazard Detection:** Specifically engineered to detect structural drops (curbs) and inclines (stairs) that traditional ultrasonic sensors often miss.
*   **Prioritized Haptic Feedback:** Translates visual intensity into hardware vibrations. The system prioritizes immediate hazards (stairs/curbs) over general obstacles to ensure the user receives the most critical information first.
*   **Hardware Accelerated:** Optimized for **Raspberry Pi 5** and local GPUs (**Apple Silicon MPS / NVIDIA CUDA**), achieving 30+ FPS for real-time safety.

## 🛠️ Built With

**Python**, **PyTorch**, **SegFormer**, **Hugging Face Transformers**, **OpenCV**, **NumPy**, **Pandas**, **Roboflow**, **Google Colab**, **Raspberry Pi 5**, **Metal Performance Shaders (MPS)**, **CUDA**, and **GitHub**.

---

## 💻 Tech Stack Depth

### The AI Brain
We fine-tuned a **SegFormer-B0** encoder-decoder architecture on a custom-curated street navigation dataset. Unlike standard object detection, this allows Vizzion to "see" the safe walking area (sidewalk) and identify the exact boundaries of structural hazards like curbs.

### The Logic Engine
To prevent "beeper fatigue," we developed a temporal filter that calculates the change in an object's area over time. By tracking the **growth rate**, the system can distinguish between a person standing still and a vehicle approaching at speed.

---

## 🔌 Hardware Setup

*   **Processor:** Raspberry Pi 5 (or Mac/PC for development)
*   **Camera:** Raspberry Pi Camera Module 3 or any USB Webcam
*   **Haptics:** Piezo buzzer or vibration motor connected to **GPIO 18** (PWM)

---

## 🏃 Getting Started

### 1. Install Dependencies
```bash
pip install torch transformers opencv-python pillow numpy roboflow
```

### 2. Configure
Adjust thresholds and performance settings in `src/config.py`.
*   `FRAME_SKIP`: Set to `2` or `3` for smoother performance on low-power devices.
*   `APPROACH_ZONE`: Define the horizontal width of your "safe path."

### 3. Run
```bash
python src/main.py
```

## 📈 Performance Tip
If the system feels laggy on your hardware, enable **Hackathon Speed Mode** by adjusting these values in `src/config.py`:
- `FRAME_SKIP = 2`
- `SHOW_DISPLAY = False` (for maximum headless performance)

---

*Developed during the 2026 AI Hackathon by Leo Santos.*
