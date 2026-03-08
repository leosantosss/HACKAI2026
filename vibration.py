import time
import threading
import config

try:
    from gpiozero import PWMOutputDevice
except ImportError:
    # Fallback for non-Pi environments
    class PWMOutputDevice:
        def __init__(self, pin, frequency):
            print(f"Warning: gpiozero not found. Mocking PWMOutputDevice on pin {pin} at {frequency}Hz.")
        def on(self): pass
        def off(self): pass
        def close(self): pass
        @property
        def value(self): return 0
        @value.setter
        def value(self, val): pass

class BuzzerController:
    """Manages prioritizing and playing buzzer patterns."""
    
    def __init__(self):
        self.buzzer = PWMOutputDevice(config.BUZZER_GPIO_PIN, frequency=config.BUZZER_FREQUENCY)
        self.current_priority = 99
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def _play_pattern(self, alert_type, intensity, stop_event):
        """Internal method to play the pattern."""
        try:
            while not stop_event.is_set():
                if alert_type == "approach":
                    # Rapid double pulse, pulse interval scales with intensity (closer = faster)
                    # intensity = growth_rate or area_ratio
                    # Higher intensity = faster beeps
                    delay = max(0.05, 0.3 / (intensity + 0.1))
                    
                    for _ in range(2):
                        if stop_event.is_set(): break
                        self.buzzer.value = 0.5
                        time.sleep(0.05)
                        self.buzzer.value = 0
                        time.sleep(0.05)
                    
                    time.sleep(delay)
                
                elif alert_type == "obstacle":
                    # Slow single pulse, steady 1 second interval
                    self.buzzer.value = 0.5
                    time.sleep(0.2)
                    self.buzzer.value = 0
                    # Wait for the remainder of the 1s interval
                    for _ in range(8):
                        if stop_event.is_set(): break
                        time.sleep(0.1)
                
                elif alert_type == "stair":
                    # 600ms continuous tone
                    self.buzzer.value = 0.5
                    time.sleep(0.6)
                    self.buzzer.value = 0
                    # For stairs, we only play once per trigger as p-spec says "plays pattern"
                    # but doesn't specify repeating. Usually 600ms tone is a one-shot.
                    break
                
                elif alert_type == "curb":
                    # Two short pulses then a 1.5s pause, repeating
                    for _ in range(2):
                        if stop_event.is_set(): break
                        self.buzzer.value = 0.5
                        time.sleep(0.1)
                        self.buzzer.value = 0
                        time.sleep(0.1)
                    
                    for _ in range(15):
                        if stop_event.is_set(): break
                        time.sleep(0.1)
                
                else:
                    break
                    
                # If it's a non-repeating pattern or we only want one loop for now
                if alert_type in ["stair"]:
                    break
        finally:
            self.buzzer.value = 0
            with self.lock:
                if self.thread == threading.current_thread():
                    self.current_priority = 99

    def trigger(self, alert_type, intensity=1.0):
        """Triggers a pattern if it's high enough priority."""
        try:
            new_priority = config.PRIORITY.index(alert_type)
        except ValueError:
            new_priority = 98

        with self.lock:
            # Only interrupt if new is higher priority (lower index) or current is idle
            if new_priority <= self.current_priority or self.current_priority == 99:
                # Stop existing pattern
                self.stop_event.set()
                if self.thread and self.thread.is_alive():
                    # This might take a tiny moment to stop, but we spawn new one quickly
                    pass
                
                self.stop_event = threading.Event()
                self.current_priority = new_priority
                self.thread = threading.Thread(
                    target=self._play_pattern, 
                    args=(alert_type, intensity, self.stop_event),
                    daemon=True
                )
                self.thread.start()

    def cleanup(self):
        self.stop_event.set()
        self.buzzer.close()

if __name__ == "__main__":
    bz = BuzzerController()
    print("Triggering obstacle (slow)...")
    bz.trigger("obstacle")
    time.sleep(2)
    print("Triggering stair (interruption)...")
    bz.trigger("stair")
    time.sleep(2)
    bz.cleanup()
