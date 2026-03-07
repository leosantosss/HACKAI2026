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
    """Manages the piezo buzzer patterns."""
    
    def __init__(self):
        self.buzzer = PWMOutputDevice(config.BUZZER_GPIO_PIN, frequency=config.BUZZER_FREQUENCY)
        self._lock = threading.Lock()

    def _play_pattern(self, pattern):
        """Internal method to play the pattern."""
        with self._lock:
            if pattern == "pulse_1":
                # 1 pulse: person far
                self.buzzer.value = 0.5
                time.sleep(0.2)
                self.buzzer.value = 0
            
            elif pattern == "pulse_2":
                # 2 pulses: person medium / bicycle / obstacle
                for _ in range(2):
                    self.buzzer.value = 0.5
                    time.sleep(0.15)
                    self.buzzer.value = 0
                    time.sleep(0.1)
            
            elif pattern == "rapid_3":
                # 3 rapid pulses: person close / vehicle
                for _ in range(3):
                    self.buzzer.value = 0.5
                    time.sleep(0.08)
                    self.buzzer.value = 0
                    time.sleep(0.05)
            
            elif pattern == "continuous":
                # 600ms continuous tone: staircase (or critical)
                self.buzzer.value = 0.5
                time.sleep(0.6)
                self.buzzer.value = 0

    def trigger(self, pattern):
        """Spawns a daemon thread to play the pattern without blocking."""
        thread = threading.Thread(target=self._play_pattern, args=(pattern,), daemon=True)
        thread.start()

    def cleanup(self):
        self.buzzer.close()

if __name__ == "__main__":
    bz = BuzzerController()
    print("Testing 1 pulse...")
    bz.trigger("pulse_1")
    time.sleep(1)
    print("Testing 3 rapid pulses...")
    bz.trigger("rapid_3")
    time.sleep(1)
