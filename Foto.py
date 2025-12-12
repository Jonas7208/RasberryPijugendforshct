
from picamera2 import Picamera2
from datetime import datetime
import time


picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (640, 480)},
    display="lores"
)

picam2.configure(config)

print("Starte Kamera...")
picam2.start()

print("Kamera passt sich an...")
time.sleep(2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"foto_{timestamp}.jpg"

print("Nehme Foto auf...")
picam2.capture_file(filename)
print(f"✓ Foto wurde gespeichert als: {filename}")

picam2.stop()
print("Kamera gestoppt.")