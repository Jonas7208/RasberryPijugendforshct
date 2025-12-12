#!/usr/bin/env python3
"""
Hauptprogramm: Foto aufnehmen und Müll klassifizieren
Kombiniert Kamera-Funktionalität mit TFLite-Klassifizierung
"""

from picamera2 import Picamera2
from datetime import datetime
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Klassennamen (WICHTIG: Muss mit Trainingsreihenfolge übereinstimmen!)
CLASS_NAMES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]


class GarbageClassifier:
    def __init__(self, model_path='models/model.tflite'):
        """Initialisiert den Klassifikator"""
        print("Lade TFLite-Modell...")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]

        print(f"✓ Modell geladen")
        print(f"  Input Shape: {self.input_shape}")
        print(f"  Erwartet: {self.img_height}x{self.img_width} RGB Bilder")

    def preprocess_image(self, image_path):
        """Bereitet ein Bild für die Klassifizierung vor"""
        img = Image.open(image_path)
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img, dtype=np.float32)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def classify(self, image_path):
        """Klassifiziert ein Bild"""
        img_array = self.preprocess_image(image_path)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()

        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        top_class_idx = np.argmax(predictions)
        top_confidence = predictions[top_class_idx]

        return {
            'class': CLASS_NAMES[top_class_idx],
            'confidence': float(top_confidence),
            'all_predictions': {CLASS_NAMES[i]: float(predictions[i])
                                for i in range(len(CLASS_NAMES))}
        }


def capture_photo():
    """Nimmt ein Foto mit der Raspberry Pi Kamera auf"""
    print("\n" + "=" * 60)
    print("FOTO AUFNEHMEN")
    print("=" * 60)

    # Kamera initialisieren
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)

    # Kamera starten
    print("Starte Kamera...")
    picam2.start()

    print("Kamera passt sich an...")
    time.sleep(2)

    # Foto aufnehmen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"foto_{timestamp}.jpg"

    print("Nehme Foto auf...")
    picam2.capture_file(filename)
    print(f"✓ Foto gespeichert: {filename}")

    # Kamera stoppen
    picam2.stop()
    print("Kamera gestoppt.")

    return filename


def main():
    """Hauptprogramm"""
    print("\n" + "=" * 60)
    print("RASPBERRY PI MÜLL-KLASSIFIZIERER")
    print("=" * 60)

    try:
        # 1. Modell laden
        classifier = GarbageClassifier('models/model.tflite')

        # 2. Foto aufnehmen
        photo_path = capture_photo()

        # 3. Foto klassifizieren
        print("\n" + "=" * 60)
        print("KLASSIFIZIERUNG")
        print("=" * 60)
        print(f"Analysiere: {photo_path}")

        result = classifier.classify(photo_path)

        # 4. Ergebnis anzeigen
        print("\n" + "=" * 60)
        print("✓ ERGEBNIS")
        print("=" * 60)
        print(f"  Kategorie:  {result['class'].upper()}")
        print(f"  Konfidenz:  {result['confidence'] * 100:.2f}%")

        print("\n  Alle Vorhersagen:")
        # Sortiert nach Konfidenz
        sorted_predictions = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, confidence in sorted_predictions:
            bar = "█" * int(confidence * 50)
            print(f"    {class_name:12s}: {confidence * 100:5.2f}% {bar}")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nProgramm abgebrochen.")
    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()