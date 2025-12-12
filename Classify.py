#!/usr/bin/env python3
"""
Raspberry Pi - Müll-Klassifizierung mit TFLite
Klassifiziert Bilder mit dem trainierten Modell
"""
from Foto import
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite  # Für Raspberry Pi

# Alternative falls TensorFlow installiert: import tensorflow as tf

# Klassennamen definieren (passe diese an deine Klassen an!)
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

        # Interpreter erstellen
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Input/Output Details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input Shape extrahieren
        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]

        print(f"✓ Modell geladen")
        print(f"  Input Shape: {self.input_shape}")
        print(f"  Erwartet: {self.img_height}x{self.img_width} RGB Bilder")

    def preprocess_image(self, image_path):
        """Bereitet ein Bild für die Klassifizierung vor"""
        # Bild laden
        img = Image.open(image_path)

        # Auf richtige Größe bringen
        img = img.resize((self.img_width, self.img_height))

        # In Array konvertieren und normalisieren
        img_array = np.array(img, dtype=np.float32)

        # Falls Graustufenbild, zu RGB konvertieren
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Normalisierung (0-255 -> 0-1)
        img_array = img_array / 255.0

        # Batch-Dimension hinzufügen
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def classify(self, image_path):
        """Klassifiziert ein Bild"""
        # Bild vorbereiten
        img_array = self.preprocess_image(image_path)

        # Inferenz durchführen
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()

        # Ergebnis auslesen
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Top-Klasse finden
        top_class_idx = np.argmax(predictions)
        top_confidence = predictions[top_class_idx]

        return {
            'class': CLASS_NAMES[top_class_idx],
            'confidence': float(top_confidence),
            'all_predictions': {CLASS_NAMES[i]: float(predictions[i])
                                for i in range(len(CLASS_NAMES))}
        }

    def classify_top_n(self, image_path, n=3):
        """Gibt die Top-N Vorhersagen zurück"""
        # Bild vorbereiten
        img_array = self.preprocess_image(image_path)

        # Inferenz
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Top-N Klassen sortieren
        top_indices = np.argsort(predictions)[-n:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[idx])
            })

        return results


def main():
    """Beispiel-Verwendung"""

    # Klassifikator initialisieren
    classifier = GarbageClassifier('models/model.tflite')

    # Einzelnes Bild klassifizieren
    image_path = 'foto_20231212_120000.jpg'  # Dein Foto

    print(f"\nKlassifiziere: {image_path}")
    result = classifier.classify(image_path)

    print(f"\n✓ Ergebnis:")
    print(f"  Klasse:      {result['class']}")
    print(f"  Konfidenz:   {result['confidence'] * 100:.2f}%")

    print(f"\n  Alle Vorhersagen:")
    for class_name, confidence in result['all_predictions'].items():
        print(f"    {class_name:12s}: {confidence * 100:.2f}%")

    # Oder Top-3 Vorhersagen
    print(f"\n✓ Top-3 Vorhersagen:")
    top_results = classifier.classify_top_n(image_path, n=3)
    for i, res in enumerate(top_results, 1):
        print(f"  {i}. {res['class']:12s}: {res['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()