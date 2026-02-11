from PIL import Image
import sys

try:
    img = Image.new('RGB', (100, 100))
    img.save("kl")
    print("Saved successfully")
except Exception as e:
    print(f"Failed to save: {e}")
