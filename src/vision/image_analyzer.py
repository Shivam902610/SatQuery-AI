
import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class SatelliteImageAnalyzer:

    def __init__(self):
        self.model_name = "Salesforce/blip-image-captioning-base"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model on: {self.device}")

        self.processor = BlipProcessor.from_pretrained(
            self.model_name
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

        print("Model loaded successfully!")

    def analyze_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SatQuery AI - Image Analyzer"
    )

    parser.add_argument(
        "image",
        help="Path to the satellite image"
    )

    args = parser.parse_args()

    analyzer = SatelliteImageAnalyzer()

    result = analyzer.analyze_image(args.image)

    print("\n========== SATQUERY AI ==========")
    print("Image Analysis:")
    print(result)
    print("================================")
