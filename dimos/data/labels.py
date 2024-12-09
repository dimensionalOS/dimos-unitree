from dimos.models.labels.llava-34b import Llava
from PIL import Image

class LabelProcessor:
    def __init__(self, debug: bool = False):
        self.model = Llava(mmproj="/app/models/mmproj-model-f16.gguf", model_path="/app/models/llava-v1.6-34b.Q4_K_M.gguf", gpu=True)
        self.prompt = 'Create a JSON representation where each entry consists of a key "object" with a numerical suffix starting from 1, and a corresponding "description" key with a value that is a concise, up to six-word sentence describing each main, distinct object or person in the image. Each pair should uniquely describe one element without repeating keys. An example: {"object1": { "description": "Man in red hat walking." },"object2": { "description": "Wooden pallet with boxes." },"object3": { "description": "Cardboard boxes stacked." },"object4": { "description": "Man in green vest standing." }}'
        self.debug = debug
    def caption_image_data(self, frame: Image.Image):
        try:
            output = self.model.run_inference(frame, self.prompt, return_json=True)
            if self.debug:
                print("output", output)
            return output
        except Exception as e:
            logger.error(f"Error in captioning image: {e}")
            return []