import os
import tempfile
from pathlib import Path

import cog
import numpy as np
from ISR.models import RDN, RRDN
from PIL import Image


class ISRPredictor(cog.Predictor):
    def setup(self):
        """Load the super-resolution ans noise canceling models"""
        self.model_gans = RRDN(weights="gans")
        self.model_noise_cancel = RDN(weights="noise-cancel")

    @cog.input("input", type=Path, help="Image path")
    @cog.input(
        "type",
        type=str,
        default="super-resolution",
        options=["super-resolution", "noise-cancel"],
        help="Precessing type: super-resolution or noise-cancel",
    )
    def predict(self, input, type):
        """Apply super-resolution or noise-canceling to input image"""
        # compute super resolution
        img = Image.open(str(input))
        lr_img = np.array(img)

        if type == "super-resolution":
            img = self.model_gans.predict(np.array(img))
        elif type == "noise-cancel":
            img = self.model_noise_cancel.predict(np.array(img))
        else:
            raise NotImplementedError("Invalid processing type selected")

        img = Image.fromarray(img)

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        img.save(str(output_path), "PNG")

        return output_path
