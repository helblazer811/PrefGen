import sys
import os
sys.path.append(
    os.path.join(
        os.environ["PREFGEN_ROOT"],
        "prefgen",
        "methods/sampling/gan_control/gan_control/src"
    )
)

from .sampler import GANControlSampler
from gan_control import utils