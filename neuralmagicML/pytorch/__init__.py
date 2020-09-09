"""
Code for working with the pytorch framework for creating /
editing models for performance in the Neural Magic System
"""

try:
    import torch
    if torch.__version__[0] != "1":
        raise Exception
except:
    raise RuntimeError(
        "Unable to import torch. torch>=1.0.0 and a compatable version of torchvision"
        " are required to use neuralmagicML.pytorch"
    )

try:
    import torchvision
except:
    raise RuntimeError(
        "Unable to import torchvision. torch>=1.0.0 and a compatable version of torchvision"
        " are required to use neuralmagicML.pytorch"
    )
