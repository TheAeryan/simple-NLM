# __init__.py

__version__ = "1.0.0"

# By default, we use the padded NLM
from .padded_NLM import NLM as NLM
from .unpadded_NLM import NLM as UnpaddedNLM

