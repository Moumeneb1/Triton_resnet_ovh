"""This script enables the use of onnx models using an onnxruntime for classification
Authors
 * Boumadane Abdelmoumene 2021
"""

# We check if transformers is installed.
try:
    import onnxruntime

except ImportError:
    MSG = "Please install onnxrtunime \n"
    MSG += "E.G. run: pip install onnxruntime-gpu"
    raise ImportError(MSG)


import numpy as np
import onnxruntime as onnxrt
import json 

class OnnxClassifier:
    """
    Arguments
    ---------
    model_path : str
        the onnx model path
    inference_engine : str
        The inference engine to use can be CPUExecutionProvider, CUDAExecutionProvider or TensorrtExecutionProvider
    """

    def __init__(self,
        model_path="model.onnx",
        inference_engine='CUDAExecutionProvider'
    ):
        
        self.model_path = model_path
        self.inference_engine = inference_engine
        self.onnx_session= onnxrt.InferenceSession(model_path,providers=[inference_engine])
    
    def infer(self,wav_array):
        """ Infers 16khz sampeled wav in numpy in type float32
        """
        output = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: wav_array})[0]
        return output

