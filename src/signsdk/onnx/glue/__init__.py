"""Repo-free preprocessing 'glue' for the standalone ONNX runtime.

These are light numpy/PIL reimplementations of the preprocessing that the model
repos (Uni-Sign / SpaMo-PJM) do, extracted so the inference runtime does NOT need
to import or chdir into those repos. The heavy compute is in the ONNX graphs; this
is the surrounding decode/crop/normalize/postprocess.
"""
