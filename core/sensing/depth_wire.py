"""Lossless native-depth payloads, shared by server, deployment and viewer.

Keep this file beside native_depth_server.py when copying it to the camera PC.
An absent encoding field denotes the original PNG protocol.
"""
import numpy as np

PNG = 'png_uint16'
RAW = 'raw_uint16_le'


def encode_depth(depth, encoding='png'):
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Expected uint16 optical-axis depth')
    if encoding == 'raw':
        payload = np.ascontiguousarray(depth, dtype='<u2').tobytes()
        name = RAW
    elif encoding == 'png':
        import cv2
        ok, encoded = cv2.imencode('.png', depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError('Depth PNG encoding failed')
        payload, name = encoded.tobytes(), PNG
    else:
        raise ValueError(f'Unsupported depth encoding: {encoding}')
    return dict(encoding=name, shape=list(depth.shape)), payload


def decode_depth(metadata, payload):
    encoding = metadata.get('encoding', PNG)
    shape = metadata.get('shape')
    if shape is not None and (not isinstance(shape, list) or len(shape) != 2 or
                             any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError('Invalid native depth shape')
    if encoding == RAW:
        if shape is None or len(payload) != shape[0] * shape[1] * 2:
            raise ValueError('Raw depth payload size does not match shape')
        # The immutable message owns these pixels for the array's lifetime.
        image = np.frombuffer(payload, dtype='<u2').reshape(shape)
    elif encoding == PNG:
        import cv2
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError('Expected lossless uint16 optical depth PNG')
    else:
        raise ValueError(f'Unsupported depth encoding: {encoding}')
    if shape is not None and list(image.shape) != shape:
        raise ValueError('Depth payload shape differs from header')
    return image
