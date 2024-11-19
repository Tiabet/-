
# Implementation from https://github.com/hukkelas/DSFD-Pytorch-Inference
# Original code and model: https://github.com/TencentYoutuResearch/FaceDetection-DSFD


from dsfd import detect
import torch

from common.det_face import DetFace

Name = 'DSFD'


def __load_model():
    return detect.DSFDDetector()


__model = __load_model()


# def detect_faces(frame, thresh=0.1, nms_iou_threshold=0.3):
#
#     faces = __model.detect_face(frame, thresh, nms_iou_threshold)
#
#     # det_faces = [DetFace(b[4], (b[0], b[1], b[2], b[3])) for b in faces]
#     return faces

def detect_faces(frame, thresh=0.1, nms_iou_threshold=0.3):
    print(f"[DEBUG] image_tensor requires_grad: {frame.requires_grad}")
    print(f"[DEBUG] image_tensor grad_fn: {frame.grad_fn}")

    # Run the detection using the DSFD model
    faces = __model.detect_face(frame, thresh, nms_iou_threshold)

    # Check if detections are connected to the graph
    print(f"[DEBUG] detections grad_fn: {faces.grad_fn if isinstance(faces, torch.Tensor) else 'None'}")

    return faces