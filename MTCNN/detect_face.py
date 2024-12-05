import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math

# OpenCV is optional, but required if using numpy arrays instead of PIL
try:
    import cv2
except ImportError:
    pass


def fixed_batch_process(im_data, model):
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i + batch_size)]
        out.append(model(batch))
    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    # Convert input to tensor if it's not already
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs, np.ndarray):
            imgs = torch.as_tensor(imgs.copy(), device=device)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs.clone().to(device)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        # Convert list of PIL Images to a 4D tensor
        imgs = torch.stack([F.to_tensor(img) for img in imgs], dim=0).to(device)

    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w) * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i *= factor
        minl *= factor

    # First stage
    boxes = []
    image_inds = []
    scale_picks = []
    offset = 0

    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        # Use PyTorch's batched_nms
        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, iou_threshold=0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes = boxes[scale_picks]
    image_inds = image_inds[scale_picks]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, iou_threshold=0.7)
    boxes = boxes[pick]
    image_inds = image_inds[pick]

    # Bounding box regression
    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.cat((torch.stack([qq1, qq2, qq3, qq4], dim=1), boxes[:, 4:]), dim=1)
    boxes = rerec(boxes)

    y, ey, x, ex = pad(boxes, w, h)

    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        if len(im_data) > 0:
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125

            # Run RNet
            out = fixed_batch_process(im_data, rnet)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > threshold[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, iou_threshold=0.7)
            boxes = boxes[pick]
            image_inds = image_inds[pick]
            mv = mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        if len(im_data) > 0:
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125

            # Run ONet
            out = fixed_batch_process(im_data, onet)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            points = out1
            ipass = score > threshold[2]
            points = points[:, ipass]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            w_i = boxes[:, 2] - boxes[:, 0] + 1
            h_i = boxes[:, 3] - boxes[:, 1] + 1
            points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
            points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
            points = torch.stack((points_x, points_y), dim=2)  # Shape: [5, N, 2] -> [N, 5, 2]
            points = points.permute(1, 0, 2)  # [N, 5, 2]
            boxes = bbreg(boxes, mv)

            # NMS within each image using "Min" strategy (using batched_nms)
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, iou_threshold=0.7)
            boxes = boxes[pick]
            image_inds = image_inds[pick]
            points = points[pick]

    # Prepare batch output without converting to NumPy
    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        # Find indices where image_inds == b_i
        b_i_inds = (image_inds == b_i).nonzero(as_tuple=True)[0]
        batch_boxes.append(boxes[b_i_inds].clone())
        batch_points.append(points[b_i_inds].clone())

    return batch_boxes, batch_points


def bbreg(boundingbox, reg):
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    b5 = boundingbox[:, 4]
    return torch.stack([b1, b2, b3, b4, b5], dim=1)


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero(as_tuple=False)
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def pad(boxes, w, h):
    boxes = boxes.trunc().int()
    x = torch.clamp(boxes[:, 0], min=1)
    y = torch.clamp(boxes[:, 1], min=1)
    ex = torch.clamp(boxes[:, 2], max=w)
    ey = torch.clamp(boxes[:, 3], max=h)
    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA_new = torch.empty_like(bboxA)
    bboxA_new[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA_new[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA_new[:, 2:4] = bboxA_new[:, :2] + l.unsqueeze(1).repeat(1, 2)

    return bboxA_new


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {torch.Tensor} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    if isinstance(face, np.ndarray) or isinstance(face, Image.Image):
        face = F.to_tensor(torch.from_numpy(face).float())
    elif isinstance(face, torch.Tensor):
        face = face.float()
    else:
        raise NotImplementedError

    return face
