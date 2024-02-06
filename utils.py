import cv2


def process_img(image, face_detection):
    H, W, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = face_detection.process(image_rgb)

    if out.detections is not None:  # if there is at least one face
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blurring face
            image[y1:y1 + h, x1:x1 + w, :] = cv2.blur(image[y1: y1 + h, x1: x1 + w, :], (30, 30))

    return image