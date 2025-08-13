import cv2
import numpy as np
from google.colab import files

uploaded = files.upload()
video_path = list(uploaded.keys())[0]

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def process_image(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    vertices = np.array([[
        (50, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width - 50, height)
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=50, maxLineGap=200)
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result

def main():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result = process_image(frame)
            # Instead of cv2.imshow, display frames inline in Colab:
            from google.colab.patches import cv2_imshow
            cv2_imshow(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
