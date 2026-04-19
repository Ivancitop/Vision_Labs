import cv2
import numpy as np

def gaussian_blur(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def sobel_filters(img):
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
    Ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], np.float32)

    Gx = cv2.filter2D(img, -1, Kx)
    Gy = cv2.filter2D(img, -1, Ky)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = magnitude / magnitude.max() * 255

    theta = np.arctan2(Gy, Gx)

    return magnitude, theta

def non_max_suppression(magnitude, theta):
    H, W = magnitude.shape
    Z = np.zeros((H, W), dtype=np.float32)

    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255

            # 0°
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # 45°
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # 90°
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # 135°
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0

    return Z

def threshold(img, lowThreshold, highThreshold):
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.uint8)

    weak = 75
    strong = 255

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(img, weak, strong=255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    return img

def canny_from_scratch(img, low=50, high=150):
    # 1. Suavizado
    blur = gaussian_blur(img)

    # 2. Gradiente (filtros derivativos tipo Sobel)
    magnitude, theta = sobel_filters(blur)

    # 3. Supresión de no máximos
    nms = non_max_suppression(magnitude, theta)

    # 4. Umbral doble
    thresh, weak, strong = threshold(nms, low, high)

    # 5. Histéresis
    result = hysteresis(thresh, weak, strong)

    return result


# ===== USO =====
img = cv2.imread("img/kodakimagecollection/kodim01.png", 0)

edges = canny_from_scratch(img)

cv2.imwrite('edges_canny_manual.jpg', edges)
cv2.imshow("Edges", edges)
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()