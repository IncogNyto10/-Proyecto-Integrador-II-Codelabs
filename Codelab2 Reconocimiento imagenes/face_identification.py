import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time
import os, sys

# Carga imagen (sube archivos en Colab o usa una URL y descárgala)

# Rutas a imágenes (asegúrate de que existan en imgs/)
script_dir = os.path.dirname(__file__) or '.'
img_paths = [
    os.path.join(script_dir, 'imgs', 'woman1.png'),  # original — en muchos repositorios el archivo puede llamarse "woman.png"
    os.path.join(script_dir, 'imgs', 'group1.png'),
    os.path.join(script_dir, 'imgs', 'middle_face1.png')
]

# Si alguno no existe, intenta nombres alternativos conocidos
if not os.path.exists(img_paths[0]):
    alt = os.path.join(script_dir, 'imgs', 'woman.png')
    if os.path.exists(alt):
        img_paths[0] = alt

imgs = []
for p in img_paths:
    if not os.path.exists(p):
        print(f"Error: no se encontró el archivo: {p}\nComprueba que la carpeta imgs/ contenga las imágenes o actualiza las rutas en el script.")
        sys.exit(1)
    imgs.append(cv2.imread(p))

img1, img2, img3 = imgs

img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

detector = MTCNN()   # puedes ajustar min_face_size
t0 = time()
res = detector.detect_faces(img_rgb)
t1 = time()

# Detección o no de un rostro
print(f"Detected: {len(res)} rostro(s) • tiempo: {(t1 - t0)*1000:.1f} ms")
for r in res:
    print(r['confidence'], r['box'], r['keypoints'].keys())

# Dibuja cajas y landmarks
vis = img_rgb.copy()
for r in res:
    x, y, w, h = r['box']
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
    for name, (px,py) in r['keypoints'].items():
        cv2.circle(vis, (px,py), 2, (255,0,0), -1)
plt.imshow(vis)
plt.axis('off')
plt.show()

# Umbrales y NMS (entendimiento de falsos positivos)
detector = MTCNN()  # su NMS interno filtra solapes
# Prueba filtrado por confianza mínima
thr = 0.90
filtrados = [r for r in res if r['confidence'] >= thr]
print(f"Con thr={thr} quedan {len(filtrados)} rostros")

# IoU simple (evaluación rápida)
def iou(a, b):
    # a,b en formato [x,y,w,h]
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1)*max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

# Si tienes una caja "ground truth" gt_box, compara:
# print(iou(res[0]['box'], gt_box))
