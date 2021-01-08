import cv2
import sys

### python main.py pictures\team.png
# Obraz dostarczony przez użytkownika

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Utworzenie haar cascade na podstawie wytreonwanego modelu kaskadowego
faceCascade = cv2.CascadeClassifier(cascPath)

# Wczytanie obrazu
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykrywanie twarzy na obrazie
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=0
)

print("Znaleziono {0} twarzy!".format(len(faces)))

# Narysowanie prostokąta wokół twarzy
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)


cv2.imwrite("Obraz_po_detekcji.jpg", image)
cv2.imshow("Znalezione twarze", image)
cv2.waitKey(0) == 27
