from libs.libs import *
from libs.functions import *


directory=[
   "datasets/dollar",
   "datasets/euro",
   "datasets/franc",
   "datasets/pound",
   "datasets/rupee",
   "datasets/yen"
]

path = "datasets/pound/pound100.jpg"
img     = cv2.imread(path)
resized = get_square(img,60)
gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray    = np.matrix(gray)
x       = gray.getA1()

print(x.size)
rows = x.shape[0]
r = RedNeuronal(
   [x.size,100,50,len(directory)],
   [sigmoid,sigmoid,sigmoid],
   0.01
)

r.reset()
r.loadModel("checkpoints")
y=r.frontPropagation(x).round(0)

print(y)
