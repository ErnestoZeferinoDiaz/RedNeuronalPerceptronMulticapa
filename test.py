from libs.libs import *
from libs.functions import *

directory = open("paths.txt", "r").read().split("\n")

path = "../train/20.jpg"
x    = preprocessingIMG(path)

print(x.size)

rows = x.shape[0]

r = RedNeuronal(
   [x.size,100,50,len(directory)],
   [sigmoid,sigmoid,sigmoid],
   0.01
)

r.loadModel("checkpoints")
y=r.frontPropagation(x).round(0)

print(y)
