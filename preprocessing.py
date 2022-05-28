from libs.normalizar import *
import json


config = json.load(open('config.json'))

pathInDataRaw = config["sample"]["pathSaveDataRaw"]
pathOutDataX  = config["pathData"]["normalizedX"]
pathOutDataY  = config["pathData"]["normalizedY"]
pathOutRows   = config["pathData"]["rows"]
pathOutXmin   = config["pathData"]["xmin"]
pathOutXmax   = config["pathData"]["xmax"]


# leemos los datos en crudo
data=np.genfromtxt(pathInDataRaw,delimiter=',',dtype="float")

# normalizamos
res = normalize(data)

#guardamos todos los datos ya que los necesitaremos para el test
np.save(pathOutDataX,res[0])
np.savetxt(pathOutDataX+".csv",res[0],delimiter=",",fmt="%g")

np.save(pathOutDataY,res[1])
np.savetxt(pathOutDataY+".csv",res[1],delimiter=",",fmt="%g")

np.save(pathOutRows,np.array([res[2]]))
np.savetxt(pathOutRows+".csv",np.array([res[2]]),delimiter=",",fmt="%g")

np.save(pathOutXmin,res[3])
np.savetxt(pathOutXmin+".csv",res[3],delimiter=",",fmt="%g")

np.save(pathOutXmax,res[4])
np.savetxt(pathOutXmax+".csv",res[4],delimiter=",",fmt="%g")

