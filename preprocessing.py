from libs.normalizar import *

pathInDataRaw         = 'data/raw/dataSet.csv'
pathOutDataX          = 'data/normalized/X'
pathOutDataY          = 'data/normalized/Y'
pathOutRows           = 'data/support/rows'
pathOutXmin           = 'data/support/Xmin'
pathOutXmax           = 'data/support/Xmax'


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

