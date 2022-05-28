import serial, time
import json


config = json.load(open('config.json'))

# puerto USB donde se escuchara al arduino
port = config["arduino"]["port"]

#ruta donde se guardaran las muestras
pathInDataRaw = config["sample"]["pathSaveDataRaw"]

# Numero de muestras a tomar
noSamples = int(config["sample"]["numberSamplesForEachLetter"])

# Letra que se va a muestrear
letterToBeSampled= int(config["sample"]["letterToBeSampled"])

# tiempo de muestreo
timeSampled = float(config["sample"]["samplingTimeInSeconds"])


arduino = serial.Serial(port,9600,timeout=1.0)
arduino.setDTR(False)
time.sleep(1)
arduino.flushInput()
arduino.setDTR(True)
file = open(pathInDataRaw, 'a+')

for i in range(noSamples):
    rawString = arduino.readline()
    cad=rawString.decode()[:-2]
    if(len(cad)>0):
        time.sleep(timeSampled)
        cad+=","+str(letterToBeSampled)
        cad+="\n"
        file.write(cad)
        print(cad,end="")
file.close()
arduino.close()