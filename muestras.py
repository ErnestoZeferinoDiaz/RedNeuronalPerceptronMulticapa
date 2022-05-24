import serial, time

pathInDataRaw = 'data/raw/dataSet.csv'


arduino = serial.Serial("COM6",9600,timeout=1.0)
arduino.setDTR(False)
time.sleep(1)
arduino.flushInput()
arduino.setDTR(True)
file = open(pathInDataRaw, 'a+')

for i in range(26):
    rawString = arduino.readline()
    cad=rawString.decode()[:-2]
    if(len(cad)>0):
        time.sleep(0.5)
        cad+="\n"
        file.write(cad)
        print(cad,end="")
file.close()
arduino.close()