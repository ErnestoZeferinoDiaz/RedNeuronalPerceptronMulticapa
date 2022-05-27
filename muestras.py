import serial, time

pathInDataRaw = 'data/raw/dataSet3.csv'


arduino = serial.Serial("COM4",9600,timeout=1.0)
arduino.setDTR(False)
time.sleep(1)
arduino.flushInput()
arduino.setDTR(True)
file = open(pathInDataRaw, 'a+')

for i in range(100):
    rawString = arduino.readline()
    cad=rawString.decode()[:-2]
    if(len(cad)>0):
        time.sleep(0.2)
        cad+="\n"
        file.write(cad)
        print(cad,end="")
file.close()
arduino.close()