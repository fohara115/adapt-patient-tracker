import serial
import time

adam_uno = '/dev/ttyACM0'
BAUD = 9600
ser = serial.Serial(adam_uno, BAUD)
d = 1 # cm
val2 = 99
incr = 5

try:
    while True:
        time.sleep(0.5)
        if d > 400:
            print('reset')
            d = 1
        d = d + 1
        print(f"{d},  {val2}")
        ser.write(f"{d},{val2},1\n".encode('utf-8'))
finally:
    ser.close()

'''
Planning:

* distance
	- centimeters
	- rarely nan if depth sensor errors occur
	- None if no patient is found
* lateral angle
	- in degrees
	- None if no patient is found
* patient state
	- 0 for idle mode
	- 1 for "patient is seated with device enabled" -> brake should be on
	- 2 for "patient is on the move" -> brake off, control systems on
	- 3 for "lost patient" mode, accompanied by None signals. The system knows that it lost the patient.
* following distance
	- close (70 cm?)
	- medium (100 cm?)
	- far (130 cm?)
'''

