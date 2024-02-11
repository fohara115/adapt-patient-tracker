import serial

ser = serial.Serial('/dev/ttyACM0', 9600)

try:
	while True:
		data = ser.readline().decode().strip()
		print(data)

except KeyboardInterrupt:
	print('Keyboard interrupt')

finally:
	ser.close()
