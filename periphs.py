import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
mode = GPIO.getmode()
print(f"GPIO is in mode {mode}")

# According to a sample, pin 32 and 33 are PWM
output_pin1 = 32 
output_pin2 = 33 

#GPIO.setup(output_pin1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(output_pin2, GPIO.OUT, initial=GPIO.HIGH)
p1 = GPIO.PWM(output_pin1, 50) # encode as % of 400 cm signal
p2 = GPIO.PWM(output_pin2, 50)
val1 = 5
val2 = 1
incr = 5
p1.start(val1)
p2.start(val2)

try:
    while True:
        time.sleep(1)
        if val1 >= 100:
            incr = -incr
        if val1 <= 0:
            incr = -incr
        if val2 == 100:
            val2 = 0
        val1 += incr
        val2 += 1
        p1.ChangeDutyCycle(val1) # assuming value is percentage of duty cycle
        p2.ChangeDutyCycle(val2)
        print(f"{val1},  {val2}")
finally:
    p1.stop()
    p2.stop()
    GPIO.cleanup()

