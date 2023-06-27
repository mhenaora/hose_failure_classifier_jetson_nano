#!/usr/bin/env python
#Power Loss Detection
import RPi.GPIO as GPIO
import Jetson.GPIO as GPIO
GPIO_PORT=6

GPIO.setmode(GPIO.BOARD)
GPIO.setup(GPIO_PORT, GPIO.IN)

# This is sampel code for python 2;
# Please change the 'print' if you are using the python 3
# Example:for python 2:
# print "---AC Power Loss OR Power Adapter Failure---"
# Example:for python 3:
# print("---AC Power Loss OR Power Adapter Failure---");

def my_callback(channel):
    if GPIO.input(GPIO_PORT):     # if port 6 == 1
        print("---AC Power Loss OR Power Adapter Failure---")
        GPIO.cleanup()
    else:                  # if port 6 != 1
        print("---AC Power OK,Power Adapter OK---")
        GPIO.cleanup()

GPIO.add_event_detect(GPIO_PORT, GPIO.BOTH, callback=my_callback)

print("1.Make sure your power adapter is connected")
print("2.Disconnect and connect the power adapter to test")
print("3.When power adapter disconnected, you will see: AC Power Loss or Power Adapter Failure")
print("4.When power adapter disconnected, you will see: AC Power OK, Power Adapter OK")

#raw_input("Testing Started")

