import Jetson.GPIO as GPIO
import time

#
# # GPIO INPUTS
# inPin_start = 11  # start button GPIO pin
# inPin_stop = 13  # stop button GPIO pin
# inPin_Reset = 15  # reset button GPIO pin
# inPin_SWAN = 19  # SWAN button GPIO pin
# inPin_GAS = 21  # GAS button GPIO pin
# inPin_PRESION = 23  # BICOLOR button GPIO pin
# inPin_BICOLOR = 29  # PRESION button GPIO pin
# inPin_CRISTAL = 31  # CRISTAL button GPIO pin
# inPin_POWER_ON = 33  # POWER ON button GPIO pin
# inPin_POWER_OFF = 35  # POWER OFF button GPIO pin
# # GPIO OUTPUTS
# outPin_GOOD = 12  # Output for GOOD detection GPIO pin
# outPin_WARNING = 16  # Output for WARNING detection GPIO pin
# outPin_FAILURE = 18  # Output for FAILURE detection GPIO pin
# outPin_SWAN = 22  # Output for SWAN class GPIO pin
# outPin_GAS = 24  # Output for GAS class GPIO pin
# outPin_PRESION = 26  # Output for PRESION class GPIO pin
# outPin_BICOLOR = 32  # Output for BICOLOR class GPIO pin
# outPin_CRISTAL = 36  # Output for CRISTAL class GPIO pin
def main ():
    #GPIO INPUTS
    switch_nc=1
    switch_no=0
    inPin_start = 16  # start button GPIO pin
    inPin_stop = 12  # stop button GPIO pin
    inPin_reset = 13  # reset button GPIO pin
    inPin_SWAN = 15  # SWAN button GPIO pin
    inPin_GAS = 16  # GAS button GPIO pin
    inPin_PRESION = 22  # BICOLOR button GPIO pin
    inPin_BICOLOR = 18  # PRESION button GPIO pin
    inPin_CRISTAL = 31  # CRISTAL button GPIO pin
    inPin_POWER_ON = 19  # POWER ON button GPIO pin
    inPin_POWER_OFF = 35  # POWER OFF button GPIO pin
    # GPIO OUTPUTS
    outPin_GOOD = 29  # Output for GOOD detection GPIO pin
    outPin_WARNING = 33  # Output for WARNING detection GPIO pin
    outPin_FAILURE = 24  # Output for FAILURE detection GPIO pin
    outPin_SWAN = 31  # Output for SWAN class GPIO pin
    outPin_GAS = 26  # Output for GAS class GPIO pin
    outPin_PRESION = 23  # Output for PRESION class GPIO pin
    outPin_BICOLOR = 32  # Output for BICOLOR class GPIO pin
    prev_value = None
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(inPin_start, GPIO.IN)#,pull_up_down=GPIO.PUD_UP)
    GPIO.setup(inPin_stop, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(inPin_reset, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(inPin_reset, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(inPin_SWAN, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(inPin_GAS, GPIO.IN)#,pull_up_down=GPIO.PUD_DOWN)
    # GPIO.setup(inPin_PRESION, GPIO.IN)
    # GPIO.setup(inPin_BICOLOR, GPIO.IN)

    GPIO.setup(outPin_GOOD,GPIO.OUT)
    GPIO.setup(outPin_WARNING,GPIO.OUT)
    GPIO.setup(outPin_FAILURE,GPIO.OUT)
    GPIO.setup(outPin_SWAN,GPIO.OUT)
    GPIO.setup(outPin_GAS,GPIO.OUT)
    GPIO.setup(outPin_PRESION,GPIO.OUT)
    GPIO.setup(outPin_BICOLOR,GPIO.OUT)

    GPIO.output(outPin_GOOD,0)
    GPIO.output(outPin_WARNING,0)
    GPIO.output(outPin_FAILURE,0)
    GPIO.output(outPin_SWAN,0)
    GPIO.output(outPin_GAS,0)
    GPIO.output(outPin_PRESION,0)
    GPIO.output(outPin_BICOLOR,0)

    try:
        print("Start Demo CTRL-C to exit: \n")
        while True:
            x1=GPIO.input(inPin_start)
            x2=GPIO.input(inPin_stop)
            x3=GPIO.input(inPin_reset)
            x4=GPIO.input(inPin_SWAN)
            x5=GPIO.input(inPin_GAS)
            #x6=GPIO.input(inPin_PRESION)
            # x7=GPIO.input(inPin_BICOLOR)
            #if x1 != prev_value:
            if x1 == GPIO.HIGH:
                x1_str = "HIGH"
                GPIO.output(outPin_GOOD,1)
                # GPIO.output(outPin_WARNING,1)
                # GPIO.output(outPin_FAILURE,1)
                # GPIO.output(outPin_SWAN,1)
                # GPIO.output(outPin_GAS,1)
                # GPIO.output(outPin_PRESION,1)
                # GPIO.output(outPin_BICOLOR,1)
            else:
                x1_str = "LOW"
                GPIO.output(outPin_GOOD,0)
                # GPIO.output(outPin_WARNING,0)
                # GPIO.output(outPin_FAILURE,0)
                # GPIO.output(outPin_SWAN,0)
                # GPIO.output(outPin_GAS,0)
                # GPIO.output(outPin_PRESION,0)
                # GPIO.output(outPin_BICOLOR,0)

            print("start:{} stop:{} reset:{} Swan:{} Gas:{}".format(x1,x2,x3,x4,x5))
            prev_value = x1
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("GPIO Cleaned")
        GPIO.output(outPin_GOOD,0)
        GPIO.output(outPin_WARNING,0)
        GPIO.output(outPin_FAILURE,0)
        GPIO.output(outPin_SWAN,0)
        GPIO.output(outPin_GAS,0)
        # GPIO.output(outPin_PRESION,0)
        # GPIO.output(outPin_BICOLOR,0)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
