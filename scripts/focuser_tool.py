# encoding: UTF-8
#!/usr/bin/env python
'''
    Arducam programable zoom-lens controller.

    Copyright (c) 2019-4 Arducam <http://www.arducam.com>.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
'''

from email.mime import image
import cv2  # sudo apt-get install python-opencv
import numpy as py
import os
import sys
import time
import argparse
# from JetsonCamera import Camera
from pathlib import Path
from utils.JetsonCamera import Camera
from utils.Focuser import Focuser
# from AutoFocus import AutoFocus
import curses
# Variables for each kind of class (SWAN-GAS-BICOLOR-PRESION)(good-even numbers)(bad-odd numbers)
global image_count1
global image_count2
global image_count3
global image_count4
global image_count5
global image_count6
global image_count7
global image_count8
global image_count9
global image_count10
global image_count11
global image_count12
global image_count13
global flag
global version

flag = ""
version = ""
image_count1 = 0
image_count2 = 0
image_count3 = 0
image_count4 = 0
image_count5 = 0
image_count6 = 0
image_count7 = 0
image_count8 = 0
image_count9 = 0
image_count10 = 0
image_count11 = 0
image_count12 = 0
image_count13 = 0
# Rendering status bar


def RenderStatusBar(stdscr):
    height, width = stdscr.getmaxyx()
    statusbarstr = "Press 'q' to exit"
    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(height-1, 0, statusbarstr)
    stdscr.addstr(height-1, len(statusbarstr), " " *
                  (width - len(statusbarstr) - 1))
    stdscr.attroff(curses.color_pair(3))
# Rendering description


def RenderDescription(stdscr):
    focus_desc = "FOCUS    : UP-DOWN ARROW"
    default_desc = "SET FOCUS VALUE : 's' Key"
    CAM_ALL_desc = "CAM (4-1) 1-2-3-4 : '0' Key"
    CAM_1_1_desc = "CAM (1-1): Keys '1,2,3,4'"
    CAM_2_1_desc = "CAM (2-1) : Keys '5,6'"
    SWAN_desc = "SWAN (GOOD,FAILURE,WARNING): Keys'A,B,C'"
    GAS_desc = "GAS (GOOD,FAILURE,WARNING): Keys'D,E,F'"
    BICOLOR_desc = "BICOLOR (GOOD,FAILURE,WARNING): Keys'G,H,I'"
    PRESION_desc = "PRESION (GOOD,FAILURE,WARNING): Keys'J,K,L'"
    BACKGROUND_desc = "BACKGROUND: 'M' Key"

    desc_y = 1

    stdscr.addstr(desc_y + 1, 0, focus_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 2, 0, default_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 3, 0, CAM_1_1_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 4, 0, CAM_2_1_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 5, 0, CAM_ALL_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 6, 0, SWAN_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 7, 0, GAS_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 8, 0, BICOLOR_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 9, 0, PRESION_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 10, 0, BACKGROUND_desc, curses.color_pair(1))

# Rendering  middle text


def RenderMiddleText(stdscr, key, focuser):
    global image_count1
    # get height and width of the window.
    height, width = stdscr.getmaxyx()
    # Declaration of strings
    title = "Controlador Enfoque y Modo Camara IMX519 "[:width-1]
    subtitle = ""[:width-1]
    keystr = "Ultima tecla presionada: {}".format(key)[:width-1]

    # Obtain device infomation
    focus_value = "Enfoque    : {}".format(
        focuser.get(Focuser.OPT_FOCUS))[:width-1]

    if key == 0:
        keystr = "No se detecto tecla presionada..."[:width-1]

    # Centering calculations
    start_x_title = int((width // 2) - (len(title) // 2) - len(title) % 2)
    start_x_subtitle = int(
        (width // 2) - (len(subtitle) // 2) - len(subtitle) % 2)
    start_x_keystr = int((width // 2) - (len(keystr) // 2) - len(keystr) % 2)
    start_x_device_info = int(
        (width // 2) - (len("Focus    : 00000") // 2) - len("Focus    : 00000") % 2)
    start_y = int((height // 2) - 6)
    image_count1
    # Turning on attributes for title
    stdscr.attron(curses.color_pair(2))
    stdscr.attron(curses.A_BOLD)

    # Rendering title
    stdscr.addstr(start_y, start_x_title, title)

    # Turning off attributes for title
    stdscr.attroff(curses.color_pair(2))
    stdscr.attroff(curses.A_BOLD)

    # Print rest of text
    stdscr.addstr(start_y + 1, start_x_subtitle, subtitle)
    stdscr.addstr(start_y + 3, (width // 2) - 2, '-' * 4)
    stdscr.addstr(start_y + 5, start_x_keystr, keystr)
    # Print device info
    stdscr.addstr(start_y + 6, start_x_device_info, focus_value)
    # Print image counter
    # exit
    # stdscr.addstr(start_y + 7, str(image_count1), str(image_count1))


def parse_cmdline():
    parser = argparse.ArgumentParser(description='Arducam Controller.')

    parser.add_argument('-i', '--i2c-bus', type=int, required=False, default=7,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')
    parser.add_argument('--version', dest='version',
                        help='Dataset Version',
                        default="2.0")
    parser.add_argument('--mode', dest='mode',
                        help='Camera Mode',
                        default="2-1")
    return parser.parse_args()

# parse input key


def parseKey(key, focuser, auto_focus, camera):
    global image_count1
    global image_count2
    global image_count3
    global image_count4
    global image_count5
    global image_count6
    global image_count7
    global image_count8
    global image_count9
    global image_count10
    global image_count11
    global image_count12
    global image_count13
    global flag
    global version
    version = "2.0"  # "1.0"
    focus_step = 10
    if key == ord('r'):
        focuser.reset(Focuser.OPT_FOCUS)
    elif key == ord('s'):
        focuser.set(Focuser.OPT_FOCUS, 800)
    elif key == ord('S'):
        focuser.set(Focuser.OPT_FOCUS, 900)
    elif key == curses.KEY_UP:
        focuser.set(Focuser.OPT_FOCUS, focuser.get(
            Focuser.OPT_FOCUS) + focus_step)
    elif key == curses.KEY_DOWN:
        focuser.set(Focuser.OPT_FOCUS, focuser.get(
            Focuser.OPT_FOCUS) - focus_step)

    elif key == ord('0'):
        i2c = "i2cset -y 7 0x24 0x24 0x00"
        flag = "4-1"
        os.system(i2c)
    elif key == ord('1'):
        i2c = "i2cset -y 7 0x24 0x24 0x02"
        flag = "1-1"
        os.system(i2c)
    elif key == ord('2'):
        i2c = "i2cset -y 7 0x24 0x24 0x12"
        flag = "1-1"
        os.system(i2c)
    elif key == ord('3'):
        i2c = "i2cset -y 7 0x24 0x24 0x22"
        flag = "1-1"
        os.system(i2c)
    elif key == ord('4'):
        i2c = "i2cset -y 7 0x24 0x24 0x32"
        flag = "1-1"
        os.system(i2c)
    elif key == ord('5'):
        i2c = "i2cset -y 7 0x24 0x24 0x01"
        flag = "2-1"
        os.system(i2c)
    elif key == ord('6'):
        i2c = "i2cset -y 7 0x24 0x24 0x11"
        flag = "2-1"
        os.system(i2c)

    elif key == ord('A'):  # SWAN GOOD
        while True:
            path = Path(
                "dataset/{}/{}/SWAN/GOOD/SWAN_GOOD_{}_.jpg".format(version, flag, image_count1))
            if path.is_file() == True:
                image_count1 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite("dataset/{}/{}/SWAN/GOOD/SWAN_GOOD_{}_.jpg".format(version,
                    flag, image_count1), camera.getFrame())
        print("Image Saved at dataset/{}/{}/SWAN/GOOD/SWAN_GOOD_{}.jpg".format(version, flag, image_count1))
        image_count1 += 1

    elif key == ord('B'):  # SWAN FAILURE
        while True:
            path = Path(
                "dataset/{}/{}/SWAN/FAILURE/SWAN_FAILURE_{}_.jpg".format(version, flag, image_count2))
            if path.is_file() == True:
                image_count2 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/SWAN/FAILURE/SWAN_FAILURE_{}_.jpg".format(version, flag, image_count2), camera.getFrame())
        print("Image Saved at dataset/{}/{}/SWAN/FAILURE/SWAN_FAILURE_{}.jpg".format(version, flag, image_count2))
        image_count2 += 1

    elif key == ord('C'):  # SWAN WARNING
        while True:
            path = Path(
                "dataset/{}/{}/SWAN/WARNING/SWAN_WARNING_{}_.jpg".format(version, flag, image_count3))
            if path.is_file() == True:
                image_count3 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/SWAN/WARNING/SWAN_WARNING_{}_.jpg".format(version, flag, image_count3), camera.getFrame())
        print("Image Saved at dataset/{}/{}/SWAN/WARNING/SWAN_WARNING_{}.jpg".format(version, flag, image_count3))
        image_count3 += 1

    elif key == ord('D'):  # GAS GOOD
        while True:
            path = Path(
                "dataset/{}/{}/GAS/GOOD/GAS_GOOD_{}_.jpg".format(version, flag, image_count4))
            if path.is_file() == True:
                image_count4 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/GAS/GOOD/GAS_GOOD_{}_.jpg".format(version, flag, image_count4), camera.getFrame())
        print("Image Saved at dataset/{}/{}/GAS/GOOD/GAS_GOOD_{}.jpg".format(version, flag, image_count4))
        image_count4 += 1

    elif key == ord('E'):  # GAS FAILURE
        while True:
            path = Path(
                "dataset/{}/{}/GAS/FAILURE/GAS_FAILURE_{}_.jpg".format(version, flag, image_count5))
            if path.is_file() == True:
                image_count5 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/GAS/FAILURE/GAS_FAILURE_{}_.jpg".format(version, flag, image_count5), camera.getFrame())
        print("Image Saved at dataset/{}/{}/GAS/FAILURE/GAS_FAILURE_{}.jpg".format(version, flag, image_count5))
        image_count5 += 1

    elif key == ord('F'):  # GAS WARNING
        while True:
            path = Path(
                "dataset/{}/{}/GAS/WARNING/GAS_WARNING_{}_.jpg".format(version, flag, image_count6))
            if path.is_file() == True:
                image_count6 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/GAS/WARNING/GAS_WARNING_{}_.jpg".format(version, flag, image_count6), camera.getFrame())
        print("Image Saved at dataset/{}/{}/GAS/WARNING/GAS_WARNING_{}.jpg".format(version, flag, image_count6))
        image_count6 += 1

    elif key == ord('G'):  # BICOLOR GOOD
        while True:
            path = Path(
                "dataset/{}/{}/BICOLOR/GOOD/BICOLOR_GOOD_{}.jpg".format(version, flag, image_count7))
            if path.is_file() == True:
                image_count7 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/BICOLOR/GOOD/BICOLOR_GOOD_{}.jpg".format(version, flag, image_count7), camera.getFrame())
        print("Image Saved at dataset/{}/{}/BICOLOR/GOOD/BICOLOR_GOOD_{}.jpg".format(version, flag, image_count7))
        image_count7 += 1

    elif key == ord('H'):  # BICOLOR FAILURE
        while True:
            path = Path(
                "dataset/{}/{}/BICOLOR/FAILURE/BICOLOR_FAILURE_{}.jpg".format(version, flag, image_count8))
            if path.is_file() == True:
                image_count8 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/BICOLOR/FAILURE/BICOLOR_FAILURE_{}.jpg".format(version, flag, image_count8), camera.getFrame())
        print("Image Saved at dataset/{}/{}/BICOLOR/FAILURE/BICOLOR_FAILURE_{}.jpg".format(
            version, flag, image_count8))
        image_count8 += 1

    elif key == ord('I'):  # BICOLOR WARNING
        while True:
            path = Path(
                "dataset/{}/{}/BICOLOR/WARNING/BICOLOR_WARNING_{}.jpg".format(version, flag, image_count9))
            if path.is_file() == True:
                image_count9 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/BICOLOR/WARNING/BICOLOR_WARNING_{}.jpg".format(version, flag, image_count9), camera.getFrame())
        print("Image Saved at dataset/{}/{}/BICOLOR/WARNING/BICOLOR_WARNING_{}.jpg".format(
            version, flag, image_count9))
        image_count9 += 1

    elif key == ord('J'):  # PRESION GOOD
        while True:
            path = Path(
                "dataset/{}/{}/PRESION/GOOD/PRESION_GOOD_{}.jpg".format(version, flag, image_count10))
            if path.is_file() == True:
                image_count10 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/PRESION/GOOD/PRESION_GOOD_{}.jpg".format(version, flag, image_count10), camera.getFrame())
        print("Image Saved at dataset/{}/{}/PRESION/GOOD/PRESION_GOOD_{}.jpg".format(
            version, flag, image_count10))
        image_count10 += 1

    elif key == ord('K'):  # PRESION FAILURE
        while True:
            path = Path(
                "dataset/{}/{}/PRESION/FAILURE/PRESION_FAILURE_{}.jpg".format(version, flag, image_count11))
            if path.is_file() == True:
                image_count11 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/PRESION/FAILURE/PRESION_FAILURE_{}.jpg".format(version, flag, image_count11), camera.getFrame())
        print("Image Saved at dataset/{}/{}/PRESION/FAILURE/PRESION_FAILURE_{}.jpg".format(
            version, flag, image_count11))
        image_count11 += 1        

    elif key == ord('L'):  # PRESION WARNING
        while True:
            path = Path(
                "dataset/{}/{}/PRESION/WARNING/PRESION_WARNING_{}.jpg".format(version, flag, image_count12))
            if path.is_file() == True:
                image_count12 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/PRESION/WARNING/PRESION_WARNING_{}.jpg".format(version, flag, image_count12), camera.getFrame())
        print("Image Saved at dataset/{}/{}/PRESION/WARNING/PRESION_WARNING_{}.jpg".format(
            version, flag, image_count12))
        image_count12 += 1        

    elif key == ord('M'):  # BACKGROUND
        while True:
            path = Path(
                "dataset/{}/{}/SWAN/BACKGROUND/BACKGROUND_{}_.jpg".format(version, flag, image_count13))
            if path.is_file() == True:
                image_count13 += 1
            elif path.is_file() == False:
                break
        # save image to file.
        cv2.imwrite(
            "dataset/{}/{}/SWAN/BACKGROUND/BACKGROUND_{}_.jpg".format(version, flag, image_count13), camera.getFrame())
        print("Image Saved at dataset/{}/{}/SWAN/BACKGROUND/BACKGROUND_{}_.jpg".format(
            version, flag, image_count13))
        image_count13 += 1


def draw_menu(stdscr, camera, i2c_bus):
    focuser = Focuser(i2c_bus)
    # auto_focus = AutoFocus(focuser,camera)
    auto_focus = None
    key = 0
    cursor_x = 0
    cursor_y = 0

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Loop where key is the last character pressed
    while (key != ord('q')):
        # Initialization
        stdscr.clear()
        # Flush all input buffers.
        curses.flushinp()
        # get height and width of the window.
        height, width = stdscr.getmaxyx()
        # parser input key
        parseKey(key, focuser, auto_focus, camera)
        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))
        # render key description
        RenderDescription(stdscr)
        # render status bar
        RenderStatusBar(stdscr)
        # render middle text
        RenderMiddleText(stdscr, key, focuser)
        # Refresh the screen
        stdscr.refresh()
        # Wait for next input
        key = stdscr.getch()


def main():
    try:
        args = parse_cmdline()
        # open camera
        camera = Camera()
        # set focus value
        focuser = Focuser(7)
        # Recommended Values Around 700-900
        focuser.set(Focuser.OPT_FOCUS, 800)
        # access to camera #1-2-3-4
        i2c = "i2cset -y 7 0x24 0x24 0x00"
        os.system(i2c)

        camera.start_preview()

        i2c_bus = args.i2c_bus
        version = args.version
        print("i2c bus: {}".format(i2c_bus))
        print("Dataset version: {}".format(version))
        curses.wrapper(draw_menu, camera, i2c_bus)

    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)

    except KeyboardInterrupt as e:
        print("Exception KeyboardInterrupt:", e)
    finally:
        # Clean up
        camera.stop_preview()
        camera.close()


if __name__ == "__main__":
    main()
