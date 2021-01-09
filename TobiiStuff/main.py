import tobii_research as tr
import time
import argparse
#from pythonosc import udp_client
import threading as thr
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=list, default=5005,
                    help="The port the OSC server is listening on")
args = parser.parse_args()

#client = udp_client.SimpleUDPClient(args.ip, args.port)


f = open("outputTobii.txt", "w+")
f.write("")
f.close()
x,y = 0,0
def gaze_data_callback(gaze_data):
    global x,y
    # Print gaze points of left and right eye
    x = (gaze_data['left_gaze_point_on_display_area'][0] + gaze_data['right_gaze_point_on_display_area'][0])/2
    y = (gaze_data['left_gaze_point_on_display_area'][1] + gaze_data['right_gaze_point_on_display_area'][1])/2
    timestamp=datetime.now()
    print(f"X: ({x}) \t Y: ({y})")
    f = open("outputTobii.txt", "a")
    f.write(str(x) + " " + str(y) + " " + str(timestamp)+'\n')
    f.close()
def thread1():

    found_eyetrackers = tr.find_all_eyetrackers()
    my_eyetracker = found_eyetrackers[0]
    print("Address: " + my_eyetracker.address)
    print("Model: " + my_eyetracker.model)
    print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
    print("Serial number: " + my_eyetracker.serial_number)
    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    time.sleep(50)
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)

if __name__ == "__main__":
    t1 = thr.Thread(target=thread1)
    t1.start()
