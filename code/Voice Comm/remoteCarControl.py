import speech_recognition as sr
import socket
import sys


if len(sys.argv) < 2:
    print('plz enter the port number')
    sys.exit()

# server-side socket
s = socket.socket()
print("Socket created")

#port = 10000
port = int(sys.argv[1])
s.bind(('',port))
print("Socket binded to %d" % port)

s.listen(5)
print("Socket listening...")

c, addr = s.accept()  # blocking function?
print("Got a connection from ", addr)
c.setblocking(0)


r = sr.Recognizer()
m = sr.Microphone()

PRINT_DEBUG = True

try:
    print("A moment of silence, please...")
    with m as source: r.adjust_for_ambient_noise(source)
    
    if PRINT_DEBUG:
        print("Set minimum energy threshold to {}".format(r.energy_threshold))

    while True:
        print("Say something! (or say bye to quit)")
        with m as source: audio = r.listen(source)
        print("> Processing it...")
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)

            # we need some special handling here to 
            # correctly print unicode characters to standard output
            if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                print(u"> You said : {}".format(value).encode("utf-8"))
            else:  # this version of Python uses unicode for strings (Python 3+)
                print("> You said : {}".format(value))

            if value == 'bye':  # stop this program
                break;
            elif (value == 'start the car') or (value == 'start') :
                print('> start the car')
                msg = 'start the car'
                c.send(msg.encode('utf-8'))
            elif (value ==  'stop the car') or (value == 'stop') :
                print('> stop the car')
                msg = 'stop the car'
                c.send(msg.encode('utf-8'))
            elif (value == 'slow down') or (value == 'slow'):
                print('> slow down')
                msg = 'slow down'
                c.send(msg.encode('utf-8'))
            elif (value == 'speed up') or (value == 'speed'):
                print('> speed up')
                msg = 'speed up'
                c.send(msg.encode('utf-8'))
            elif value == 'accident':
                print('> accident')
                msg = 'accident'
                c.send(msg.encode('utf-8'))
            elif value == 'clear':
                print('> clear')
                msg = 'clear'
                c.send(msg.encode('utf-8'))
            else:
                print('> unrecognized command...')

        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    pass

c.close()
s.close()
