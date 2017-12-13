# import pyserial
import serial
import sys
import time
from serial.tools.list_ports import comports as list_comports
from queue import Queue
from threading import Thread

SETUP_TIME = 1
INTERMESSAGE_TIME = .125

def message_worker(portname, q):
	print "Message worker started with portname {}".format(portname)

	done = False
	with serial.Serial(portname, 9200) as ser:
		time.sleep(SETUP_TIME)
		# print "Setup complete"
		while not done:
			message = q.get()
			if message is None:
				done = True
			else:
				ser.write(message + '\n')
				time.sleep(INTERMESSAGE_TIME)
				resp = ser.read(ser.inWaiting())
				if 'e' in resp:
					print "Error occured for message {}".format(message)

			q.task_done()

def find_serial_port(tag='usb'):
	port = None
	for s in list_comports():
	    if tag in s[0]:
	        port = s[0]
	        break

	if port is None:
	    print "NO SERIAL PORT COULD BE FOUND"
	    sys.exit()
	return port

def Color(r,g,b):
	r, g, b = int(r), int(g), int(b)
	assert r >= 0 and r < 256
	assert g >= 0 and g < 256
	assert b >= 0 and b < 256
	return (r << 16) | (g << 8) | b

class ThreadedSerialClient():

	def __init__(self, portname):
		self.q = Queue()
		self.portname = portname
		self.worker = Thread(target=message_worker, args=(self.portname, self.q))
		self.worker.start()

	def _send_message(self, message):
		self.q.put(message)

	def close(self):
		self.q.put(None)
		self.q.join()

class LEDDriver(ThreadedSerialClient):

	def on(self):
		self._send_message('on')

	def off(self):
		self._send_message('off')

	def red(self):
		self._send_message('r')

	def green(self):
		self._send_message('g')

	def blue(self):
		self._send_message('b')

	def set(self, brightness, color):
		assert brightness > 0 and brightness <= 255
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("set {} {}".format(brightness, color))

	def pulse(self, color, wait, stepsize):
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("p {} {} {}".format(color, wait, stepsize))

	def race(self, color, wait, count):
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("race {} {} {}".format(color, wait, count))

	def bars(self, color, wait, count):
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("fb {} {} {}".format(color, wait, count))

	def vbars(self, color, wait, count):
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("fbv {} {} {}".format(color, wait, count))

	def load(self, color, wait, count):
		assert color >= Color(0, 0, 0) and color <= Color(255, 255, 255)
		self._send_message("load {} {} {}".format(color, wait, count))

	def loadr(self, wait, count):
		self._send_message("loadr {} {}".format(wait, count))


if __name__ == "__main__":
	port = find_serial_port()
	cli = LEDDriver(port)

	cli.on()
	time.sleep(.5)
	# cli.red()
	# time.sleep(.5)
	# cli.blue()
	# time.sleep(.5)
	# cli.green()
	# time.sleep(.5)
	# cli.set(125, Color(255,255,0))
	# time.sleep(.5)
	# cli.pulse(Color(0,255,255), 5, 2)
	# time.sleep(.5)
	# cli.race(Color(0,255,0), 50, 20)
	# time.sleep(.5)
	cli.bars(Color(255,0,0), 200, 10)
	time.sleep(.5)
	cli.vbars(Color(0,255,0), 200, 10)
	time.sleep(.5)
	cli.load(Color(0,0,255), 30, 255)
	time.sleep(.5)
	cli.loadr(30, 255)
	cli.off()
	cli.close()

	# cli.close()
