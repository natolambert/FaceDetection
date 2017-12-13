import requests
import json
import datetime

from ratelimit import rate_limited
from queue import Queue
from threading import Thread
from time import sleep

from random import random

def get_jacobs_lat_lon():
    lat, lon = 37.875393, -122.258436
    return lat + (random()-.5)/4000., lon + (random()-.5)/4000.

def message_worker(q):
    # print "Message worker started with url {}".format(url)

    # @rate_limited(2)
    def _post(url, **data):
        return requests.post(url, data=data)

    done = False
    while not done:
        message, url = q.get() 
        if message is None:
            done = True
        else:
            resp =  _post(url, **message)
            # print resp
            # print resp.json()
        q.task_done()
    print "Worker exiting"

class AttenthiaClient():
    def __init__(self, baseurl):
        self.q = Queue()
        self.url = baseurl
        self.worker = Thread(target=message_worker, args=(self.q,))
        self.worker.start()
        self.user = None
        self.trip = None

    def log(self, distraction_type, timestamp, lat, lon):
        if self.trip is None:
            print "NONONO can't do thattttttt"
            return
        data = {
                'distraction_type':distraction_type,
                'timestamp':timestamp,
                'lat': lat,
                'lon': lon
                }
        self.q.put((data, self.url + '/api/v1/trips/{}/events'.format(self.trip)))

    def create_user(self, username):
        data = {
            'username': username
        }
        resp = requests.post(self.url + '/api/v1/drivers', data=data).json()
        self.user = resp["id"]

    def start_trip(self):
        if self.user is None:
            print "Can't do thattttttt!!!"
            return 
        resp = requests.post(self.url + '/api/v1/drivers/{}/start_trip'.format(self.user)).json()
        self.trip = resp["id"]

    def stop_trip(self):
        self.trip = None

    def get_trip_url(self):
        return self.url + '/trips/{}'.format(self.trip)

    def close(self):
        print "Closing queue"
        self.q.put((None, None))
        self.q.join()
        print "Closed"

if __name__ == '__main__':
    try:
        cli = AttenthiaClient("http://0.0.0.0:5000")
        # cli = AttenthiaClient("https://idd-wb-attenthia.herokuapp.com/")
        cli.create_user("max_gerber")
        cli.start_trip()
        cli.log(1, datetime.datetime.now(), 10, -30)
        sleep(3)
        cli.log(0, datetime.datetime.now(), 10, -30)
    finally:
        cli.close()

