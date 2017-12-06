import requests
import json
import datetime

from ratelimit import rate_limited
from queue import Queue
from threading import Thread

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
            print resp
            print resp.json()
        q.task_done()
    print "Worker exiting"

class AttenthiaClient():
    def __init__(self, baseurl):
        self.q = Queue()
        self.url = baseurl
        self.worker = Thread(target=message_worker, args=(self.q,))
        self.worker.start()

    def log(self, driver_id, distraction_type, timestamp):
        data = {
                'driver_id':driver_id,
                'distraction_type':distraction_type,
                'timestamp':timestamp
                }
        self.q.put((data, self.url + '/api/v1/log_data'))

    def create_user(self, username):
        data = {
            'username': username
        }
        self.q.put((data, self.url + '/api/v1/users'))

    def close(self):
        print "Closing queue"
        self.q.put((None, None))
        self.q.join()
        print "Closed"
