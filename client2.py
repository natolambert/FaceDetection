import requests
import json
import datetime

from ratelimit import rate_limited
from queue import Queue
from threading import Thread

def message_worker(url, q):
    print "Message worker started with url {}".format(url)

    # @rate_limited(2)
    def _post(url, **data):
        return requests.post(url, data=data)

    done = False
    while not done:
        message = q.get()
        if message is None:
            done = True
        else:
            print _post(url + "/api/v1/log_data", **message)
        q.task_done()
    print "Worker exiting"

class AttenthiaClient():
    def __init__(self, baseurl):
        self.q = Queue()
        self.url = baseurl
        self.worker = Thread(target=message_worker, args=(self.url, self.q))
        self.worker.start()

    def log(self, driver_id, distraction_type, timestamp):
        data = {
                'driver_id':driver_id,
                'distraction_type':distraction_type,
                'timestamp':timestamp
                }
        self.q.put(data)

    def list(self, driver=None, username=None):
        if not driver is None:
            return requests.get(self.url + "/api/v1/log_data", params={"driver_id": driver}).json()
        elif not username is None:
            return requests.get(self.url + "/api/v1/log_data", params={"username": username}).json()
        else:
            return requests.get(self.url + "/api/v1/log_data").json()

    def close(self):
        print "Closing queue"
        self.q.put(None)
        self.q.join()
        print "Closed"

# if _name_ == '_main_':
#     # cli = AttenthiaClient("http://0.0.0.0:5000")
#     cli = AttenthiaClient("https://idd-wb-attenthia.herokuapp.com")
#     cli.log(2,0, datetime.datetime.now())
#     cli.log(2,0, datetime.datetime.now())
#     cli.log(2,0, datetime.datetime.now())
#     cli.log(2,0, datetime.datetime.now())
#     cli.log(2,0, datetime.datetime.now())
#     cli.log(2,0, datetime.datetime.now())
#     cli.close()
    # print json.dumps(cli.list(), indent=4)
    # print json.dumps(cli.list(driver=1), indent=4)
    # print json.dumps(cli.list(username="joe_schmoe"), indent=4)
    # print cli.list()
