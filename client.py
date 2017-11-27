import requests
import json
import datetime
class AttenthiaClient():
    def __init__(self, baseurl):
        self.url = baseurl

    def _post(self, url, **data):
        return requests.post(url, data=data)

    def log(self, driver_id, distraction_type, timestamp):
        data = {
                'driver_id':driver_id,
                'distraction_type':distraction_type,
                'timestamp':timestamp
                }
        return self._post(self.url + "/api/v1/log_data", **data)

    def list(self, driver=None, username=None):
        if not driver is None:
            return requests.get(self.url + "/api/v1/log_data", params={"driver_id": driver}).json()
        elif not username is None:
            return requests.get(self.url + "/api/v1/log_data", params={"username": username}).json()
        else:
            return requests.get(self.url + "/api/v1/log_data").json()

# if _name_ == '_main_':
#     # cli = Atten`````thiaClient("http://0.0.0.0:5000")
#     cli = AttenthiaClient("https://idd-wb-attenthia.herokuapp.com")
#     # print cli.log(1, 0, datetime.datetime.now())
#     print json.dumps(cli.list(), indent=4)
#     print json.dumps(cli.list(driver=1), indent=4)
#     print json.dumps(cli.list(username="joe_schmoe"), indent=4)
#     # print cli.list()
