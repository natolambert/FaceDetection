import requests
import json

class AttenthiaClient():
    def __init__(self, baseurl):
        self.url = baseurl

    def log(self, driver, severity):
        data = {
                'driver':driver,
                'severity':severity
                }
        return requests.post(self.url + "/api/v1/log_data", data=data)

    def list(self, driver=None):
        if driver is None:
            return requests.get(self.url + "/api/v1/log_data").json()
        else:
            return requests.get(self.url + "/api/v1/log_data", params={"driver": driver}).json()

if _name_ == '_main_':
    # cli = AttenthiaClient("http://0.0.0.0:5000")
    cli = AttenthiaClient("https://idd-wb-attenthia.herokuapp.com")
    cli.log(-100, 350)
    print json.dumps(cli.list(-100), indent=4)
    # print cli.list()
