import h5py
import numpy as np

Path = "./../wsdream/dataset2/"
#Path = "./../../dataset/"
lookback = 5
time_low = 0
time_high = 63
records_per_service = {}


class record:
    def __init__(self, user, rtime):
        self.user = user
        self.rtime = rtime

    def set_thput(self, thput):
        self.thput = thput


def fetch_past_records(service, time, usr_id):
    res = ""
    try:
        for count in range(time-lookback, time):
            for record in records_per_service.get(service).get(count):
                if record.user == usr_id:
                    res += str(record.rtime) + ","
    except:
        return ""
    return res


with open(Path + "rtdata.txt", 'r') as r:
    for line in r:
        data = list(map(float, line.strip().split(" ")))
        if data[1] in records_per_service.keys():
            temp_service = records_per_service.get(data[1])
            if data[2] in temp_service.keys():
                temp_time = temp_service.get(
                    data[2]) + [record(data[0], data[3])]
                temp_service[data[2]] = temp_time
            else:
                temp_service[data[2]] = [record(data[0], data[3])]
        else:
            records_per_service[data[1]] = {
                data[2]: [record(data[0], data[3])]}

with open(Path + "tpdata.txt", 'r') as t:
    for line in t:
        data = list(map(float, line.strip().split(" ")))
        for item in records_per_service.get(data[1]).get(data[2]):
            if item.user == data[0]:
                item.set_thput(data[3])

with open("./../Input/training_data.csv", 'w') as f:
    for service in records_per_service.keys():
        for time in range(time_low+lookback, time_high+1):
            try:
                for record in records_per_service.get(service).get(time):
                    usr_id = record.user
                    past_data = fetch_past_records(service, time, usr_id)
                    if len(past_data.split(",")) == lookback+1 and past_data != "":
                        f.write(str(record.thput)+"," +
                                past_data+str(record.rtime)+"\n")
                    else:
                        print("ghapla")
            except:
                pass
