import json
import jsonlines
import pickle
import csv
import re

class IOProcessor():
    def read_jsonline(self,file):
        file=open(file,"r",encoding="utf-8")
        data=[json.loads(line) for line in file.readlines()]
        return  data

    def write_jsonline(self,file,data):
        f=jsonlines.open(file,"w")
        for each in data:
            jsonlines.Writer.write(f,each)
        return

    def read_json(self,file):
        f=open(file,"r",encoding="utf-8").read()
        return json.loads(f)

    def write_json(self,file,data):
        f=open(file,"w",encoding="utf-8")
        json.dump(data,f,indent=2,ensure_ascii=False)
        return

    def read_pickle(self,filename):
        return pickle.loads(open(filename,"rb").read())

    def write_pickle(self,filename,data):
        open(filename,"wb").write(pickle.dumps(data))
        return


    def read_csv(self,filename):
        csv_data = csv.reader(open(filename, "r", encoding="utf-8"))
        csv_data=[each for each in csv_data]
        return csv_data


if __name__ == '__main__':
    pass











