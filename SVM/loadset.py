# -*- coding:utf-8 -*-
import os
import numpy as np
class SVMDataLoader(object):
    def __init__(self,path):
        self.txtpath=path
        self.setrow=[]
        self.setlabel=[]
    def readdata(self,path):
        try:
            with open(path) as fp:
                line = fp.readline()
                while line:
                    if line=='\n' :
                        line = fp.readline()
                        continue

                    data=line.split(',')
                    self.setlabel.append(int(data[1].strip("\n")))
                    train=data[0].split(' ')
                    features=[float(x) for x in train]
                    features=features[0:4]
                    self.setrow.append(features)
                    line = fp.readline()

        except Exception as e:
            print("%s",e)

    def loaddatafromtxt(self):
        self.readdata(self.txtpath)
        return np.array(self.setrow,dtype=np.float),np.array(self.setlabel,dtype=np.int)

