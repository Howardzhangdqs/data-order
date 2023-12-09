import os


class DefaultConfig():

    def __init__(
        self,
        dataPath="../data",
        modelPath="../model/basemodel/model.pth",
        logPath="../log",
        batchSize=64,
        lr=0.001,
        epoch=10,
        numWorkers=0,
        device="cuda:0"
    ):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.logPath = logPath
        self.batchSize = batchSize
        self.lr = lr
        self.epoch = epoch
        self.numWorkers = numWorkers
        self.device = device
