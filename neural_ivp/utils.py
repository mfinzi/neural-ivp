import time

class LogTimer(object):
    """ Timer to automatically control time spent on expensive logs
                    by not logging when computations performed in __enter__
                    exceed the specified fraction of total time outside.
        """
    def __init__(self, minPeriod=1, timeFrac=1 / 10, **kwargs):
        """ minPeriod: minimum time between logs.
                timeFrac: max fraction of total time spent inside __enter__ block."""
        self.avgLogTime = 0
        self.numLogs = 0
        self.lastLogTime = 0
        self.minPeriod = minPeriod  #(measured in minutes)
        self.timeFrac = timeFrac
        self.performedLog = False
        super().__init__(**kwargs)

    def __enter__(self):
        """ returns yes iff the number of minutes have elapsed > minPeriod 
                and  > (1/timeFrac) * average time it takes to log """
        timeSinceLog = time.time() - self.lastLogTime
        self.performedLog = (timeSinceLog > 60*self.minPeriod) \
                            and (timeSinceLog > self.avgLogTime/self.timeFrac)
        if self.performedLog: self.lastLogTime = time.time()
        return self.performedLog

    def __exit__(self, *args):
        if self.performedLog:
            timeSpentLogging = time.time() - self.lastLogTime
            n = self.numLogs
            self.avgLogTime = timeSpentLogging / (n + 1) + self.avgLogTime * n / (n + 1)
            self.numLogs += 1
            self.lastLogTime = time.time()
