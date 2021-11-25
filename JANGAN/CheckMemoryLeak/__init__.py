from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tracemalloc")

import tracemalloc
import time
import os
from os import path, write

# A little how-to-use guide:
## 1. Create an instance of this class somewhere
## 2. Call the ConfigurePath(...) method, to setup the paths for log files
## 3. Call the StartChecker(...) method when you are ready to start tracing
## 4. Call the SaveKey(...) method before taking snapshots, so that it is possible to save snapshots to a certain key
## 5. Call the SaveSnapshot(...) method when you wanna take a memory snapshot to a certain key
## 6. Call the WriteToFile(...) sometimes, when you want to save the log file
## 7. Finally you can call the CompareSnapshots(...) method, to get a comparison for all the saved snapshots

class CheckMemoryLeak():
    TraceMallocPath = ""
    TraceMallocFile = ""

    Snapshot = {}
    SnapshotKeys = []
    Top_stats = any

    def ConfigurePath(self, path, fileName):
        self.TracemallocPath = path

        if not os.path.isdir(self.TracemallocPath):
            os.makedirs(self.TracemallocPath)

        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m-%d-%Y %H-%M-%S", named_tuple)

        self.TracemallocFile = '/' + time_string + ' ' + fileName

        file = open(self.TracemallocPath + self.TracemallocFile,"x")
        file.close()

    def StartChecker(self):
        tracemalloc.start()
    
    def SaveSnapshot(self, key):
        self.Snapshot[key] = tracemalloc.take_snapshot()
        self.Top_stats = self.Snapshot[key].statistics('lineno')

    def WriteToFile(self, key):
        file = open(self.TracemallocPath + self.TracemallocFile, 'a')
        file.write(f"[Writing tracemalloc for {key}]\n")

        for stat in self.Top_stats[:25]:
            text = str(stat)
            file.writelines(text)
            file.write('\n')

        file.write('\nTraced memory is (current, peak):' + str(tracemalloc.get_traced_memory()) + '\n\n')
        file.close()

    def CompareSnapshots(self):
        if (len(self.SnapshotKeys) > 1):
            for num in range(len(self.SnapshotKeys) - 1):
                if (self.SnapshotKeys[num + 1] != None):
                    snapshot1 = self.Snapshot[self.SnapshotKeys[num]]
                    snapshot2 = self.Snapshot[self.SnapshotKeys[num + 1]]

                    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

                    tm = open(self.TracemallocPath + self.TracemallocFile, 'a')
                    tm.write(f'[ Top 10 differences between {self.SnapshotKeys[num]} and {self.SnapshotKeys[num + 1]}]\n')
                    for stat in top_stats[:10]:
                        tm.write(str(stat))
                        tm.write('\n')
                    tm.write('\n')
                    tm.close()
    
    def SaveKey(self, savekey):
        self.SnapshotKeys.append(savekey)
