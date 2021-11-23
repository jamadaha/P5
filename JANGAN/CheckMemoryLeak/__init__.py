import tracemalloc
import time
import os
from os import path, write

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
        time_string = time.strftime("%m-%d-%Y, %H:%M:%S", named_tuple)

        self.TracemallocFile = '/' + time_string + ' ' + fileName

        file = open(self.TracemallocPath + self.TracemallocFile,"x")
        file.close()
    
    def SaveSnapshot(self, key, snapshot):
        self.Snapshot[key] = snapshot
        self.Top_stats = self.Snapshot[key].statistics('lineno')

    def WriteToFile(self, key, traced_memory):
        file = open(self.TracemallocPath + self.TracemallocFile, 'a')
        file.write(f"[Writing tracemalloc for {key}]\n")

        for stat in self.Top_stats[:25]:
            text = str(stat)
            file.writelines(text)
            file.write('\n')

        file.write('\nTraced memory is (current, peak):' + str(traced_memory) + '\n\n')
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
