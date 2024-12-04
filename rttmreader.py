from dataclasses import dataclass
from pathlib import Path
import os
import uuid

@dataclass
class rttm_line:
    speaker: str 
    begin: float
    end: float
    duration: float
    text: str
    channel: int
    uuid: str

class rttmreader:

    def __init__(self,filename):
        self.intputfile = {'file': '', 'name': ''}
        self.rttms = []
        self.__read_rttm(filename)
    
    def __read_rttm(self,filename):
        print('Loading  %s' % filename)

        if not os.path.exists(filename):
            print('Unable to open %s' % filename)
            return 0
        try:
            self.intputfile["file"] = filename
            self.intputfile["name"] = Path(filename).resolve().stem
            with open(filename) as input_file:
                self.clear()
                for line in input_file:
                    a = self.__parse_rttm_line(line)
                    self.rttms.append(a)
            print('Done. Loaded',len(self.rttms),'lines')
        except IOError as e:
            print('Invalid input file: %s. %s' % (filename, e))
            return 0
    
    def filter_rttms(self, start_time, end_time):
        return [rttm_line for rttm_line in self.rttms if rttm_line.end >= start_time and rttm_line.begin <= end_time]

    def get_rttms(self):
        return self.rttms
    
    def get_rttms_sorted_by_time(self):
        return sorted(self.rttms, key=lambda x: x.begin)
    
    def sort_rttms_sorted_by_speaker_and_time(self):
        return sorted(self.rttms, key=lambda x: (x.speaker,x.begin))
    
    def __parse_rttm_line(self,line):
        fields = line.split()
        if len(fields) != 10:
            raise IOError('Number of fields != 10. LINE: "%s"' % line)
        channel = int(fields[2])
        begin = float(fields[3])
        duration = float(fields[4])
        end = round(begin+duration,3)
        speaker = fields[7]
        return rttm_line(speaker, begin, end, duration, '', channel, str(uuid.uuid4()))
    
    def clear(self):
        self.rttms = []
    
    def print(self):
        print('Input:',self.intputfile)
        print('Length:',len(self.rttms))
        for an in self.rttms:
            print(an)

    def get_max_time(self):
        return sorted(self.rttms, key=lambda x: x.end)[-1].end
    

        