import os
import re

def setup_log_files(log_path):
    assert os.path.isdir(log_path)
    # first remove all log files
    for f in os.listdir(log_path):
        if re.search("*.log", f):
            os.remove(os.path.join(log_path, f))

def record(log_path, name, measure, loss, step):
    with open("%s/%s_%s.log" % (log_path, name, measure), "a") as f:
        f.write("%s\t%.4f\n" % (step, loss))
