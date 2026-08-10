import sys
import os
sys.path.insert(0, os.path.abspath(r"env\Lib\site-packages"))
import allin1
print(dir(allin1))
try:
    help(allin1.analyze)
except:
    pass
