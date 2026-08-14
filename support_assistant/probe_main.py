import os
import sys

print('START')
print('CWD', os.getcwd())
sys.path.insert(0, '.')
print('BEFORE_IMPORT')
import app.main
print('AFTER_IMPORT')
print('TITLE', app.main.app.title)
print('END')
