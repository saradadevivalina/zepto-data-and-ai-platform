import os, sys, traceback
print('START')
print('CWD', os.getcwd())
sys.path.insert(0, '.')
try:
    import app.main
    print('IMPORT_OK')
    print(app.main.app.title)
except Exception as e:
    print('EXCEPTION', type(e).__name__, e)
    traceback.print_exc()
    raise
