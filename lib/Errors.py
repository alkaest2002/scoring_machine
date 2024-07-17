from pathlib import Path

class ValidationError(Exception):
  pass

class NotFoundError(Exception):
  pass

class TracebackNotifier():

    def __init__(self, error):
        # store error
        self.error = error

    def notify_traceback(self):
        try:
            # get error's traceback
            traceback = self.error.__traceback__
            # consume traceback
            while traceback is not None:
                # notify current traceback step
                print(
                    "-->",
                    Path(traceback.tb_frame.f_code.co_filename),
                    traceback.tb_frame.f_code.co_name,
                    "line code",
                    traceback.tb_lineno
                , end="\n")
                # get next traceback step
                traceback = traceback.tb_next
        # on error
        except Exception as e:
            # store error
            self.error = e
            # notify traceback
            self.notify_traceback()
