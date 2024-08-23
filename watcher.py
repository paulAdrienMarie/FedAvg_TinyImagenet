import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, command):
        self.command = command
        self.process = None
        self.restart()

    def on_any_event(self, event):
        if event.src_path.endswith((".py", ".html", ".js")):
            self.restart()

    def restart(self):
        if self.process:
            print("Stopping process...")
            self.process.terminate()  # Arrêter le processus en cours
            self.process.wait()  # Attendre que le processus soit complètement arrêté
        print("Starting process...")
        self.process = subprocess.Popen(self.command, shell=True)


if __name__ == "__main__":
    path = "./"
    command = f"{sys.executable} main.py"

    event_handler = ChangeHandler(command)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
