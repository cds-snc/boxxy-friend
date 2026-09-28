import os

def log(message, clear_screen=False):
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    print(message)