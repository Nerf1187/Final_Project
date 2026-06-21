import os


def clear_checkpoints():
    path = "../models/Faster_RCNN/checkpoints/"
    for file in os.listdir(path):
        os.remove(path + file)
        
if __name__ == "__main__":
    clear_checkpoints()