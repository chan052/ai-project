import os
import shutil

if not os.path.exists("PDF"):
    os.mkdir("PDF")

if not os.path.exists("JPG"):
    os.mkdir("JPG")

if not os.path.exists("PNG"):
    os.mkdir("PNG")

if not os.path.exists("MP3"):
    os.mkdir("MP3")

if not os.path.exists("MP4"):
    os.mkdir("MP4")

files = os.listdir()

for file in files:
    if file.endswith(".pdf"):
        shutil.move(file, "PDF/" + file)
    elif file.endswith(".jpg"):
        shutil.move(file, "JPG/" + file)
    elif file.endswith(".png"):
        shutil.move(file, "PNG/" + file)
    elif file.endswith(".mp3"):
        shutil.move(file, "MP3/" + file)
    elif file.endswith(".mp4"):
        shutil.move(file, "MP4/" + file)

print("완료!")