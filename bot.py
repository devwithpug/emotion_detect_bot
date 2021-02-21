import discord, os, requests
from discord.ext import commands, tasks
from io import BytesIO
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.chdir(os.path.dirname(os.path.abspath(__file__)))

token = "NzUxMzAyODU5MTE0NTQ1MjAz.X1HHUA.qvUHx_VTIQAbopj-bq6APUhTANU"

bot_activity = discord.Game(name="test")
bot = commands.Bot(command_prefix="~", activity=bot_activity)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

mode = "display"

model.load_weights("model.h5")
print("model loaded")

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("haarcascade loaded")


def emotion_detect(url):
    # frame = cv2.imread('/content/drive/MyDrive/Colab Notebooks/facial_emotion_recog/오.png')
    raw = requests.get(url).content
    nparr = np.asarray(bytearray(raw), dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("1")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print("2")
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        cv2.putText(
            frame,
            emotion_dict[maxindex],
            (x + 20, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    print("3")
    return frame


@bot.event
async def on_ready():
    print("봇이 시작되었습니다.")


@bot.command()
async def 표정(ctx, *args):
    try:
        raw = requests.get(args[0]).content
    except Exception as err:
        await ctx.send("이미지 디코딩 실패: " + err)
        return
    after_image = emotion_detect(args[0])
    try:
        is_success, buffer = cv2.imencode(".png", after_image)
    except Exception as err:
        await ctx.send("이미지 인코딩 실패: " + err)
        return
    io_buf = BytesIO(buffer.tobytes())

    await ctx.send(file=discord.File(fp=io_buf, filename="image.png"))


bot.run(token)
