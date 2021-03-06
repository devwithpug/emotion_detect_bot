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
from PIL import Image, ImageDraw, ImageFont
import sys, io
#sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')

font = ImageFont.truetype('Bazzi.ttf', 30)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

token_path = os.path.dirname(os.path.abspath(__file__)) + "/.token"
with open(token_path, "r", encoding="utf-8") as t:
    token = t.readline()
print(token)

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


# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {
    0: "화난",
    1: "구역질나는",
    2: "무서운",
    3: "행복한",
    4: "무표정",
    5: "슬픈",
    6: "놀란",
}

facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



def emotion_detect(url):
    # frame = cv2.imread('/content/drive/MyDrive/Colab Notebooks/facial_emotion_recog/오.png')
    raw = requests.get(url).content
    nparr = np.asarray(bytearray(raw), dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        
        print("DEBUG: {}".format(maxindex))
        text_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(text_img)
        
        draw.text((x+5, y-30), emotion_dict[maxindex], font=font, fill=(255, 255, 255))
        
        frame = np.array(text_img)
        # cv2.putText(
        #     frame,
        #     emotion_dict[maxindex],
        #     (x + 20, y - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
    try:
        print("result: ", emotion_dict[maxindex])
    except Exception as err:
        print(err)
        return

    return frame


@bot.event
async def on_ready():
    print("BOT IS READY NOW!")


@bot.command()
async def 표정(ctx, *args):
    if not args:
        await ctx.send("`~표정 (이미지 주소)`")
        return
    try:
        raw = requests.get(args[0]).content
    except Exception as err:
        await ctx.send("이미지 디코딩 실패: {}".format(err))
        return
    after_image = emotion_detect(args[0])
    
    try:
        is_success, buffer = cv2.imencode(".png", after_image)
    except Exception as err:
        await ctx.send("`얼굴을 인식할 수 없습니다.`")
        return
    
    io_buf = BytesIO(buffer.tobytes())
    try:
        await ctx.send(file=discord.File(fp=io_buf, filename="image.png"))
    except Exception:
        await ctx.send('`이미지 크기가 너무 큽니다.`')
        return


bot.run(token)
