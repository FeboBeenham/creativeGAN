import shutil
import os
import tensorflow as tf
import glob
import serial
from tensorflow import keras

folder_files = os.listdir("./pr_wikiart/")  # You can also use full path
dst_dir = "./working_imgs"
data_dir = "./working_imgs_copy/"
lightt = []
humm = []
tempp = []
ccl = []
cch = []
cct = []

SIZE_working_imgs = 20  # how many files do we want in the memory

def image_select():
    dst_dir_copy = "./working_imgs_copy"
    file_exists = []

    for file in os.listdir(dst_dir_copy):
        file_exists.append(file)

    for file in range(len(file_exists)):
        os.remove("./working_imgs_copy/{}".format(file_exists[file]))

    for i in range(no_of_files):
        random_folder = random.choice(folder_files)
        selected_folder = os.listdir("./pr_wikiart/{}".format(random_folder))
        random_file = random.choice(selected_folder)
        while random_file in file_exists:
            random_file = random.choice(selected_folder)
            print(random_file)
        file_exists.append(random_file)
        selected_file = "./pr_wikiart/{}/{}".format(random_folder, random_file)
        shutil.copy(selected_file, dst_dir_copy)

def image_renew():
    dst_dir_copy = "./working_imgs_copy"

    #   add an image
    # create function that translates the iot input into a database output
    ser = serial.Serial('COM4', 9600)
    ser.close()
    ser.open()
    while True:
        data = ser.readline()
        print(data.decode())
        datastr = data.decode()
        light = datastr.split(",")[0]
        light = int(float(light))
        lightt.append(light)
        humidity = datastr.split(",")[1]
        humidity = int(float(humidity))
        humm.append(humidity)
        temperature = datastr.split(",")[2]
        temperature = int(float(temperature))
        tempp.append(temperature)
        break

    if 1 == len(lightt):
        selected_file = "./pr_wikiart/Cubism/{}.jpg".format(light)
        shutil.copy(selected_file, dst_dir_copy)
        print("ADDED new cubism:", light, ".jpg")
    elif (lightt[-2] >= lightt[-1]+5 or lightt[-2] <= lightt[-1]-5):
        selected_file = "./pr_wikiart/Cubism/{}.jpg".format(light)
        shutil.copy(selected_file, dst_dir_copy)
        print("ADDED new cubism:", light, ".jpg")
        ccl.append(light)
    else:
        print(lightt)
        lightt.pop()
        print(lightt)
        selected_file = "./pr_wikiart/Cubism/{}.jpg".format(lightt[-1])
        shutil.copy(selected_file, dst_dir_copy)
        print("ADDED old cubism:", lightt[-1], ".jpg")

    if 1 == len(humm):
        k = 1
    elif (humm[-2] >= humm[-1]+1 or humm[-2] <= humm[-1]-1):
        cch.append(humm)
    selected_file = "./pr_wikiart/Abstract_Expressionism/{}.jpg".format(humidity)
    shutil.copy(selected_file, dst_dir_copy)
    print("ADDED AE:", humidity, ".jpg")

    if 1 == len(tempp):
        k = 1
    elif (tempp[-2] >= tempp[-1]+1 or tempp[-2] <= tempp[-1]-1):
        cct.append(tempp)
    selected_file = "./pr_wikiart/Fauvism/{}.jpg".format(temperature)
    shutil.copy(selected_file, dst_dir_copy)
    print("ADDED Fauvism:", temperature, ".jpg")
    print("Temp change:", len(cct), ". Hum change:", len(cch), ". Light change:", len(ccl))


    nr_exists = []
    for file in os.listdir(dst_dir_copy):
        nr_exists.append(file)

    # delete oldest image
    for i in range(SIZE_working_imgs, len(nr_exists)):
        os.chdir(dst_dir_copy)
        files = [file for file in os.listdir('.') if (file.lower().endswith('.jpg'))]
        files.sort(key=os.path.getmtime)
        print('DELETED:', files[0])
        os.remove(files[0])
        os.chdir(os.path.dirname(os.getcwd()))

    train_images = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode=None, image_size=(128, 128), batch_size=128)

    # Normalize the images to [-1, 1] which is the range of the tanh activation
    train_images = train_images.map(lambda x: (x - 127.5) / 127.5)

    return train_images
