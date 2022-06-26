import streamlit as st
import torch
import timm
from torch import nn
import torchvision
from torchvision import transforms,models
from torch.utils.data import Dataset
from PIL import Image
import torch_optimizer as optim
import torchmetrics
import pytorch_lightning as pl
import os
import pandas as pd
import numpy as np
import time 
from tqdm import tqdm
from pillow_heif import register_heif_opener
import torch_optimizer as optim
import torchmetrics
import io
import ruclip
import exif
from flash import Trainer
from flash.image import ImageClassifier, ImageClassificationData
from st_aggrid import AgGrid
import pandas as pd


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        print(uploaded_file.name)
        with open(uploaded_file.name, "wb") as f:
            f.write(io.BytesIO(image_data).getbuffer())
        st.image(image_data)
        return Image.open(io.BytesIO(image_data)), exif.Image(io.BytesIO(image_data)),uploaded_file.name
    else:
        return None, None,None

def load_model():
    #model = LizaNet().load_from_checkpoint('lizanet_b0.ckpt')
    #model.eval()
    return None

def load_clip():
    device = 'cpu'
    clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
    return clip, processor

def get_clip_predictions(clip,processor, search_string,image):
    classes = [(search_string), 'точно не ' + (search_string)]
    templates = ['{}', 'это {}', 'на картинке изображена {}']
    predictor = ruclip.Predictor(clip, processor, 'cpu', bs=8, templates=templates)
    with torch.no_grad():
        text_latents = predictor.get_text_latents(classes)
        pred_labels = predictor.run([image], text_latents)
        print(pred_labels)
    return 1 - pred_labels[0]


def predict(model, image):
    model = ImageClassifier.load_from_checkpoint("image_classification_model_resnet18.pt")
    trainer = Trainer()
    datamodule = ImageClassificationData.from_files(
    predict_files=[image],
    batch_size=1)
    print(trainer.predict(model, datamodule=datamodule, output="labels"))
    #for o in outputs.keys():
    #    st.write(outputs[o].detach().numpy())
    predictions = trainer.predict(model, datamodule=datamodule, output="preds")

    answers = []
    for pred in predictions:
        probs = (pred[0].numpy())
        time_of_day = np.argmax(probs[:4])
        season = np.argmax(probs[4:9])
        area = np.argmax(probs[9:12])
        avia = np.argmax(probs[12:14])
        auto = np.argmax(probs[14:16])
        drone = np.argmax(probs[16:18])
        scuba = np.argmax(probs[18:20])
        dog = np.argmax(probs[20:22])
        horse = np.argmax(probs[22:24])
        hugs = np.argmax(probs[24:26])
        sherp = np.argmax(probs[26:28])
        answers.append([image,time_of_day,season,area,avia,auto,drone,scuba,dog,horse,hugs,sherp])
    print(answers)
    return answers

def main():
    st.title('Поиск тэгов')
    title = st.text_input('Дополнительные тэги', 'Забор&Мужчина&Экскаватор')

    image, exifs, name = load_image()
    model = load_model()   
    clip, processor = load_clip()
    result = st.button('Run on image')
    if result:
        
        st.write('Calculating results...')
        first_tags = predict(model, name)
        df_tags = pd.DataFrame(first_tags,columns = ["Файл",'Время Дня', 'Сезон', 'Местность',"Авиа","Авто","Дрон","Водолаз","Кинолог","Лошадь","Обьятия","Шерп"])
        tags = title.split('&')
        tag_results = dict()
        for tag in tags:
            tag_results[tag] = [get_clip_predictions(clip,processor,tag,image)]
        if exifs.list_all():
            tag_results['latitude'] = [exifs.gps_latitude]
            tag_results['longitude'] = [exifs.gps_longitude]
            tag_results['time'] = [exifs.datetime]
            tag_results['model'] = [exifs.model]
        AgGrid(pd.concat([df_tags, pd.DataFrame(tag_results)], axis=1),editable=True)
        



if __name__ == '__main__':
    main()
