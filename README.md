# nicolasleao/python-chatbot

This project contains training scripts for machine learning models that are able to classify a sentence into a list of conversation intents that can be defined in `src/training/intents.json`, and an application loop to interact with the model in `src/chatbot.py`

## Technologies
Here, I used `nltk` and `numpy` to prepare the traning data, and `tensorflow` to create and train the models.
In future iterations, I plan on integrating with ChatGPT to generate more human responses after classifying the text.

## Setup
To test this project, you'll need to install the pip dependencies using
`pip install -r requirements.txt`

In this repo, i've already included a trained model using demonstrative real estate sales conversation intents in portuguese, but if you want to use your own intents
you need to update `src/training/intents.json` and tailor it to your specific needs, then run the training script.
`cd src/training && python train_sequential_network.py`

This will generate three files: `words.pkl`, `tags.pkl` and `chatbot_model.h5` that will be automatically loaded when you run the program.
You only need to train your model when you change the `intents.json` file.

## Usage
To interact with the chatbot, navigate to the `/src` folder and run:
`python chatbot.py`

## Demo

### Training:
![training gif demo](https://github.com/nicolasleao/python-chatbot/blob/main/demo/training.gif?raw=true)

### Chatting:
![chatting gif demo](https://github.com/nicolasleao/python-chatbot/blob/main/demo/chatting.gif?raw=true)
