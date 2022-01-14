import random
import json
from typing import List

import torch
from model import NeuralNet
from nltkUtils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "CMSC BOT"
user_name = None

# One-word responses that would confuse the natural learning process
Exit_inputs = (
    "goodbye", "bye bye", "dueces", "quit", "end", "cya", "exit", "see you later", "bye", "goodnight", "good night",
    "stop")
feeling_inputs = ("awesome", "cool", "great", "good", "perfect", "amazing", "spectacular", "ok", "okay", "that is good",
                  "thats good", "thats perfect", "epic", "coolio", "word", "nice")

#TOGGLES for appropriate response to msg
user_name_toggle = False
run1 = False
math = False
cs = False
pre_cs = False

# function for when the bot does not understand the request
def misInput(sentence):
    with open("misinputData.txt", "a") as f:
        f.write("\n" + str(sentence))

    return "I’m sorry. I didn’t understand, how about asking me something about math or computer science courses?"


def get_response(msg):
    #Toggles for appropriate responses
    global user_name_toggle
    global user_name
    global run1
    global math
    global cs
    global pre_cs

    #Getting msg ready for processing
    msg = msg.lower()
    punc = " ' ' ! ( ) - [ ] { } ; : '  \ , < > . / ? @ # $ % ^ & * _ ~ ' ' "
    punc = punc.split()
    msg = "".join(t for t in msg if t not in punc)

    #Run1 - if the user needs help with any scheduling purposes
    if run1 is True and msg == 'yes':
        math = True
        run1 = False
        return "Zoobi Doobi Cool!\n Have you taken any college math courses(AP exemption allowed)? 'Yes' or 'No'"
    elif run1 is True and msg == 'no':
        run1 = False
        pre_cs = True
        return "Do you need help scheduling computer science courses? 'Yes' or 'No'"

    #math run - if the user needs help with math scheduling
    if math is True and msg == 'yes':
        math = False
        return "Zoobi Doobi awesome! what was the most recent math course(s)? Specify by class code"
    elif math is True and msg == 'no':
        math = False
        return "Take Math 140 as your first math course. Math 140 is calculus 1.Keep in mind that there is an " \
               "exemption exam for this course if you already know the material. "

    #Pre-cursor question for students who might need help with CS
    if pre_cs is True and msg == 'yes':
        pre_cs = False
        cs = True
        return "Zoobi Doobi Cool!\n Have you taken any college computer courses(AP exemption allowed)? 'Yes' or 'No"
    elif pre_cs is True and msg == 'no':
        pre_cs = False
        return "I am just a CMSC bot sadly, I can't help you with any other issues. Maybe go to " \
               "https://undergrad.cs.umd.edu/degree-requirements-cs-major for more information "

    #CS run - students who need help with computer science
    if cs is True and msg == 'yes':
        cs = False
        return "Zoobi Doobi awesome! what was the most recent CS course(s)? Specify by class code"
    elif cs is True and msg == 'no':
        cs = False
        return "Take CMSC 131 or 133 as your first computer course. "\
                    "\n CMSC 131 is object oriented programming 1, and 133 is the advanced verson."\
                    "Keep in mind that there is an exemption exam for this course if you already know"\
                    "the material."

    wrong_answer = ("Hate to break it to you, but I kinda need a 'yes' or a 'no'", "Could you say 'yes' or 'no' for me?",
                    "I need a 'yes' or a 'no'", "Input 'yes' or 'no'")
    if cs is True or pre_cs is True or math is True or run1 is True and (msg != "no" and msg != "yes"):
        return random.choice(wrong_answer)

    #One-word responses that won't work with intents.json
    if msg == 'thanks' or msg == 'thank you' or msg == 'thank':
        return " You're welcome!"
    elif msg == "quit" or msg in Exit_inputs:
        return "Goodbye! :)"
    elif msg in feeling_inputs:
        return "Smiley Face! Please chat with me more, I like you!"
    else:
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        #If the prediction is more than 90% accurate
        if prob.item() > 0.90:
            #Primary check to see if user is asking about Username
            if tag == "user_name":
                if not user_name_toggle and user_name is None:
                    user_name_toggle = True
                    return random.choice(intents['intents'][2]['responses'])
                elif user_name is not None:
                    return f"I know your name. You are {user_name}"
            else:
                for intent in intents["intents"]:
                    if tag == intent["tag"]:
                        #Any questions regarding scheduling in general
                        if random.choice(intent['responses']) == "Agent Code1":
                            run1 = True
                            return "Epic!!! Do you need help scheduling Math courses? 'Yes' or 'No'"
                        #Any questions regarding Computer scheduling
                        if random.choice(intent['responses']) == "Agent Code2":
                            cs = True
                            return "Zoobi Doobi Cool!\n Have you taken any college CS courses(AP exemption allowed)? " \
                                   "'Yes' or 'No'"
                        #Any questions regarding Math Scheduling
                        if random.choice(intent['responses']) == "Agent Code3":
                            math = True
                            return "Zoobi Doobi Cool!\n Have you taken any college Math courses(AP exemption " \
                                   "allowed)? 'Yes' or 'No'"

                        #Other Miscellenious responses
                        return random.choice(intent['responses'])
        else:
            #Fixes any user_name issues
            if user_name_toggle is True:
                user_name = msg
                user_name_toggle = False
                return "Cool name!"
            else:
                # checks if the user inputs an upper level class
                new_string = ""
                for s in str(sentence):
                    if s.isdigit():
                        new_string = new_string + s
                if new_string != "" and 300 < int(new_string) < 500:
                    return "I think your question refers to an upper level CMSC course. \nConsult " \
                           "https://undergrad.cs.umd.edu/degree-requirements-cs-major for more information "

                # If there is no accurate prediction, the response is written to another text file to be recorded
                else:
                    return misInput(sentence)
