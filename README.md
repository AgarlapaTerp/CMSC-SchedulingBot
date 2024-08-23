# CMSC Scheduling Bot

CMSC Scheduling Bot is a Python-based chatbot designed to assist University of Maryland (Terps) students with their scheduling needs. The bot utilizes natural language processing and deep learning techniques to streamline the scheduling process.

## File Overview

- **CMSC-Bot.py**: Main file containing the GUI and chatbot functionality. Running this file initializes the entire chatbot system.
- **misinput.txt**: Contains examples of messages that the chatbot may not understand.
- **intents.json**: Defines the blueprint of responses that guide the chatbot's interactions.
- **train.py**: Responsible for training the chatbot's deep learning model.
- **chatBotResponse.py**: Handles the generation of responses based on user input.
- **nltkUtils.py**: Provides utility functions used in `chatBotResponse.py`.

## Setup and Usage

1. **Install Dependencies**: Ensure that `nltk`, `tkinter`, `model`, `torch`, and `numpy` are installed in your Python environment (e.g., PyCharm).

2. **Run Initialization Files**: Execute all the Python files to load functions and perform necessary training.

3. **Start the Chatbot**: Run `CMSC-Bot.py` to launch the chatbot and enjoy its functionalities.

   **Important**: Make sure to run `train.py` before `CMSC-Bot.py` to train the chatbot. The console will indicate when the training is complete.

## Examples

Here are some screenshots of the chatbot in action:

![Chatbot Example 1](https://user-images.githubusercontent.com/97764660/230502794-2c7a994f-b3c2-4b21-9319-5020b2e5be73.png)
![Chatbot Example 2](https://user-images.githubusercontent.com/97764660/230502807-a17b7945-800d-4d3e-92a9-3762299b3b42.png)
![Chatbot Example 3](https://user-images.githubusercontent.com/97764660/230502815-8a7230ef-00d4-46c9-81a0-65b261516990.png)
