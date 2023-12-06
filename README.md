# Light: An Educational Chatbot for Computer Science Students
Light is a chatbot that aims to help computer science students in need of assistance with their coursework, assignments, or projects. Light can answer questions related to various topics in computer science, such as data structures, algorithms, programming languages, databases, etc.

## Features
Light uses a CNN-based architecture to classify user intent and respond to the queries. The architecture is defined in the *main.py* file.
Light is trained on a large dataset of computer science questions and answers collected from various sources, such as Stack Overflow, Quora, Reddit, etc.
Light can handle multiple types of questions, such as factual, conceptual, procedural, or opinion-based.
Light can also evaluate and critique the students’ code snippets and provide constructive feedback and best practices.
## Installation
To use Light, you need to install the following dependencies:
<ul>
  <li>Python 3.7 or higher</li>
  <li>NLTK 3.6 or higher</li>
  <li>NumPy 1.19 or higher</li>
  <li>Pandas 1.2 or higher</li>
</ul>

You can install them by running the following command:

```
pip install -r requirements.txt
```

## Usage
To run Light, you need to download the pre-trained model and tokenizer files from this repository. You can also download the datasets and dictionaries used for training and testing the model.

To start a conversation with Light, you can run the following command:
```
streamlit run main.py
```
You can then type your questions in the web interface and press enter to get a response from Light. To quit, you can simply select the stop program from the burger menu.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
