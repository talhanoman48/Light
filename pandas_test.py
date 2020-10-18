import pandas as pd
import numpy as np
import streamlit as st
import json

with open("Datasets/CS.json", "r") as f:
    data = json.load(f)

train = {}

for intent in data['intents']:
    train.update({"tag": intent["tag"], "patterns": intent["patterns"]})

train