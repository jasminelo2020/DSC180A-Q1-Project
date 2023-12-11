import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QStackedLayout, QDialog, QFormLayout, QLineEdit
import pandas as pd
import numpy as np
import torch
from PIL.ImageQt import ImageQt




import clip
import utils
import data_utils
import similarity

from matplotlib import pyplot as plt


#GUI Main Code

class InputDialog(QDialog):
    def __init__(self, parent=None):
        super(InputDialog, self).__init__(parent)
        self.setFixedSize(200, 100)  # Set the fixed size of the input dialog
        self.setWindowTitle("Input Text")

        layout = QFormLayout(self)

        self.text_edit = QLineEdit()
        layout.addRow("Enter text:", self.text_edit)

        self.text_edit.returnPressed.connect(self.accept)  # Accept when Enter is pressed

class MainWindow(QMainWindow):
    def __init__(self, image_sets, clip_label, neuron,neuron_count,title):
        super(MainWindow, self).__init__()
        self.title = title
        self.setWindowTitle(self.title)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        self.label_layout = QHBoxLayout()
        self.image_layout = QHBoxLayout()

        self.image_sets = image_sets
        self.clip_label = clip_label
        self.neuron = neuron
        self.neuron_count = neuron_count
        self.image_labels = []
        self.match = []
        self.wrong = []
        self.similar = []
        self.inputted_texts = []
        self.return_df = pd.DataFrame()

        self.current_set_index = 0  # Index to track the current set of images
        self.current_label_index = 0  # Index to track the current label within the set

        self.show_next_image_set()
        layout.addLayout(self.label_layout)
        layout.addLayout(self.image_layout)

        # Create a stacked layout to switch between buttons and text input fields
        self.stacked_layout = QStackedLayout()

        button_layout = self.create_button_layout()

        self.similar_text_edit = InputDialog(self)
        self.wrong_text_edit = InputDialog(self)

        self.similar_text_edit.accepted.connect(self.append_similar_text)
        self.wrong_text_edit.accepted.connect(self.append_wrong_text)

        # Wrap the button layout in a QWidget before adding it to the stacked layout
        button_wrapper = QWidget()
        button_wrapper.setLayout(button_layout)

        self.stacked_layout.addWidget(button_wrapper)
        self.stacked_layout.addWidget(self.similar_text_edit)
        self.stacked_layout.addWidget(self.wrong_text_edit)

        layout.addLayout(self.stacked_layout)

        self.similar_texts = []  # To store similar texts
        self.wrong_texts = []  # To store wrong texts

    

    def create_button_layout(self):
        button_layout = QHBoxLayout()

        match_button = QPushButton("Match")
        similar_button = QPushButton("Similar")
        wrong_button = QPushButton("Wrong")

        button_layout.addWidget(match_button)
        button_layout.addWidget(similar_button)
        button_layout.addWidget(wrong_button)

        similar_button.clicked.connect(self.similar_clicked)
        wrong_button.clicked.connect(self.wrong_clicked)
        match_button.clicked.connect(self.match_clicked)  # Show next set on Match button press

        return button_layout


    def match_clicked(self):
        self.match.append(1)
        self.wrong.append(0)
        self.similar.append(0)
        self.inputted_texts.append("None")
        if self.current_set_index < len(self.image_sets):
            self.show_next_image_set()
        else:
            # All sets have been processed
            # Handle completion or exit here
            print("All sets have been processed.")
            # print(self.neuron)
            # print(self.clip_label)
            # print(self.match)
            # print(self.wrong)
            # print(self.similar)
            self.return_df = pd.DataFrame({"neuron":self.neuron,"CLIP dissect label":self.clip_label,"match":self.match,"wrong":self.wrong,"similar":self.similar,"user label":self.inputted_texts})
            self.return_df.to_csv("neuron_labels/{}_{}_{}_{}_{}.csv".format(target_name,target_layer,d_probe,concept_set_str,similarity_fn.__name__))
            sys.exit(app.exec())

    def similar_clicked(self):
        self.match.append(0)
        self.wrong.append(0)
        self.similar.append(1)
        self.show_similar_input()

    def wrong_clicked(self):
        self.match.append(0)
        self.wrong.append(1)
        self.similar.append(0)
        self.show_wrong_input()
        

    
    def show_similar_input(self):
        self.stacked_layout.setCurrentIndex(1)  # Show the similar text input

    def show_wrong_input(self):
        self.stacked_layout.setCurrentIndex(2)  # Show the wrong text input

    def show_next_image_set(self):
        if self.current_set_index < len(self.image_sets):
            # Clear previous images and labels
            for labelx in self.image_labels:
                labelx.clear()
                labelx.deleteLater()
            while self.label_layout.count():
                item = self.label_layout.takeAt(0)  # Take the first item (widget) in the layout
                if item.widget():
                    item.widget().deleteLater()  # Delete the widget if it exists
            
    
            self.image_labels = []  # Clear the list of labels
    
            # Show the next set of 5 images
            
            image_set = self.image_sets[self.current_set_index]
    
            # Create labels for ground truth and neuron information
            clip_label_label = QLabel(f"CLIP dissect label: {self.clip_label[self.current_set_index]}")
            neuron_label = QLabel(f"Neuron Number: {self.neuron[self.current_set_index]} / {self.neuron_count-1}")
    
            for image_arr in image_set:
                height, width, channel = image_arr.shape
                bytesPerLine = 3 * width
                qImg = QImage(image_arr.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
                labelnew = QLabel()
                #q_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, pil_image.width * 3)
                pixmap = QPixmap.fromImage(qImg)
                # pixmap = pixmap.scaled(128, 128)  # Resize the images to 128x128
                labelnew.setPixmap(pixmap)
                self.image_layout.addWidget(labelnew)
                self.image_labels.append(labelnew)
                
            
            self.label_layout.addWidget(clip_label_label)
            self.label_layout.addWidget(neuron_label)
    
            self.current_set_index += 1
            self.current_label_index = 0

    def append_similar_text(self):
        text = self.similar_text_edit.text_edit.text()
        if text:
            self.inputted_texts.append(text)
            # Process similar text as needed
    
            self.similar_texts.append(text)  # Store the similar text
            self.similar_text_edit.text_edit.clear()
            self.stacked_layout.setCurrentIndex(0)  # Switch back to buttons layout
    
            self.current_label_index += 1
    
            if self.current_label_index == len(self.image_labels):
                # All images in the current set have been processed
                self.current_label_index = 0  # Reset the current label index
                self.current_set_index += 1  # Move to the next set
    
            if self.current_set_index < len(self.image_sets):
                self.show_next_image_set()
            else:
                # All sets have been processed
                # Handle completion or exit here
                print("All sets have been processed.")
                # print(self.neuron)
                # print(self.clip_label)
                # print(self.match)
                # print(self.wrong)
                # print(self.similar)
                self.return_df = pd.DataFrame({"neuron":self.neuron,"CLIP dissect label":self.clip_label,"match":self.match,"wrong":self.wrong,"similar":self.similar,"user label":self.inputted_texts})
                self.return_df.to_csv("neuron_labels/{}_{}_{}_{}_{}.csv".format(target_name,target_layer,d_probe,concept_set_str,similarity_fn.__name__))
                sys.exit(app.exec())
    
    def append_wrong_text(self):
        text = self.wrong_text_edit.text_edit.text()
        if text:
            # Process wrong text as needed
            self.inputted_texts.append(text)
            self.wrong_texts.append(text)  # Store the wrong text
            self.wrong_text_edit.text_edit.clear()
            self.stacked_layout.setCurrentIndex(0)  # Switch back to buttons layout
    
            self.current_label_index += 1
    
            if self.current_label_index == len(self.image_labels):
                # All images in the current set have been processed
                self.current_label_index = 0  # Reset the current label index
                self.current_set_index += 1  # Move to the next set
    
            if self.current_set_index < len(self.image_sets):
                self.show_next_image_set()
            else:
                # All sets have been processed
                # Handle completion or exit here
                print("All sets have been processed.")

                self.return_df = pd.DataFrame({"neuron":self.neuron,"CLIP dissect label":self.clip_label,"match":self.match,"wrong":self.wrong,"similar":self.similar,"user label":self.inputted_texts})
                self.return_df.to_csv("neuron_labels/{}_{}_{}_{}_{}.csv".format(target_name,target_layer,d_probe,concept_set_str,similarity_fn.__name__))
                sys.exit(app.exec())






app = QApplication(sys.argv)



#Process Image Inputs and CLIP dissect arguments

clip_name = 'ViT-B/16'
target_name = 'resnet50'
target_layer = 'layer2'
d_probe = 'broden'
concept_set = 'data/20k.txt'
batch_size = 200
device = 'cuda'
pool_mode = 'avg'

concept_set_str = concept_set.split("/")[-1].split(".")[0]



save_dir = 'saved_activations'
similarity_fn = similarity.soft_wpmi

utils.save_activations(clip_name = clip_name, target_name = target_name, target_layers = [target_layer], 
                       d_probe = d_probe, concept_set = concept_set, batch_size = batch_size, 
                       device = device, pool_mode=pool_mode, save_dir = save_dir)

save_names = utils.get_save_names(clip_name = clip_name, target_name = target_name,
                                  target_layer = target_layer, d_probe = d_probe,
                                  concept_set = concept_set, pool_mode=pool_mode,
                                  save_dir = save_dir)

target_save_name, clip_save_name, text_save_name = save_names

similarities, target_feats = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                             text_save_name, similarity_fn, device=device)

neuron_count = similarities.shape[0]

with open(concept_set, 'r') as f: 
    words = (f.read()).split('\n')

pil_data = data_utils.get_data(d_probe)
top_vals, top_ids = torch.topk(target_feats, k=5, dim=0)
vals, idsx = torch.topk(similarities,k=1,largest=True)

image_sets = [[np.array(pil_data[j][0].resize([128,128])) for j in i] for i in np.array(top_ids).T]



clip_label = [words[int(i)] for i in idsx]
neuron = list(range(neuron_count))
title = "{} - {} | D probe: {} | Concept: {} | Sim: {}".format(target_name,target_layer,d_probe,concept_set_str,similarity_fn.__name__)

w = MainWindow(image_sets,clip_label,neuron,neuron_count,title)
w.show()
sys.exit(app.exec())








