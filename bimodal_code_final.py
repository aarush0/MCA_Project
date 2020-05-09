# -*- coding: utf-8 -*-
"""MCA_BiModal_code

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Iz3_Y_72_73V1RTumxfsIQtDwxhfNuzC
"""

import numpy as np
import pandas as pd
import pickle
import os, sys
from collections import Counter, defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from tensorflow.keras.layers import Embedding
from functools import cmp_to_key
import pickle

x = pickle.load(open("/content/drive/My Drive/mca/MCA_Project/data/pickles/data_{}.p".format("emotion"),"rb"))
revs, W, word_idx_map, vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]

def get_word_indices(data_x):
  length = len(data_x.split())
  return np.array([word_idx_map[word] for word in data_x.split()] + [0]*(50-length))[:50]

def comp_id(x, y):
  xd = int(x[:x.find('_')])
  xu = int(x[x.find('_')+1:])

  yd = int(y[:y.find('_')])
  yu = int(y[y.find('_')+1:])

  if xd != yd:
    return xd - yd
  else:
    return xu - yu


def preprocess():

  train_data, val_data, test_data = {},{},{}

  counts_train = np.zeros((5,1))
  counts_test = np.zeros((5,1))
  counts_val = np.zeros((5,1))

  for i in range(len(revs)):

    utterance_id = revs[i]['dialog']+"_"+revs[i]['utterance']
    
    sentence_word_indices = get_word_indices(revs[i]['text'])
    
    label = label_index[revs[i]['y']]

    if label != 0 and label != 3 and label != 4 and label != 6:
      continue

    if label == 0:
      label = 0
    elif label == 3:
      label = 1
    elif label == 4:
      label = 2
    else:
      label = 3 

    if revs[i]['split']=="train" and counts_train[label] > 1500:
      continue

    if revs[i]['split']=="train":
        train_data[utterance_id]=(sentence_word_indices,label)
        counts_train[label] += 1
    elif revs[i]['split']=="val":
        val_data[utterance_id]=(sentence_word_indices,label)
        counts_val[label] += 1
    elif revs[i]['split']=="test":
        test_data[utterance_id]=(sentence_word_indices,label)
        counts_test[label] += 1

  dialogs = []
  utrs = -1
  d_cur = -1

  t_d = {}
  t_map = {}
  sorted_tr_keys = sorted(train_data.keys(), key=cmp_to_key(comp_id))

  for i in sorted_tr_keys:
    d = i[:i.find('_')]
    u = i[i.find('_') + 1:]
    ouid = d + '_' + u

    if d not in dialogs:
      d_cur += 1
      utrs = 0
      dialogs.append(d)
    else:
      utrs += 1

    df = d_cur
    uf = utrs

    uid = str(df) +'_' + str(uf)
    t_d[uid] = train_data[i]

    t_map[uid] = ouid

  dialogs = []
  utrs = -1
  d_cur = -1

  v_d = {}
  v_map = {}
  sorted_val_keys = sorted(val_data.keys(), key=cmp_to_key(comp_id))

  for i in sorted_val_keys:
    d = i[:i.find('_')]
    u = i[i.find('_') + 1:]
    ouid = d + '_' + u

    if d not in dialogs:
      d_cur += 1
      utrs = 0
      dialogs.append(d)
    else:
      utrs += 1

    df = d_cur
    uf = utrs

    uid = str(df) +'_' + str(uf)
    v_d[uid] = val_data[i]
    v_map[uid] = ouid

  dialogs = []
  utrs = -1
  d_cur = -1

  ts_d = {}
  ts_map = {}
  sorted_ts_keys = sorted(test_data.keys(), key=cmp_to_key(comp_id))

  for i in sorted_ts_keys:
    d = i[:i.find('_')]
    u = i[i.find('_') + 1:]
    ouid = d + '_' + u

    if d not in dialogs:
      d_cur += 1
      utrs = 0
      dialogs.append(d)
    else:
      utrs += 1

    df = d_cur
    uf = utrs

    uid = str(df) +'_' + str(uf)
    ts_d[uid] = test_data[i]
    ts_map[uid] = ouid
  
  return t_d, v_d, ts_d, t_map, v_map, ts_map


#preprocess()

max_length=50 # Maximum length of the sentence

class Dataloader:
    
    def __init__(self, mode=None):

        try:
            assert(mode is not None)
        except AssertionError as e:
            print("Set mode as 'Sentiment' or 'Emotion'")
            exit()

        self.MODE = mode # Sentiment or Emotion classification mode
        self.max_l = max_length

        """
            Loading the dataset: 
                - revs is a dictionary with keys/value: 
                    - text: original sentence
                    - split: train/val/test :: denotes the which split the tuple belongs to
                    - y: label of the sentence
                    - dialog: ID of the dialog the utterance belongs to
                    - utterance: utterance number of the dialog ID
                    - num_words: number of words in the utterance
                - W: glove embedding matrix
                - vocab: the vocabulary of the dataset
                - word_idx_map: mapping of each word from vocab to its index in W
                - label_index: mapping of each label (emotion or sentiment) to its assigned index, eg. label_index['neutral']=0
        """
        x = pickle.load(open("/content/drive/My Drive/mca/MCA_Project/data/pickles/data_{}.p".format(self.MODE.lower()),"rb"))
        self.revs, self.W, self.word_idx_map, self.vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        
        self.num_classes = 4
        print("Labels used for this classification: ", label_index)

        self.train_data, self.val_data, self.test_data, self.tr_map, self.v_map, self.ts_map = preprocess()

        # Creating dialogue:[utterance_1, utterance_2, ...] ids
        self.train_dialogue_ids = self.get_dialogue_ids(self.train_data.keys())
        self.val_dialogue_ids = self.get_dialogue_ids(self.val_data.keys())
        self.test_dialogue_ids = self.get_dialogue_ids(self.test_data.keys())

        # Max utternance in a dialog in the dataset
        self.max_utts = self.get_max_utts(self.train_dialogue_ids, self.val_dialogue_ids, self.test_dialogue_ids)

    def get_dialogue_ids(self, keys):
        ids=defaultdict(list)
        for key in keys:
            ids[key.split("_")[0]].append(int(key.split("_")[1]))
        for ID, utts in ids.items():
            ids[ID]=[str(utt) for utt in sorted(utts)]
        return ids

    def get_max_utts(self, train_ids, val_ids, test_ids):
        max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
        max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
        max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
        return np.max([max_utts_train, max_utts_val, max_utts_test])

    def get_one_hot(self, label):
        label_arr = [0]*self.num_classes
        label_arr[label]=1
        return label_arr[:]

    def get_dialogue_text_embs(self):
        key = list(self.train_data.keys())[0]
        
        pad = [0]*len(self.train_data[key][0])

        def get_emb(dialogue_id, local_data):
            dialogue_text = []
            for vid in dialogue_id.keys():
                local_text = []
                for utt in dialogue_id[vid]:
                    local_text.append(local_data[vid+"_"+str(utt)][0][:])
                for _ in range(self.max_utts-len(local_text)):
                    local_text.append(pad[:])
                dialogue_text.append(local_text[:self.max_utts])
            return np.array(dialogue_text)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_data)

    def get_dialogue_labels(self):

        def get_labels(ids, data):
            dialogue_label=[]

            for vid, utts in ids.items():
                local_labels=[]
                for utt in utts:
                    local_labels.append(self.get_one_hot(data[vid+"_"+str(utt)][1]))
                for _ in range(self.max_utts-len(local_labels)):
                    local_labels.append(self.get_one_hot(1)) # Dummy label
                dialogue_label.append(local_labels[:self.max_utts])
            return np.array(dialogue_label)

        self.train_dialogue_label=get_labels(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_label=get_labels(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_label=get_labels(self.test_dialogue_ids, self.test_data)

    def get_dialogue_labels_audio(self):

        def get_labels(ids, data, map):
            dialogue_label=[]

            for vid, utts in ids.items():
                local_labels=[]
                for utt in utts:
                    print(vid+"_"+str(utt), map[vid+"_"+str(utt)])
                    local_labels.append(self.get_one_hot(data[map[vid+"_"+str(utt)]][1]))
                for _ in range(self.max_utts-len(local_labels)):
                    local_labels.append(self.get_one_hot(1)) # Dummy label
                dialogue_label.append(local_labels[:self.max_utts])
            return np.array(dialogue_label)

        self.train_dialogue_label=get_labels(self.train_dialogue_ids, self.train_data, self.tr_map)
        self.val_dialogue_label=get_labels(self.val_dialogue_ids, self.val_data, self.v_map)
        self.test_dialogue_label=get_labels(self.test_dialogue_ids, self.test_data, self.ts_map)

        
    def get_dialogue_lengths(self):

        self.train_dialogue_length, self.val_dialogue_length, self.test_dialogue_length=[], [], []
        for vid, utts in self.train_dialogue_ids.items():
            self.train_dialogue_length.append(len(utts))
        for vid, utts in self.val_dialogue_ids.items():
            self.val_dialogue_length.append(len(utts))
        for vid, utts in self.test_dialogue_ids.items():
            self.test_dialogue_length.append(len(utts))

    def get_masks(self):

        self.train_mask = np.zeros((len(self.train_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.train_dialogue_length)):
            self.train_mask[i,:self.train_dialogue_length[i]]=1.0
        self.val_mask = np.zeros((len(self.val_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.val_dialogue_length)):
            self.val_mask[i,:self.val_dialogue_length[i]]=1.0
        self.test_mask = np.zeros((len(self.test_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.test_dialogue_length)):
            self.test_mask[i,:self.test_dialogue_length[i]]=1.0
        
    def load_text_data(self, ):

        self.get_dialogue_text_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()

    def load_audio_data(self, ):

        AUDIO_PATH = "/content/drive/My Drive/mca/MCA_Project/data/pickles/audio_embeddings_feature_selection_{}.pkl".format(self.MODE.lower())
        self.train_audio_emb, self.val_audio_emb, self.test_audio_emb = pickle.load(open(AUDIO_PATH,"rb"))

        self.get_dialogue_audio_embs()
        self.get_dialogue_lengths()
        #self.get_dialogue_labels_audio()
        self.get_dialogue_labels()
        self.get_masks()

    def get_dialogue_audio_embs(self):
        key = list(self.train_audio_emb.keys())[0]
        pad = [0]*len(self.train_audio_emb[key])

        def get_emb(dialogue_id, audio_emb, map):
            dialogue_audio=[]
            for vid in dialogue_id.keys():
                local_audio=[]
                for utt in dialogue_id[vid]:
                    try:
                        local_audio.append(audio_emb[map[vid+"_"+str(utt)]][:])
                    except:
                        print("oops")
                        print(vid+"_"+str(utt))
                        local_audio.append(pad[:])
                for _ in range(self.max_utts-len(local_audio)):
                    local_audio.append(pad[:])
                dialogue_audio.append(local_audio[:self.max_utts])
            return np.array(dialogue_audio)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_audio_emb, self.tr_map)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_audio_emb, self.v_map)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_audio_emb, self.ts_map)

import argparse
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import os, pickle
import numpy as np
import keras
import tensorflow as tf

class Network1:

	def __init__(self):
		self.classification_mode = "emotion"
		self.modality = "text"
		
    #self.PATH = "/content/drive/My Drive/Colab Notebooks/data/models/{}_weights_{}.hdf5".format("text",self.classification_mode.lower())
		#self.OUTPUT_PATH = "/content/drive/My Drive/Colab Notebooks/data/pickles/{}_{}.pkl".format("text",self.classification_mode.lower())
		print("Model initiated for {} classification".format(self.classification_mode))

	def load_data(self,m):
    
		print('Loading data')
    
		self.data = Dataloader(mode = self.classification_mode)
    
		if m == "text":
			self.data.load_text_data()
		elif m == "audio":
			self.data.load_audio_data()
		else:
			exit()
    
		self.train_x = self.data.train_dialogue_features
		self.val_x = self.data.val_dialogue_features
		self.test_x = self.data.test_dialogue_features
    
		self.train_y = self.data.train_dialogue_label
		self.val_y = self.data.val_dialogue_label
		self.test_y = self.data.test_dialogue_label
    
		self.train_mask = self.data.train_mask
		self.val_mask = self.data.val_mask
		self.test_mask = self.data.test_mask
    
		self.train_id = self.data.train_dialogue_ids.keys()
		self.val_id = self.data.val_dialogue_ids.keys()
		self.test_id = self.data.test_dialogue_ids.keys()
    
		self.sequence_length = self.train_x.shape[1]

		self.classes = self.train_y.shape[2]
    
		self.epochs = 2
		self.batch_size = 50

		if m == "text":
			self.train_x_text = self.train_x
			self.val_x_text = self.val_x
			self.test_x_text = self.test_x

			self.train_y_text = self.train_y
			self.val_y_text = self.val_y 
			self.test_y_text = self.test_y 
			
			self.train_mask_text = self.train_mask 
			self.val_mask_text = self.val_mask 
			self.test_mask_text = self.test_mask
			
			self.train_id_text = self.train_id 
			self.val_id_text = self.val_id 
			self.test_id_text = self.test_id 

			self.sequence_length_text = self.sequence_length

		if m == "audio":
			self.train_x_audio = self.train_x
			self.val_x_audio = self.val_x
			self.test_x_audio = self.test_x

			self.train_y_audio = self.train_y
			self.val_y_audio = self.val_y 
			self.test_y_audio = self.test_y 
			
			self.train_mask_audio = self.train_mask 
			self.val_mask_audio = self.val_mask 
			self.test_mask_audio = self.test_mask
			
			self.train_id_audio = self.train_id 
			self.val_id_audio = self.val_id 
			self.test_id_audio = self.test_id 
			self.sequence_length_audio = self.sequence_length

	def get_text_lstm(self):
		self.sentence_length = self.train_x.shape[2]
    
		self.embedding_dim = self.data.W.shape[1]
    
		self.vocabulary_size = self.data.W.shape[0]
		
		embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim, weights=[self.data.W], input_length=self.sentence_length, trainable=False)
    
		def slicer(x, index):
			return x[:,K.constant(index, dtype='int32'),:]
    
		def slicer_output_shape(input_shape):
			shape = list(input_shape)
			assert len(shape) == 3  # batch, seq_len, sent_len
			new_shape = (shape[0], shape[2])
			return new_shape

		def reshaper(x):
			return K.expand_dims(x, axis=3)
    
		def flattener(x):
			x = K.reshape(x, [-1,x.shape[1]*x.shape[2]])
			return x

		def flattener_output_shape(input_shape):
			shape = list(input_shape)
			new_shape = (shape[0], shape[2]*shape[1])
			return new_shape

		inputs = Input(shape=(self.sequence_length, self.sentence_length), dtype='int32')
		
		
		output = []
		for ind in range(self.sequence_length):
			local_input = Lambda(slicer, output_shape=slicer_output_shape, arguments={"index":ind})(inputs) # Batch, word_indices

			emb_output = embedding(local_input)
			reshape = Lambda(reshaper)(emb_output)

			flatten = Lambda(flattener, output_shape=flattener_output_shape,)(reshape)

			output.append(flatten)

		def stack(x):
			return K.stack(x, axis=1)
      
		outputs = Lambda(stack)(output)
		masked = Masking(mask_value =0)(outputs)
		
		lstm = Bidirectional(LSTM(200, activation='relu', return_sequences = True, dropout=0.3), name = 'lstm_t')(masked)
		self.text_lstm_layer = lstm

		lstm = Bidirectional(LSTM(200, activation='relu', return_sequences = True, dropout=0.3), name="utter_t")(lstm)
		output = TimeDistributed(Dense(self.classes,activation='softmax',kernel_initializer='uniform'))(lstm)

		model = Model(inputs, output)

		#model.summary()

		self.text_lstm =  model

		return lstm, inputs

	def get_audio_lstm(self):

		self.embedding_dim = self.train_x.shape[2]

		print("Creating Model...")
		
		inputs = Input(shape=(self.sequence_length, self.embedding_dim), dtype='float32')
		masked = Masking(mask_value =0)(inputs)
		lstm = Bidirectional(LSTM(200, activation='tanh', return_sequences = True, dropout=0.4), name='lstm_a')(masked)
		self.audio_lstm_layer = lstm
		lstm = Bidirectional(LSTM(200, activation='tanh', return_sequences = True, dropout=0.4), name="utter_a")(lstm)
		output = TimeDistributed(Dense(self.classes,activation='softmax',kernel_initializer='uniform'))(lstm)

		model = Model(inputs, output)

		self.audio_lstm = model

		return lstm, inputs

	def get_final_model(self, tl, al, ti, ai):
		#attn_out = tensorflow.keras.layers.Attention()([tl, al])
	
		
		at_layer = HanAttention()
		at_layer.build(tl.shape)
		attn_scores = at_layer.call([tl,al])

		concat_output2 = Concatenate(axis=-1, name='concat_layer')([attn_scores,tl])

		lstm = Bidirectional(LSTM(200, activation='tanh', return_sequences = True, dropout=0.4), name='lstm_f')(concat_output2)
		output = TimeDistributed(Dense(self.classes,activation='softmax',kernel_initializer='uniform'))(lstm)

		self.merged_model = Model([ti, ai], output)
		
	def train_lstm(self, m):
		if m == 'text':
			model = self.text_lstm
		elif m == 'audio':
			model = self.audio_lstm
		
		model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal')
		early_stopping = EarlyStopping(monitor='val_loss', patience=20)

		model.fit(self.train_x, self.train_y,
		                epochs=self.epochs,
		                batch_size=self.batch_size,
		                sample_weight=self.train_mask,
		                shuffle=True, 
		                callbacks=[early_stopping],
		                validation_data=(self.val_x, self.val_y, self.val_mask))

		
		self.test_model(m)
		return model
	
	def train_network(self):
		model = self.merged_model
		#print("HELLOOO")
		#multi_head = MultiHeadAttention( head_num=5, name='Multi-Head' )(lstm)
		model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal')
		early_stopping = EarlyStopping(monitor='loss', patience=10)
		
		#model.summary()
		model.fit([self.train_x_text, self.train_x_audio], self.train_y, epochs=self.epochs,batch_size=self.batch_size)
	
		return model


	def test_model(self, m):
		if m == 'text':
			model = self.text_lstm
			#intermediate_layer_model = Model(input=model.input, output=model.get_layer("lstm_t").output)
		elif m == 'audio':
			model = self.audio_lstm
			#intermediate_layer_model = Model(input=model.input, output=model.get_layer("lstm_a").output)
		elif m == 'merged':
			model = self.merged_model
			#intermediate_layer_model = Model(input=model.input, output=model.get_layer("lstm_f").output)

		calc_test_result(model.predict(self.test_x), self.test_y, self.test_mask)
		
def calc_test_result(pred_label, test_label, test_mask):

		true_label=[]
		predicted_label=[]

		for i in range(pred_label.shape[0]):
			for j in range(pred_label.shape[1]):
				if test_mask[i,j]==1:
					true_label.append(np.argmax(test_label[i,j] ))
					predicted_label.append(np.argmax(pred_label[i,j] ))
		print("Confusion Matrix :")
		print(confusion_matrix(true_label, predicted_label))
		print("Classification Report :")
		print(classification_report(true_label, predicted_label, digits=4))
		print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))


from tensorflow.keras.layers import Layer

class HanAttention(Layer):
  """
  Refer to [Hierarchical Attention Networks for Document Classification]
    (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    wrap `with tf.variable_scope(name, reuse=tf.AUTO_REUSE):`
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, features)
  """

  def __init__(self,
               W_regularizer=None,
               u_regularizer=None,
               b_regularizer=None,
               W_constraint=None,
               u_constraint=None,
               b_constraint=None,
               use_bias=True,
               **kwargs):

    super().__init__(**kwargs)
    self.supports_masking = True
    self.init = tf.keras.initializers.get('glorot_uniform')

    self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
    self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
    self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

    self.W_constraint = tf.keras.constraints.get(W_constraint)
    self.u_constraint = tf.keras.constraints.get(u_constraint)
    self.b_constraint = tf.keras.constraints.get(b_constraint)

    self.use_bias = use_bias

  def build(self, input_shape):
    # pylint: disable=attribute-defined-outside-init
    #assert len(input_shape) == 3

    self.W = self.add_weight(
        name='{}_W'.format(self.name),
        shape=(
            int(input_shape[-1]),
            int(input_shape[-1]),
        ),
        initializer=self.init,
        regularizer=self.W_regularizer,
        constraint=self.W_constraint)

    if self.use_bias:
      self.b = self.add_weight(
          name='{}_b'.format(self.name),
          shape=(int(input_shape[-1]),),
          initializer='zero',
          regularizer=self.b_regularizer,
          constraint=self.b_constraint)

    self.attention_context_vector = self.add_weight(
        name='{}_att_context_v'.format(self.name),
        shape=(int(input_shape[-1]),),
        initializer=self.init,
        regularizer=self.u_regularizer,
        constraint=self.u_constraint)
    self.built = True

  # pylint: disable=missing-docstring, no-self-use
  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    # do not pass the mask to the next layers
    return None

  
  def call(self, inputs, training=None, mask=None):
    batch_size = tf.shape(inputs)[1]
    W_3d = tf.tile(tf.expand_dims(self.W, axis=0), tf.stack([batch_size, 1, 1]))
    # [batch_size, steps, features]
    input_projection = tf.matmul(inputs, W_3d)

    if self.use_bias:
      input_projection += self.b

    input_projection = tf.tanh(input_projection)

    # [batch_size, steps, 1]
    similaritys = tf.reduce_sum(
        tf.multiply(input_projection, self.attention_context_vector),
        axis=2,
        keep_dims=True)

    # [batch_size, steps, 1]
    if mask is not None:
      attention_weights = masked_softmax(similaritys, mask, axis=1)
    else:
      attention_weights = tf.nn.softmax(similaritys, axis=1)

    # [batch_size, features]
    attention_output = tf.reduce_sum(
        tf.multiply(inputs, attention_weights), axis=0)
    return attention_output

  # pylint: disable=no-self-use

  def compute_output_shape(self, input_shape):
    """compute output shape"""
    return input_shape[0], input_shape[-1]

