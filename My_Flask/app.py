from flask import Flask,render_template,url_for,request

import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import gensim
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess

# Text librairies
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tag.util import untag
import contractions
from contractions import CONTRACTION_MAP
# import pycontractions # Alternative better package for removing contractions
from autocorrect import Speller

# Pour les données textuelles
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer

# Module to binarize data
from sklearn.preprocessing import Binarizer

spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))

# List of Adjective's tag from nltk package
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS'])


# All required cleaning functions
def clean_text(text):
	text = re.sub(r"\'", "'", text)
	text = re.sub(r"\n", " ", text)
	text = re.sub(r"\xa0", " ", text)
	text = re.sub('\s+', ' ', text)
	text = text.strip(' ')
	return text

def expand_contractions(text):
	text = contractions.fix(text)
	return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
	contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
	def expand_match(contraction):
		match = contraction.group(0)
		first_char = match[0]
		expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
		expanded_contraction = first_char+expanded_contraction[1:]
		return expanded_contraction
        
	expanded_text = contractions_pattern.sub(expand_match, text)
	expanded_text = re.sub("'", "", expanded_text)
	return expanded_text

def reducing_incorrect_character_repeatation(text):
	# Pattern matching for all case alphabets
	Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    
	# Limiting all the  repeatation to two characters.
	Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    
	# Pattern matching for all the punctuations that can occur
	Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    
	# Limiting punctuations in previously formatted string to only one.
	Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    
	# The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
	Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
	return Final_Formatted

def autocorrect(text):
	words = token.tokenize(text)
    
	# Option 'fast' in place
	spell = Speller(fast=True)
    
	words_correct = [spell(w) for w in words]
	return ' '.join(map(str, words_correct)) # Return the text untokenize

def remove_punctuation_and_number(text):
	"""remove all punctuation and number"""
	return text.translate(str.maketrans(" ", " ", charac)) 


def remove_non_alphabetical_character(text):
	"""remove all non-alphabetical character"""
	text = re.sub("[^a-z]+", " ", text) # remove all non-alphabetical character
	text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
	return text

def remove_single_letter(text):
	"""remove single alphabetical character"""
	text = re.sub(r"\b\w{1}\b", "", text) # remove all single letter
	text = re.sub("\s+", " ", text) # remove whitespaces left after the last operation
	text = text.strip(" ")
	return text

def remove_stopwords(text):
	"""remove common words in english by using nltk.corpus's list"""
	words = token.tokenize(text)
	filtered = [w for w in words if not w in stop_words]
    
	return ' '.join(map(str, filtered)) # Return the text untokenize

def remove_by_tag(text, undesired_tag):
	"""remove all words by using ntk tag (adjectives, verbs, etc.)"""
	words = token.tokenize(text) # Tokenize each words
	words_tagged = nltk.pos_tag(tokens=words, tagset=None, lang='eng') # Tag each words and return a list of tuples (e.g. ("have", "VB"))
	filtered = [w[0] for w in words_tagged if w[1] not in undesired_tag] # Select all words that don't have the undesired tags
    
	return ' '.join(map(str, filtered)) # Return the text untokenize

def stem_text(text):
	"""Stem the text"""
	words = nltk.word_tokenize(text) # tokenize the text then return a list of tuple (token, nltk_tag)
	stem_text = []
	for word in words:
		stem_text.append(stemmer.stem(word)) # Stem each words
	return " ".join(stem_text) # Return the text untokenize

def lemmatize_text(text):
	"""Lemmatize the text by using tag """

	tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
	lemmatized_text = []
	for word, tag in tokens_tagged:
		if tag.startswith('J'):
			lemmatized_text.append(lemmatizer.lemmatize(word,'a'))
		elif tag.startswith('V'):
			lemmatized_text.append(lemmatizer.lemmatize(word,'v'))
		elif tag.startswith('N'):
			lemmatized_text.append(lemmatizer.lemmatize(word,'n'))
		elif tag.startswith('R'):
			lemmatized_text.append(lemmatizer.lemmatize(word,'r'))
		else:
			lemmatized_text.append(lemmatizer.lemmatize(word))
	return " ".join(lemmatized_text)

# Cleaning function    
def prepared_text(df_doc_test):
	df_doc_test['text'] = df_doc_test['title'] + ' ' + df_doc_test['document']
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: clean_text(x))
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: expand_contractions(x)) 
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: autocorrect(x)) 
	df_doc_test['text'] = df_doc_test['text'].str.lower()
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: remove_non_alphabetical_character(x)) 
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: remove_single_letter(x)) 
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: remove_stopwords(x))
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: remove_by_tag(x, adjective_tag_list))
	df_doc_test['text'] = df_doc_test['text'].apply(lambda x: lemmatize_text(x))
	doc_test_prepared = list(df_doc_test['text'])
    
	return doc_test_prepared # Format !!

# Tags prediction function
def predict_tags_supervised_unseen_direct(doc_test_transformed, optimal_model, binarizer):
	# Get tags from inverse transform of prediction
	tags_supervised = binarizer.inverse_transform(optimal_model.predict(doc_test_transformed))[0]
        
	tags_returned = ', '.join(tags_supervised)
        
	return tags_returned

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
# Initialize dataframe
	df_doc_test = pd.DataFrame(columns=['document', 'title'])

# Load the dictionary/vocabulary 
	dictionary = open('dictionary.pkl','rb')
	dico = joblib.load(dictionary)
	n_terms = len(dico)
# Load the optimal supervised model
	model = open('optimal_model_supervised.pkl','rb')
	optimal_model = joblib.load(model)
# Load the binarizer for tags
	binarizer = open('binarizer.pkl','rb')
	binarizer = joblib.load(binarizer)

	if request.method == 'POST':
# Read title and document
		document = request.form['document']
		title = request.form['title']
# Store into one corpus
		df_doc_test['document'] = [document]
		df_doc_test['title'] = [title]
		# Manual testing
# 		df_doc_test['title'] = ['Machine-Learning on python, dataframes, data, coding']
# 		df_doc_test['document'] = ['record attribute et core thank reviewer find duplicate target attribute class et web record featuring require property relate try record attribute et core razor page web source http github djhmateer record test http daveabrock simplify model record get api class loginmodelnrtb pagemodel bindproperty never get set inputmodel cshtml produce dereference warning https get warning never get dereferences ie framework produce reference exception inputmodel input get set class class inputmodel require property non require make fire correct address form emailaddress always email post maybe js validator fire framework handle string get set property call password datatype datatype password datatype datatype password string password get set display name remember bool rememberme get set record record inputmodel string email get datatype datatype password string password get display name remember bool rememberme get record attribute pick record inputmodel string datatype datatype password string password display name remember rememberme onset iactionresult onpost modelstate input property type inputmodel bound bindproperty attribute log information success input return log information failure modelstate validation input return page']
# Clean the corpus
		doc_test_prepared = prepared_text(df_doc_test)
# Get to right formatting
		list_doc = [d.split() for d in doc_test_prepared]
# To BOW
		doc_new = [dico.doc2bow(doc, allow_update=False) for doc in list_doc]
# To Sparsed Matrix
		doc_test_transformed = corpus2csc(doc_new, num_terms=n_terms).transpose()
# Make prediction from Sparsed input
		tag_prediction = predict_tags_supervised_unseen_direct(doc_test_transformed, optimal_model, binarizer)
		#test_1 = df_doc_test['text']
		test_2 = tag_prediction
	return render_template('result.html',
                           #prediction_1 = test_1,
                           prediction_2 = test_2
                          )


if __name__ == '__main__':
	app.run(debug=True)