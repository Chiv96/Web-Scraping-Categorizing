from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk
import os
import yaml


class FrequencySummarizer:
	def __init__(self, min_cut=0.1, max_cut=0.9):
		"""
		 Initilize the text summarizer.
		 Words that have a frequency term lower than min_cut 
		 or higer than max_cut will be ignored.
		"""
		self._min_cut = min_cut
		self._max_cut = max_cut 
		self._stopwords = set(stopwords.words('english') + list(punctuation))

	def _compute_frequencies(self, word_sent):
		""" 
		  Compute the frequency of each of word.
		  Input: 
		   word_sent, a list of sentences already tokenized.
		  Output: 
		   freq, a dictionary where freq[w] is the frequency of w.
		"""
		freq = defaultdict(int)
		for s in word_sent:
			for word in s:
				if word not in self._stopwords:
					freq[word] += 1
		# frequencies normalization and fitering
		m = float(max(freq.values()))
		for w in list(freq.keys()):
			freq[w] = freq[w]/m
			if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
				del freq[w]
		return freq

	def summarize(self, text, n):
		"""
		  Return a list of n sentences 
		  which represent the summary of text.
		"""
		sents = sent_tokenize(text)	
		assert n <= len(sents)
		word_sent = [word_tokenize(s.lower()) for s in sents]
		self._freq = self._compute_frequencies(word_sent)
		ranking = defaultdict(int)
		for i,sent in enumerate(word_sent):
			for w in sent:
				if w in self._freq:
					ranking[i] += self._freq[w]
		sents_idx = self._rank(ranking, n)    
		return [sents[j] for j in sents_idx]

	def _rank(self, ranking, n):
		#return the first n sentences with highest ranking
		return nlargest(n, ranking, key=ranking.get)
		
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup

fs = FrequencySummarizer()
print('\nEnter the URL : ')
url=input()
feed_xml = urlopen(url).read()
soup = BeautifulSoup(feed_xml,"html.parser")
text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
summary=fs.summarize(text, 3)

#------------------------CATEGORIZING BEGINS HERE


class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):
    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos
		
class Score(object):
    def __init__(self):
        self.bscore = 0			#Business related terms add to this score
        self.tscore = 0			#Tech related terms add to this score
        self.hscore = 0			#Health related terms add to this score

textscore=Score()
		
text=str(summary)
splitter = Splitter()
postagger = POSTagger()
splitted_sentences=splitter.split(text)
#print (splitted_sentences)
pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
#print (pos_tagged_sentences)

class DictionaryTagger(object):
	def __init__(self, dictionary_paths):
	
		files = [open(path, 'r') for path in dictionary_paths]
		dictionaries = [yaml.load(dict_file) for dict_file in files]
		map(lambda x: x.close(), files)
		self.dictionary = {}
		self.max_key_size = 0
		for curr_dict in dictionaries:
			for key in curr_dict:
				if key in self.dictionary:
					self.dictionary[key].extend(curr_dict[key])
				else:
					self.dictionary[key] = curr_dict[key]
					self.max_key_size = max(self.max_key_size, len(key))

	def tag(self, postagged_sentences):
		return [self.tag_sentence(sentence) for sentence in postagged_sentences]

	def tag_sentence(self, sentence, tag_with_lemmas=False):
		"""
		the result is only one tagging of all the possible ones.
		The resulting tagging is determined by these two priority rules:
			- longest matches have higher priority
			- search is made from left to right
		"""
		
		tag_sentence = []
		N = len(sentence)
		if self.max_key_size == 0:
			self.max_key_size = N
		i = 0
		while (i < N):
			j = min(i + self.max_key_size, N) #avoid overflow
			tagged = False
			while (j > i):
				expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
				expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
				if tag_with_lemmas:
					literal = expression_lemma
				else:
					literal = expression_form
				if literal in self.dictionary:
					#self.logger.debug("found: %s" % literal)
					is_single_token = j - i == 1
					original_position = i
					i = j
					taggings = [tag for tag in self.dictionary[literal]]
					tagged_expression = (expression_form, expression_lemma, taggings)
					if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
						original_token_tagging = sentence[original_position][2]
						tagged_expression[2].extend(original_token_tagging)
					tag_sentence.append(tagged_expression)
					flag = 1
					tagged = True				
				else:
					j = j - 1
			if not tagged:
				tag_sentence.append(sentence[i])
				i += 1
		return tag_sentence
		
def value_of(type):
	if type == 'business': textscore.bscore+=1
	if type == 'tech': textscore.tscore+=1
	if type == 'health': textscore.hscore+=1
	

def type_score(dict_tagged_sentences):    
	for sentence in dict_tagged_sentences:
		for token in sentence :
			for tag in token[2]:
				value_of(tag)
	
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
rel_path1 = "Dictionaries2/business.yml"
rel_path2 = "Dictionaries2/tech.yml"
rel_path3 = "Dictionaries2/health.yml"
abs_file_path = os.path.join(script_dir, rel_path1)
abs_file_path = os.path.join(script_dir, rel_path2)
abs_file_path = os.path.join(script_dir, rel_path3)
dicttagger = DictionaryTagger([ 'Dictionaries2/business.yml', 'Dictionaries2/tech.yml', 'Dictionaries2/health.yml'])
dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
print(dict_tagged_sentences)
type_score(dict_tagged_sentences)
#print("\n score is --", textscore.bscore,textscore.tscore,textscore.hscore)

if textscore.bscore>textscore.tscore:		#see which score is the max 
	if textscore.bscore>textscore.hscore:
		max='b'
	else:
		max='h'
else:
	if textscore.tscore>textscore.hscore:
		max='t'
	else:
		max='h'
		
print()
if max=='b':
	print("The text is business related.\n")
elif max=='t':
	print("The text is tech related.\n")	
elif max=='h':
	print("The text is health related.\n")
print()
	

#------------------------CATEGORIZING ENDS HERE

print("\nSummary is -:\n\n", summary)