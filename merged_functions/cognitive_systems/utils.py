# Merged file for cognitive_systems/utils
# This file contains code merged from multiple repositories

import sys
import sysconfig
import site

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from Cython.Distutils import build_ext
import os

# From cython/setup.py
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

import nltk
import yaml
import re

# From sentiment/basic_sentiment_analysis.py
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

# From sentiment/basic_sentiment_analysis.py
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

# From sentiment/basic_sentiment_analysis.py
class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.safe_load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                elif key is not False and key is not True:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))
                elif key is False:
#                    print curr_dict[key]
                    key = "false"
                    self.dictionary[key] = curr_dict [False]
                    self.max_key_size = max(self.max_key_size, len(key))
                else:
                    key = "true"
                    self.dictionary[key] = curr_dict [True]
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
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

# From sentiment/basic_sentiment_analysis.py
def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

# From sentiment/basic_sentiment_analysis.py
def sentence_score(sentence_tokens, previous_token, acum_score, neg_num):
    if not sentence_tokens:
        if(neg_num % 2 == 0):
            return acum_score
        else:
            acum_score *= -1.0
            return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                neg_num += 1
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score, neg_num)

# From sentiment/basic_sentiment_analysis.py
def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0, 0) for sentence in review])

# From sentiment/basic_sentiment_analysis.py
def sentiment_parse(plain_text):
    splitter = Splitter()
    postagger = POSTagger()
    splitted_sentences = splitter.split(plain_text)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    score = sentiment_score(dict_tagged_sentences)
    return score

# From sentiment/basic_sentiment_analysis.py
def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

# From sentiment/basic_sentiment_analysis.py
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

# From sentiment/basic_sentiment_analysis.py
def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

# From sentiment/basic_sentiment_analysis.py
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
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

from opencog.scheme_wrapper import scheme_eval_as
from opencog.scheme_wrapper import scheme_eval
import logging
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import Filters

# From chatbot/telegram_bot.py
def start(bot, update):
    """Send a message when the  /start command is issued."""
    bot.send_message(chat_id=update.message.chat_id, text='Hello!')

# From chatbot/telegram_bot.py
def help(bot, update):
    """Send a message when the  /help command is issued."""
    bot.send_message(chat_id=update.message.chat_id, text='Help!')

# From chatbot/telegram_bot.py
def echo(bot, update):
    """Echo the user message."""
    print ("Ok, we got message {}".format(update.message.text))
    reply = scheme_eval(atomspace, '(process-query "{}" "{}")'.format(update.message.from_user.first_name, update.message.text))
    print ("And now we have a reply {}".format(reply))
    reply_decoded = reply.decode("utf-8")
    print ("Decoding the reply: {}".format(reply_decoded))
    bot.send_message(chat_id=update.message.chat_id, text=reply_decoded)

# From chatbot/telegram_bot.py
def error(bot, update):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', bot, update.error)

# From chatbot/telegram_bot.py
def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

from __future__ import print_function
from pprint import pprint
from opencog.atomspace import types
from opencog.atomspace import AtomSpace
from opencog.atomspace import TruthValue
from opencog.cogserver_type_constructors import *
from agents.hobbs import HobbsAgent
from agents.dumpAgent import dumpAgent
from opencog.scheme_wrapper import load_scm
from opencog.scheme_wrapper import scheme_eval_h
from opencog.scheme_wrapper import __init__

from opencog.cogserver import MindAgent
from opencog.type_constructors import TruthValue
from opencog import logger
import queue
import time

# From agents/hobbs.py
class BindLinkExecution():

    '''
    Executes a (cog-execute! xxx) command and return the results of it
    '''

    def __init__(self,atomspace,anchorNode, target, command):

        '''
        Stores necessary information
        '''

        self.atomspace=atomspace
        self.anchorNode=anchorNode
        self.target=target
        self.command=command
        self.response=None
        scheme_eval(self.atomspace, "(use-modules (opencog) (opencog exec))")
        scheme_eval(self.atomspace, "(use-modules (opencog nlp))")
        scheme_eval(self.atomspace, "(use-modules (opencog nlp oc))")

    def execution(self):

        '''
        First binds the "anchorNode" with the "target" if "anchorNode" exists, then executes scheme command "command"
        '''

        if self.anchorNode != None and self.target != None:
            self.tmpLink=self.atomspace.add_link(types.ListLink, [self.anchorNode, self.target], TruthValue(1.0, 100))
        else:
            self.tmpLink=None
        self.response = scheme_eval_h(self.atomspace, self.command)
        d=3;

    def returnResponse(self):

        '''
        Returns list of atoms resulted in previous execution of a scheme command
        '''

        if self.response==None:
            return
        rv=[]
        listOfLinks=self.response.out
        for link in listOfLinks:
            atom=(link.out)[1]
            rv.append(atom)
        for link in listOfLinks:
            self.atomspace.remove(link)
        self.atomspace.remove(self.response)
        self.response=None
        return rv

    def clear(self):

        '''
        Cleans up the Link between the "anchorNode" and the "target".
        '''

        if self.tmpLink!=None:
            self.atomspace.remove(self.tmpLink)

# From agents/hobbs.py
class HobbsAgent(MindAgent):

    '''
    Does anaphora resolutions by doing Breadth-First search on the parse tree, rejects any antecedents which are matched by filters
    '''

    def __init__(self):
        self.checked=dict()
        self.wordNumber=dict()
        self.atomspace = None

        self.currentPronoun = None
        self.currentPronounNode = None
        self.currentTarget = None
        self.currentProposal = None
        self.pronounNumber = None

        self.pronouns = None
        self.roots = None

        self.confidence = 1.0

        self.numOfFilters=7
        self.number_of_searching_sentences=3
        self.DEBUG = True

        log.fine("\n===========================================================\n Starting hobbs agent.....\n=========================================================== ")

    def bindLinkExe(self,anchorNode, target, command):

        '''
        Just combines all the steps of executing a scheme command into a single function.
        '''

        exe=BindLinkExecution(self.atomspace,anchorNode, target, command)
        exe.execution()
        rv=exe.returnResponse()
        exe.clear()
        return rv

    def StringToNumber(self,str):

        '''
        Converts a string to an integer.
        '''

        # Add 0.1 to avoid float-point rounding error.
        return int(float(str) + 0.1)

    def getWordNumber(self,node):

        '''
        Returns the WordSequence number associated with the 'node'
        '''

        return self.wordNumber[node.name]

    def getSentenceNumber(self,node):

        '''
        Given a ParseNode, returns a SentenceNumber of a SentenceNode associated with it.
        '''

        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! getNumberNode_ParseNode)')
        return int(rv[0].name)

    def sortNodes(self,list,keyFunc):

        '''
        Sorts nodes according to their word sequence number and returns the sorted list.
        '''
        return sorted(list,key=keyFunc)

    def getChildren(self,node):

        '''
        Returns a sorted list of children nodes of current node.
        '''

        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! getChildren)')
        return self.sortNodes(rv,self.getWordNumber)

    def generateReferenceLink(self,anaphora,antecedent,tv):
        '''
        Generates a reference Link for a pair of anaphora and antecedent with confidence "confidence".
        '''

        link = self.atomspace.add_link(types.ReferenceLink, [anaphora, antecedent], tv)
        log.fine("Generated a Reference :\n")
        log.fine("{0}\n".format(link))
        log.fine("===========================================================")

    def getConjunction(self,node):

        '''
        Returning the other part of a conjunction if conjunction exists and anaphor is "Plural"
        '''

        return self.bindLinkExe(self.currentProposal,node,'(cog-execute! getConjunction)')

    def checkConjunctions(self,node):

        '''
        Checking if conjunction resolution applies to the "node", returning True if it applies, False otherwise.
        '''

        conjunction=self.getConjunction(node);

        if len(conjunction)>0:

            conjunction_list=[]
            conjunction_list.append(node)
            conjunction_list.extend(conjunction)

            # We don't want to output this to unit tests
            if self.DEBUG and filter!=-1:
                print("accepted \n"+str(conjunction_list))
            log.fine("accepted \n"+str(conjunction_list))
            self.generateReferenceLink(self.currentPronoun,self.atomspace.add_link(types.AndLink, conjunction_list, TruthValue(1.0, 1.0)),TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
            self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
            return True
        return False

    def propose(self,node,filter=-1):
        '''
        It iterates all filters, reject the antecedent or "node" if it's matched by any filters.
        '''

        self.currentResolutionLink_pronoun=self.atomspace.add_link(types.ListLink, [self.currentResolutionNode, self.currentPronoun, node], TruthValue(1.0, 100))
        rejected = False
        filterNumber=-1

        self.checkConjunctions(node)

        start=1
        end=self.numOfFilters+1

        '''
        For debugging purposes.
        '''
        if filter!=-1:
            start=filter
            end=filter+1

        for index in range(start,end):
            command='(cog-execute! filter-#'+str(index)+')'
            rv=self.bindLinkExe(self.currentProposal,node,command)
            if len(rv)>0:
                '''
                Reject it
                '''
                rejected = True
                filterNumber=index
                break

        if not rejected:

            # We don't want to output this to unit tests
            if self.DEBUG:
                    print("accepted "+node.name)
            log.fine("accepted "+node.name)
            self.generateReferenceLink(self.currentPronoun,node,TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
            self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
        else:
            self.generateReferenceLink(self.currentPronoun,node,TV_FOR_FILTERED_OUT_ANTECEDENTS)
            #if self.DEBUG:
                   # print("rejected "+node.name+" by filter-#"+str(index))

        self.atomspace.remove(self.currentResolutionLink_pronoun)
        return not rejected

    def Checked(self,node):

        '''
        Since graph is not necessarily a forest, this agent actually does a Breadth-First search on a general graph for
        each pronoun, so we need to avoid cycling around the graph by marking each node as checked if we have visited it once.
        '''

        if node.name in self.checked:
            return True
        self.checked[node.name]=True
        return False

    def bfs(self,node):

        '''
        Does a Breadth-First search, starts with "node"
        '''

        '''
        rv is used for unit tests
        '''
        rv=[]

        if node==None:
            #print("found you bfs")
            return
        q=queue.Queue()
        q.put(node)
        while not q.empty():
            front=q.get()
            rv.append(front)
            self.propose(front)
            children=self.getChildren(front)
            if len(children)>0:
                for node in children:
                    if not self.Checked(node):
                        q.put(node)
        return rv

    def getWords(self):

        '''
        Returns a list of words in the atomspace
        '''

        rv=self.bindLinkExe(None,None,'(cog-execute! getWords)')
        return self.sortNodes(rv,self.getWordNumber)

    def getTargets(self,words):

        '''
        Returns a list of references needed to be resolved.
        '''

        targets=[]
        for word in words:
            matched=False
            for index in range(1,self.numOfPrePatterns+1):
                command='(cog-execute! pre-process-#'+str(index)+')'
                rv=self.bindLinkExe(self.currentTarget,word,command)
                if len(rv)>0:
                    matched=True
                    break
            if matched:
                targets.append(word)
        return targets

    def getPronouns(self):
        rv=self.bindLinkExe(None,None,'(cog-execute! getPronouns)')
        return self.sortNodes(rv,self.getWordNumber)

    def getRoots(self):

        '''
        Return a list of roots(incoming degree of 0)
        '''

        self.bindLinkExe(None,None,'(cog-execute! connectRootsToParseNodes)')
        rv= self.bindLinkExe(None,None,'(cog-execute! getAllParseNodes)')
        return self.sortNodes(rv,self.getSentenceNumber)

    def getRootOfNode(self,target):
        '''
        Returns a ParseNode associated with the "target"
        '''

        rv=self.bindLinkExe(self.currentTarget,target,'(cog-execute! getParseNode)')
        return rv[0]

    def  previousRootExist(self,root):

        '''
        "previous" means that a root with smaller word sequence number than the word sequence number of current "roots".
        '''
        return not self.roots[0].name==root.name

    def getPrevious(self,root):

        '''
        Return a previous root.
        '''

        rootNumber=self.getSentenceNumber(root)
        for root in reversed(self.roots):
            number=self.getSentenceNumber(root)
            if number<rootNumber:
                return root

    def getAllNumberNodes(self):

        '''
        Finds word sequence number for each word
        '''

        rv= self.bindLinkExe(None, None, '(cog-execute!  getAllNumberNodes)')
        for link in rv:
            out=link.out
            if out[0].type==types.WordInstanceNode:
                self.wordNumber[out[0].name]=self.StringToNumber(out[1].name)

    def initilization(self,atomspace):
        '''
        Initializes necessary variables. Loads rules.
        '''

        self.atomspace = atomspace

        self.PleonasticItNode=atomspace.add_node(types.AnchorNode, 'Pleonastic-it', TruthValue(1.0, 100))
        self.currentPronounNode = atomspace.add_node(types.AnchorNode, 'CurrentPronoun', TruthValue(1.0, 100))
        self.currentTarget = atomspace.add_node(types.AnchorNode, 'CurrentTarget', TruthValue(1.0, 100))
        self.currentProposal = atomspace.add_node(types.AnchorNode, 'CurrentProposal', TruthValue(1.0, 100))
        self.resolvedReferences=atomspace.add_node(types.AnchorNode, 'Resolved references', TruthValue(1.0, 100))
        self.currentResolutionNode=atomspace.add_node(types.AnchorNode, 'CurrentResolution', TruthValue(1.0, 100))
        self.pronounNumber = -1

        data=["opencog/nlp/anaphora/rules/getChildren.scm",
              "opencog/nlp/anaphora/rules/getNumberNode_WordInstanceNode.scm",
              "opencog/nlp/anaphora/rules/getNumberNode_ParseNode.scm",
              "opencog/nlp/anaphora/rules/connectRootsToParseNodes.scm",
              "opencog/nlp/anaphora/rules/getAllNumberNodes.scm",
              "opencog/nlp/anaphora/rules/getAllParseNodes.scm",
              "opencog/nlp/anaphora/rules/getConjunction.scm",
              "opencog/nlp/anaphora/rules/getParseNode.scm",
              "opencog/nlp/anaphora/rules/getWords.scm",
              "opencog/nlp/anaphora/rules/isIt.scm",

              "opencog/nlp/anaphora/rules/filters/filter-#1.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#2.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#3.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#4.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#5.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#6.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#7.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#8.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#9.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#10.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#11.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#12.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#13.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#14.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#15.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#16.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#17.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#18.scm",

              "opencog/nlp/anaphora/rules/pre-process/pre-process-#1.scm",
              "opencog/nlp/anaphora/rules/pre-process/pre-process-#2.scm",
              "opencog/nlp/anaphora/rules/pre-process/pre-process-#3.scm",

              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#1.scm",
              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#2.scm",
              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#3.scm",

              ]

        self.numOfFilters=18
        self.numOfPrePatterns=3
        self.numOfPleonasticItPatterns=3

        for item in data:
            load_scm(atomspace, item)

        self.getAllNumberNodes()
        self.pronouns=self.getTargets(self.getWords())
        self.roots = self.getRoots()


    def addPronounToResolvedList(self,node):
        '''
        Mark current pronoun as resolved.
        '''

        self.atomspace.add_link(types.ListLink,[self.resolvedReferences,node],TruthValue(1.0, 100))

    def pleonastic_it(self,node):
        '''
        Check if the node is the word "it".
        '''
        matched=False
        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! isIt)')
        if len(rv)>0:
            for index in range(1,self.numOfPleonasticItPatterns+1):
                command='(cog-execute! pleonastic-it-#'+str(index)+')'
                rv=self.bindLinkExe(self.currentTarget,node,command)
                if len(rv)>0:
                    matched=True
                    break
                #print("rejected "+node.name+" by filter-#"+str(index))

        return matched

    def run(self, atomspace):
        self.initilization(atomspace)

        for pronoun in self.pronouns:


            self.checked.clear()
            self.pronounNumber=self.getWordNumber(pronoun)
            self.confidence=1-CONFIDENCE_DECREASING_RATE

            '''
            Binds current "pronoun" with "currentPronounNode".
            This part is used by pattern matcher.
            '''

            tmpLink=self.atomspace.add_link(types.ListLink, [self.currentPronounNode, pronoun], TruthValue(1.0, 100))
            self.currentPronoun=pronoun
            root=self.getRootOfNode(pronoun)

            if self.DEBUG:
                print("Resolving....")
                print(pronoun)
            log.fine("Resolving \n{0}".format(pronoun))

            '''
            Check if it's a pleonastic it.
            '''

            if self.pleonastic_it(pronoun):
                self.generateReferenceLink(pronoun,self.PleonasticItNode,TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
                self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
                if self.DEBUG:
                    print("accepted "+self.PleonasticItNode.name)
                    log.fine("accepted "+self.PleonasticItNode.name)

            sent_counter=1;
            while True:
                if root==None:
                    break
                self.bfs(root)
                if self.previousRootExist(root) and sent_counter<=NUMBER_OF_SEARCHING_SENTENCES:
                    root=self.getPrevious(root)
                    sent_counter=sent_counter+1
                else:
                    break
            self.atomspace.remove(tmpLink)
            self.addPronounToResolvedList(pronoun)

# From agents/hobbs.py
def execution(self):

        '''
        First binds the "anchorNode" with the "target" if "anchorNode" exists, then executes scheme command "command"
        '''

        if self.anchorNode != None and self.target != None:
            self.tmpLink=self.atomspace.add_link(types.ListLink, [self.anchorNode, self.target], TruthValue(1.0, 100))
        else:
            self.tmpLink=None
        self.response = scheme_eval_h(self.atomspace, self.command)
        d=3;

# From agents/hobbs.py
def returnResponse(self):

        '''
        Returns list of atoms resulted in previous execution of a scheme command
        '''

        if self.response==None:
            return
        rv=[]
        listOfLinks=self.response.out
        for link in listOfLinks:
            atom=(link.out)[1]
            rv.append(atom)
        for link in listOfLinks:
            self.atomspace.remove(link)
        self.atomspace.remove(self.response)
        self.response=None
        return rv

# From agents/hobbs.py
def clear(self):

        '''
        Cleans up the Link between the "anchorNode" and the "target".
        '''

        if self.tmpLink!=None:
            self.atomspace.remove(self.tmpLink)

# From agents/hobbs.py
def bindLinkExe(self,anchorNode, target, command):

        '''
        Just combines all the steps of executing a scheme command into a single function.
        '''

        exe=BindLinkExecution(self.atomspace,anchorNode, target, command)
        exe.execution()
        rv=exe.returnResponse()
        exe.clear()
        return rv

# From agents/hobbs.py
def StringToNumber(self,str):

        '''
        Converts a string to an integer.
        '''

        # Add 0.1 to avoid float-point rounding error.
        return int(float(str) + 0.1)

# From agents/hobbs.py
def getWordNumber(self,node):

        '''
        Returns the WordSequence number associated with the 'node'
        '''

        return self.wordNumber[node.name]

# From agents/hobbs.py
def getSentenceNumber(self,node):

        '''
        Given a ParseNode, returns a SentenceNumber of a SentenceNode associated with it.
        '''

        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! getNumberNode_ParseNode)')
        return int(rv[0].name)

# From agents/hobbs.py
def sortNodes(self,list,keyFunc):

        '''
        Sorts nodes according to their word sequence number and returns the sorted list.
        '''
        return sorted(list,key=keyFunc)

# From agents/hobbs.py
def getChildren(self,node):

        '''
        Returns a sorted list of children nodes of current node.
        '''

        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! getChildren)')
        return self.sortNodes(rv,self.getWordNumber)

# From agents/hobbs.py
def generateReferenceLink(self,anaphora,antecedent,tv):
        '''
        Generates a reference Link for a pair of anaphora and antecedent with confidence "confidence".
        '''

        link = self.atomspace.add_link(types.ReferenceLink, [anaphora, antecedent], tv)
        log.fine("Generated a Reference :\n")
        log.fine("{0}\n".format(link))
        log.fine("===========================================================")

# From agents/hobbs.py
def getConjunction(self,node):

        '''
        Returning the other part of a conjunction if conjunction exists and anaphor is "Plural"
        '''

        return self.bindLinkExe(self.currentProposal,node,'(cog-execute! getConjunction)')

# From agents/hobbs.py
def checkConjunctions(self,node):

        '''
        Checking if conjunction resolution applies to the "node", returning True if it applies, False otherwise.
        '''

        conjunction=self.getConjunction(node);

        if len(conjunction)>0:

            conjunction_list=[]
            conjunction_list.append(node)
            conjunction_list.extend(conjunction)

            # We don't want to output this to unit tests
            if self.DEBUG and filter!=-1:
                print("accepted \n"+str(conjunction_list))
            log.fine("accepted \n"+str(conjunction_list))
            self.generateReferenceLink(self.currentPronoun,self.atomspace.add_link(types.AndLink, conjunction_list, TruthValue(1.0, 1.0)),TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
            self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
            return True
        return False

# From agents/hobbs.py
def propose(self,node,filter=-1):
        '''
        It iterates all filters, reject the antecedent or "node" if it's matched by any filters.
        '''

        self.currentResolutionLink_pronoun=self.atomspace.add_link(types.ListLink, [self.currentResolutionNode, self.currentPronoun, node], TruthValue(1.0, 100))
        rejected = False
        filterNumber=-1

        self.checkConjunctions(node)

        start=1
        end=self.numOfFilters+1

        '''
        For debugging purposes.
        '''
        if filter!=-1:
            start=filter
            end=filter+1

        for index in range(start,end):
            command='(cog-execute! filter-#'+str(index)+')'
            rv=self.bindLinkExe(self.currentProposal,node,command)
            if len(rv)>0:
                '''
                Reject it
                '''
                rejected = True
                filterNumber=index
                break

        if not rejected:

            # We don't want to output this to unit tests
            if self.DEBUG:
                    print("accepted "+node.name)
            log.fine("accepted "+node.name)
            self.generateReferenceLink(self.currentPronoun,node,TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
            self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
        else:
            self.generateReferenceLink(self.currentPronoun,node,TV_FOR_FILTERED_OUT_ANTECEDENTS)
            #if self.DEBUG:
                   # print("rejected "+node.name+" by filter-#"+str(index))

        self.atomspace.remove(self.currentResolutionLink_pronoun)
        return not rejected

# From agents/hobbs.py
def Checked(self,node):

        '''
        Since graph is not necessarily a forest, this agent actually does a Breadth-First search on a general graph for
        each pronoun, so we need to avoid cycling around the graph by marking each node as checked if we have visited it once.
        '''

        if node.name in self.checked:
            return True
        self.checked[node.name]=True
        return False

# From agents/hobbs.py
def bfs(self,node):

        '''
        Does a Breadth-First search, starts with "node"
        '''

        '''
        rv is used for unit tests
        '''
        rv=[]

        if node==None:
            #print("found you bfs")
            return
        q=queue.Queue()
        q.put(node)
        while not q.empty():
            front=q.get()
            rv.append(front)
            self.propose(front)
            children=self.getChildren(front)
            if len(children)>0:
                for node in children:
                    if not self.Checked(node):
                        q.put(node)
        return rv

# From agents/hobbs.py
def getWords(self):

        '''
        Returns a list of words in the atomspace
        '''

        rv=self.bindLinkExe(None,None,'(cog-execute! getWords)')
        return self.sortNodes(rv,self.getWordNumber)

# From agents/hobbs.py
def getTargets(self,words):

        '''
        Returns a list of references needed to be resolved.
        '''

        targets=[]
        for word in words:
            matched=False
            for index in range(1,self.numOfPrePatterns+1):
                command='(cog-execute! pre-process-#'+str(index)+')'
                rv=self.bindLinkExe(self.currentTarget,word,command)
                if len(rv)>0:
                    matched=True
                    break
            if matched:
                targets.append(word)
        return targets

# From agents/hobbs.py
def getPronouns(self):
        rv=self.bindLinkExe(None,None,'(cog-execute! getPronouns)')
        return self.sortNodes(rv,self.getWordNumber)

# From agents/hobbs.py
def getRoots(self):

        '''
        Return a list of roots(incoming degree of 0)
        '''

        self.bindLinkExe(None,None,'(cog-execute! connectRootsToParseNodes)')
        rv= self.bindLinkExe(None,None,'(cog-execute! getAllParseNodes)')
        return self.sortNodes(rv,self.getSentenceNumber)

# From agents/hobbs.py
def getRootOfNode(self,target):
        '''
        Returns a ParseNode associated with the "target"
        '''

        rv=self.bindLinkExe(self.currentTarget,target,'(cog-execute! getParseNode)')
        return rv[0]

# From agents/hobbs.py
def  previousRootExist(self,root):

        '''
        "previous" means that a root with smaller word sequence number than the word sequence number of current "roots".
        '''
        return not self.roots[0].name==root.name

# From agents/hobbs.py
def getPrevious(self,root):

        '''
        Return a previous root.
        '''

        rootNumber=self.getSentenceNumber(root)
        for root in reversed(self.roots):
            number=self.getSentenceNumber(root)
            if number<rootNumber:
                return root

# From agents/hobbs.py
def getAllNumberNodes(self):

        '''
        Finds word sequence number for each word
        '''

        rv= self.bindLinkExe(None, None, '(cog-execute!  getAllNumberNodes)')
        for link in rv:
            out=link.out
            if out[0].type==types.WordInstanceNode:
                self.wordNumber[out[0].name]=self.StringToNumber(out[1].name)

# From agents/hobbs.py
def initilization(self,atomspace):
        '''
        Initializes necessary variables. Loads rules.
        '''

        self.atomspace = atomspace

        self.PleonasticItNode=atomspace.add_node(types.AnchorNode, 'Pleonastic-it', TruthValue(1.0, 100))
        self.currentPronounNode = atomspace.add_node(types.AnchorNode, 'CurrentPronoun', TruthValue(1.0, 100))
        self.currentTarget = atomspace.add_node(types.AnchorNode, 'CurrentTarget', TruthValue(1.0, 100))
        self.currentProposal = atomspace.add_node(types.AnchorNode, 'CurrentProposal', TruthValue(1.0, 100))
        self.resolvedReferences=atomspace.add_node(types.AnchorNode, 'Resolved references', TruthValue(1.0, 100))
        self.currentResolutionNode=atomspace.add_node(types.AnchorNode, 'CurrentResolution', TruthValue(1.0, 100))
        self.pronounNumber = -1

        data=["opencog/nlp/anaphora/rules/getChildren.scm",
              "opencog/nlp/anaphora/rules/getNumberNode_WordInstanceNode.scm",
              "opencog/nlp/anaphora/rules/getNumberNode_ParseNode.scm",
              "opencog/nlp/anaphora/rules/connectRootsToParseNodes.scm",
              "opencog/nlp/anaphora/rules/getAllNumberNodes.scm",
              "opencog/nlp/anaphora/rules/getAllParseNodes.scm",
              "opencog/nlp/anaphora/rules/getConjunction.scm",
              "opencog/nlp/anaphora/rules/getParseNode.scm",
              "opencog/nlp/anaphora/rules/getWords.scm",
              "opencog/nlp/anaphora/rules/isIt.scm",

              "opencog/nlp/anaphora/rules/filters/filter-#1.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#2.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#3.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#4.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#5.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#6.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#7.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#8.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#9.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#10.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#11.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#12.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#13.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#14.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#15.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#16.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#17.scm",
              "opencog/nlp/anaphora/rules/filters/filter-#18.scm",

              "opencog/nlp/anaphora/rules/pre-process/pre-process-#1.scm",
              "opencog/nlp/anaphora/rules/pre-process/pre-process-#2.scm",
              "opencog/nlp/anaphora/rules/pre-process/pre-process-#3.scm",

              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#1.scm",
              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#2.scm",
              "opencog/nlp/anaphora/rules/pleonastic-it/pleonastic-it-#3.scm",

              ]

        self.numOfFilters=18
        self.numOfPrePatterns=3
        self.numOfPleonasticItPatterns=3

        for item in data:
            load_scm(atomspace, item)

        self.getAllNumberNodes()
        self.pronouns=self.getTargets(self.getWords())
        self.roots = self.getRoots()

# From agents/hobbs.py
def addPronounToResolvedList(self,node):
        '''
        Mark current pronoun as resolved.
        '''

        self.atomspace.add_link(types.ListLink,[self.resolvedReferences,node],TruthValue(1.0, 100))

# From agents/hobbs.py
def pleonastic_it(self,node):
        '''
        Check if the node is the word "it".
        '''
        matched=False
        rv=self.bindLinkExe(self.currentTarget,node,'(cog-execute! isIt)')
        if len(rv)>0:
            for index in range(1,self.numOfPleonasticItPatterns+1):
                command='(cog-execute! pleonastic-it-#'+str(index)+')'
                rv=self.bindLinkExe(self.currentTarget,node,command)
                if len(rv)>0:
                    matched=True
                    break
                #print("rejected "+node.name+" by filter-#"+str(index))

        return matched

# From agents/hobbs.py
def run(self, atomspace):
        self.initilization(atomspace)

        for pronoun in self.pronouns:


            self.checked.clear()
            self.pronounNumber=self.getWordNumber(pronoun)
            self.confidence=1-CONFIDENCE_DECREASING_RATE

            '''
            Binds current "pronoun" with "currentPronounNode".
            This part is used by pattern matcher.
            '''

            tmpLink=self.atomspace.add_link(types.ListLink, [self.currentPronounNode, pronoun], TruthValue(1.0, 100))
            self.currentPronoun=pronoun
            root=self.getRootOfNode(pronoun)

            if self.DEBUG:
                print("Resolving....")
                print(pronoun)
            log.fine("Resolving \n{0}".format(pronoun))

            '''
            Check if it's a pleonastic it.
            '''

            if self.pleonastic_it(pronoun):
                self.generateReferenceLink(pronoun,self.PleonasticItNode,TruthValue(STRENGTH_FOR_ACCEPTED_ANTECEDENTS, self.confidence))
                self.confidence=self.confidence*CONFIDENCE_DECREASING_RATE
                if self.DEBUG:
                    print("accepted "+self.PleonasticItNode.name)
                    log.fine("accepted "+self.PleonasticItNode.name)

            sent_counter=1;
            while True:
                if root==None:
                    break
                self.bfs(root)
                if self.previousRootExist(root) and sent_counter<=NUMBER_OF_SEARCHING_SENTENCES:
                    root=self.getPrevious(root)
                    sent_counter=sent_counter+1
                else:
                    break
            self.atomspace.remove(tmpLink)
            self.addPronounToResolvedList(pronoun)


# From agents/dumpAgent.py
class dumpAgent(MindAgent):
    
    def run(self, atomspace):
        with open('/tmp/log.txt', 'w') as logfile:
            all_atoms = atomspace.get_atoms_by_type(t=types.Atom)
            for atom in all_atoms:
                print(atom, file=logfile)

import unittest
from unittest import TestCase
from opencog.type_constructors import *
from opencog.utilities import pop_default_atomspace
from opencog.execute import execute_atom
from opencog.openpsi import *
import __main__

# From openpsi/openpsi_test.py
class OpenPsiTest(TestCase):

    @classmethod
    def setUpClass(cls):
        global atomspace
        atomspace = AtomSpace()
        set_default_atomspace(atomspace)

    @classmethod
    def tearDownClass(cls):
        pop_default_atomspace()
        global atomspace
        del atomspace

    def test_create_rule(self):
        openpsi = OpenPsi(atomspace)

        goal = openpsi.create_goal("goal")

        context = [
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("apple")),
            AbsentLink(
                InheritanceLink(
                    VariableNode("$APPLE"),
                    ConceptNode("handled")))
        ]

        action = ExecutionOutputLink(
            GroundedSchemaNode("py: eat_apple"),
            ListLink(
                VariableNode("$APPLE")))

        component = openpsi.create_component("test-component")

        rule = openpsi.add_rule(context, action, goal, TruthValue(1.0, 1.0), component)

        self.assertFalse(openpsi.is_rule(goal))
        self.assertTrue(openpsi.is_rule(rule.get_rule_atom()))
        self.assertEqual(ConceptNode("goal"), rule.get_goal())

        categories = openpsi.get_categories()
        print("duuuude its cat=", categories)
        self.assertEqual(2, len(categories))
        self.assertTrue(component in categories)
        self.assertFalse(ConceptNode("new-category") in categories)

        new_category = openpsi.add_category(ConceptNode("new-category"))
        self.assertEqual(ConceptNode("new-category"), new_category)
        categories = openpsi.get_categories()
        self.assertEqual(3, len(categories))
        self.assertTrue(new_category in categories)

        self.assertEqual(context, rule.get_context())
        self.assertEqual(action, rule.get_action())

    def test_run_openpsi(self):
        openpsi = OpenPsi(atomspace)

        goal = openpsi.create_goal("goal-run")

        context = [
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("apple")),
            AbsentLink(
                InheritanceLink(
                    VariableNode("$APPLE"),
                    ConceptNode("handled")))
        ]

        action = ExecutionOutputLink(
            GroundedSchemaNode("py: eat_apple"),
            ListLink(
                VariableNode("$APPLE")))

        component = openpsi.create_component("test-component-run")

        openpsi.add_rule(context, action, goal, TruthValue(1.0, 1.0), component)

        openpsi.run(component)

        # Apples are handled by OpenPsi loop
        InheritanceLink(ConceptNode("apple-1"), ConceptNode("apple"))
        InheritanceLink(ConceptNode("apple-2"), ConceptNode("apple"))

        delay = 5
        time.sleep(delay)
        openpsi.halt(component)

        handled_apples = GetLink(
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("handled")))

        result_set = execute_atom(atomspace, handled_apples)
        result1 = result_set.out[0]
        result2 = result_set.out[1]
        self.assertEqual(result1, ConceptNode("apple-1"))
        self.assertEqual(result2, ConceptNode("apple-2"))

# From openpsi/openpsi_test.py
def eat_apple(apple):
    # Mark apple as handled
    InheritanceLink(apple, ConceptNode("handled"))
    return ConceptNode("finished")

# From openpsi/openpsi_test.py
def setUpClass(cls):
        global atomspace
        atomspace = AtomSpace()
        set_default_atomspace(atomspace)

# From openpsi/openpsi_test.py
def tearDownClass(cls):
        pop_default_atomspace()
        global atomspace
        del atomspace

# From openpsi/openpsi_test.py
def test_create_rule(self):
        openpsi = OpenPsi(atomspace)

        goal = openpsi.create_goal("goal")

        context = [
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("apple")),
            AbsentLink(
                InheritanceLink(
                    VariableNode("$APPLE"),
                    ConceptNode("handled")))
        ]

        action = ExecutionOutputLink(
            GroundedSchemaNode("py: eat_apple"),
            ListLink(
                VariableNode("$APPLE")))

        component = openpsi.create_component("test-component")

        rule = openpsi.add_rule(context, action, goal, TruthValue(1.0, 1.0), component)

        self.assertFalse(openpsi.is_rule(goal))
        self.assertTrue(openpsi.is_rule(rule.get_rule_atom()))
        self.assertEqual(ConceptNode("goal"), rule.get_goal())

        categories = openpsi.get_categories()
        print("duuuude its cat=", categories)
        self.assertEqual(2, len(categories))
        self.assertTrue(component in categories)
        self.assertFalse(ConceptNode("new-category") in categories)

        new_category = openpsi.add_category(ConceptNode("new-category"))
        self.assertEqual(ConceptNode("new-category"), new_category)
        categories = openpsi.get_categories()
        self.assertEqual(3, len(categories))
        self.assertTrue(new_category in categories)

        self.assertEqual(context, rule.get_context())
        self.assertEqual(action, rule.get_action())

# From openpsi/openpsi_test.py
def test_run_openpsi(self):
        openpsi = OpenPsi(atomspace)

        goal = openpsi.create_goal("goal-run")

        context = [
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("apple")),
            AbsentLink(
                InheritanceLink(
                    VariableNode("$APPLE"),
                    ConceptNode("handled")))
        ]

        action = ExecutionOutputLink(
            GroundedSchemaNode("py: eat_apple"),
            ListLink(
                VariableNode("$APPLE")))

        component = openpsi.create_component("test-component-run")

        openpsi.add_rule(context, action, goal, TruthValue(1.0, 1.0), component)

        openpsi.run(component)

        # Apples are handled by OpenPsi loop
        InheritanceLink(ConceptNode("apple-1"), ConceptNode("apple"))
        InheritanceLink(ConceptNode("apple-2"), ConceptNode("apple"))

        delay = 5
        time.sleep(delay)
        openpsi.halt(component)

        handled_apples = GetLink(
            InheritanceLink(
                VariableNode("$APPLE"),
                ConceptNode("handled")))

        result_set = execute_atom(atomspace, handled_apples)
        result1 = result_set.out[0]
        result2 = result_set.out[1]
        self.assertEqual(result1, ConceptNode("apple-1"))
        self.assertEqual(result2, ConceptNode("apple-2"))


# From openpsi/ping.py
def ping():
    time.sleep(5)
    print("\nJust pinged\n")
    return StateLink(ball, pinged)

from ping import *

# From openpsi/ping_pong.py
def pong():
    time.sleep(1)
    print("\nJust ponged\n")
    StateLink(ball, ponged)
    # The side-effect of the action decreases the urge.
    return op.decrease_urge(pong_goal, 1)

# From openpsi/ping_pong.py
def pong_step():
    time.sleep(3)
    urge = op.get_urge(pong_goal)
    if urge < 0.7:
        print("\nNot yet feeling like ponging the ball. Urge = %f\n" % urge)
        op.increase_urge(pong_goal, 0.2)
    else:
        print("\nFeeling like ponging the ball. Urge = %f\n" % urge)
        op.increase_urge(pong_goal, 0.2)
        op.step(ConceptNode("pong"))

    return TruthValue(1, 1)

# From openpsi/ping_pong.py
def pong_action_selector():
    return op.get_satisfiable_rules(pong_component)

from openagi.llms.azure import AzureChatOpenAIModel
from openagi.agent import Admin
from openagi.memory import Memory
from openagi.worker import Worker
from openagi.planner.task_decomposer import TaskPlanner
from openagi.actions.base import BaseAction
import wikipedia
import joblib
import requests
import string
from tqdm import tqdm
from collections import Counter
from pydantic import Field
from pydantic import validator
import numpy

# From openagi/benchmark.py
class WikiSearchAction(BaseAction):
    """
    Use this Action to get the information from Wikipedia Search
    """
    query: str = Field(
        default_factory=str,
        description="The search string. Be simple."
    )

    @validator('query')
    def validate_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Query must be a non-empty string.')
        return v

    def execute(self):
        try:
            search_res = wikipedia.search(self.query)
            if not search_res:
                return 'No results found.'
            article = wikipedia.page(search_res[0])
            return article.summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation error: {str(e)}"
        except wikipedia.exceptions.PageError as e:
            return f"Page error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

# From openagi/benchmark.py
def download_file(url, filename):
    """
    Download a file from a URL and save it locally.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {str(e)}")

# From openagi/benchmark.py
def load_hotpot_qa_data(level):
    """
    Load HotpotQA data for a given level. If data doesn't exist, download it.
    """
    file_path = f"./data/{level}.joblib"
    data_url = f"https://github.com/salesforce/BOLAA/raw/main/hotpotqa_run/data/{level}.joblib"

    if not os.path.exists(file_path):
        print(f"{level} data not found, downloading...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        download_file(data_url, file_path)
    return joblib.load(file_path)

# From openagi/benchmark.py
def normalize_answer(s):
    """
    Normalize answers for evaluation.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# From openagi/benchmark.py
def f1_score(prediction, ground_truth):
    """
    Compute the F1 score between prediction and ground truth answers.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

# From openagi/benchmark.py
def agent(query, llm):
    planner = TaskPlanner(autonomous=True)
    admin = Admin(
        planner=planner,
        memory=Memory(),
        actions=[WikiSearchAction],
        llm=llm,
    )
    res = admin.run(
        query=query,
        description="Provide answer for the query. You should decompose your task into executable actions.",
    )
    return res

# From openagi/benchmark.py
def run_agent(level = 'easy'):
    os.environ["AZURE_BASE_URL"] = ""
    os.environ["AZURE_DEPLOYMENT_NAME"] = ""
    os.environ["AZURE_MODEL_NAME"]="gpt4"
    os.environ["AZURE_OPENAI_API_VERSION"]=""
    os.environ["AZURE_OPENAI_API_KEY"]=  ""
    config = AzureChatOpenAIModel.load_from_env_config()
    llm = AzureChatOpenAIModel(config=config)

    hotpot_data = load_hotpot_qa_data(level)
    hotpot_data = hotpot_data.reset_index(drop=True)
    task_instructions = [
        (row["question"], row["answer"]) for _, row in hotpot_data.iterrows()
    ]

    f1_list = []
    correct = 0
    results = {}

    for task , answer in tqdm(task_instructions[0:30]):
        response = agent(task , llm)
        f1 , _ ,_ = f1_score(response,answer)
        f1_list.append(f1)
        correct += int(response == answer)

        avg_f1 = np.mean(f1_list)
        acc = correct / len(task_instructions[0:30])
    return avg_f1, acc

# From openagi/benchmark.py
def validate_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Query must be a non-empty string.')
        return v

# From openagi/benchmark.py
def execute(self):
        try:
            search_res = wikipedia.search(self.query)
            if not search_res:
                return 'No results found.'
            article = wikipedia.page(search_res[0])
            return article.summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation error: {str(e)}"
        except wikipedia.exceptions.PageError as e:
            return f"Page error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

# From openagi/benchmark.py
def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

# From openagi/benchmark.py
def white_space_fix(text):
        return " ".join(text.split())

# From openagi/benchmark.py
def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

# From openagi/benchmark.py
def lower(text):
        return text.lower()

from openagi.actions.tools.document_loader import TextLoaderTool
from openagi.llms.xai import XAIModel

from openagi.actions.tools.tavilyqasearch import TavilyWebSearchQA
from openagi.llms.mistral import MistralModel
from getpass import getpass

from openagi.llms.gemini import GeminiModel

from openagi.actions.tools.searchapi_search import SearchApiSearch

from openagi.actions.files import WriteFileAction
from openagi.actions.files import ReadFileAction
from openagi.actions.tools.ddg_search import DuckDuckGoSearch
from openagi.actions.tools.webloader import WebBaseContextTool
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv


from openagi.actions.tools.youtubesearch import YouTubeSearchTool

from openagi.actions.tools.ddg_search import DuckDuckGoNewsSearch






from enum import Enum
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from pydantic import BaseModel
from pydantic import field_validator
from openagi.actions.compressor import SummarizerAction
from openagi.actions.formatter import FormatterAction
from openagi.actions.obs_rag import MemoryRagAction
from openagi.actions.utils import run_action
from openagi.exception import OpenAGIException
from openagi.llms.azure import LLMBaseModel
from openagi.memory.memory import Memory
from openagi.planner.task_decomposer import BasePlanner
from openagi.prompts.worker_task_execution import WorkerAgentTaskExecution
from openagi.tasks.lists import TaskLists
from openagi.utils.extraction import find_last_r_failure_content
from openagi.utils.extraction import get_act_classes_from_json
from openagi.utils.extraction import get_last_json
from openagi.utils.helper import get_default_llm
from openagi.utils.tool_list import get_tool_list
from openagi.memory.sessiondict import SessionDict
from openagi.actions.human_input import HumanCLIInput
from openagi.prompts.ltm import LTMFormatPrompt

# From openagi/agent.py
class OutputFormat(str, Enum):
    markdown = "markdown"
    raw_text = "raw_text"

# From openagi/agent.py
class Admin(BaseModel):
    planner: Optional[BasePlanner] = Field(
        description="Type of planner to use for task decomposition.",
        default=None,
    )
    llm: Optional[LLMBaseModel] = Field(
        description="LLM Model to be used.",
        default=None,
    )
    memory: Optional[Memory] = Field(
        default_factory=list, description="Memory to be used.", exclude=True
    )
    actions: Optional[List[Any]] = Field(
        description="Actions that the Agent supports", default_factory=list
    )
    max_iterations: int = Field(
        default=20, description="Maximum number of steps to achieve the objective."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.markdown,
        description="Format to be converted the result while returning.",
    )
    workers: List[Worker] = Field(
        default_factory=list,
        description="List of workers managed by the Admin agent.",
    )
    summarize_task_context: bool = Field(
        default=True,
        description="If set to True, the task context will be summarized and passed to the next task else the task context will be passed as is.",
    )
    output_key: str = Field(
        default="final_output",
        description="Key to be used to store the output.",
    )

    input_action: Optional[HumanCLIInput] = Field(default_factory=HumanCLIInput,
                                               description="To get feedback in case long term memory has been enabled")

    def model_post_init(self, __context: Any) -> None:
        model = super().model_post_init(__context)

        if not self.llm:
            self.llm = get_default_llm()

        if not self.planner:
            self.planner = TaskPlanner(workers=self.workers)

        if not self.memory:
            self.memory = Memory()

        self.actions = self.actions or []

        default_actions = [MemoryRagAction]
        self.actions.extend(default_actions)

        return model

    @field_validator("actions")
    @classmethod
    def actions_validator(cls, act_clss):
        for act_cls in act_clss:
            if not issubclass(act_cls, BaseAction):
                raise ValueError(f"{act_cls} is not a subclass of BaseAction")
        return act_clss

    def assign_workers(self, workers: List[Worker]):
        if workers:
            for worker in workers:
                if not getattr(worker, "llm", False):
                    setattr(worker, "llm", self.llm)
                if not getattr(worker, "memory", False):
                    setattr(worker, "memory", self.memory)

        if not self.workers:
            self.workers = workers
        else:
            self.workers.extend(workers)

    def run_planner(self, query: str, description: str, long_term_context: str):
        if self.planner:
            if not getattr(self.planner, "llm", False):
                setattr(self.planner, "llm", self.llm)

            setattr(self.planner, "workers", self.workers)

        logging.info("Thinking...")
        actions_dict: List[BaseAction] = []

        for act in self.actions:
            actions_dict.append(act.cls_doc())

        workers_dict = []
        for worker in self.workers:
            workers_dict.append(worker.worker_doc())
            for action in worker.actions:
                actions_dict.append(action.cls_doc())

        return self.planner.plan(
            query=query,
            description=description,
            long_term_context=long_term_context,
            supported_actions=actions_dict,
            supported_workers=workers_dict,
        )

    def _generate_tasks_list(self, planned_tasks):
        task_lists = TaskLists()
        task_lists.add_tasks(tasks=planned_tasks)
        logging.debug(f"Created {task_lists.get_tasks_queue().qsize()} Tasks.")
        return task_lists

    def get_previous_task_contexts(self, task_lists: TaskLists):
        task_summaries = []
        logging.info("Retrieving completed task contexts...")
        t_list = task_lists.completed_tasks.queue
        for indx, task in enumerate(t_list):
            memory = run_action(
                action_cls=MemoryRagAction,
                task=task,
                llm=self.llm,
                memory=self.memory,
                query=task.id,
            )
            if memory and self.summarize_task_context:
                params = {
                    "past_messages": memory,
                    "llm": self.llm,
                    "memory": self.memory,
                    "instructions": "Include summary of all the thoughts, but include all the relevant points from the observations without missing any.",
                }
                memory = run_action(action_cls=SummarizerAction, **params)
                if not memory:
                    raise Exception("No memory returned after summarization.")
            task_summaries.append(f"\n{indx+1}. {task.name} - {task.description}\n{memory}")
        else:
            logging.warning("No Tasks to summarize.")
        if task_summaries:
            return "\n".join(task_summaries).strip()
        return "None"

    def _get_worker_by_id(self, worker_id: str):
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
        raise ValueError(f"Worker with id {worker_id} not found.")

    def worker_task_execution(self, query: str, description: str, task_lists: TaskLists):
        res = None

        while not task_lists.all_tasks_completed:
            cur_task = task_lists.get_next_unprocessed_task()
            worker = self._get_worker_by_id(cur_task.worker_id)
            res, task = worker.execute_task(
                cur_task,
                context=self.get_previous_task_contexts(task_lists=task_lists),
            )
            self.memory.update_task(task)
            task_lists.add_completed_tasks(task)

        logging.info("Finished Execution...")

        if self.output_format == OutputFormat.markdown and res:
            logging.info("Output Formatting...")
            output_formatter = FormatterAction(
                content=res,
                format_type=OutputFormat.markdown,
                llm=self.llm,
                memory=self.memory,
            )
            res = output_formatter.execute()
        logging.debug(f"Execution Completed for Session ID - {self.memory.session_id}")
        return res

    def _provoke_thought_obs(self, observation):
        thoughts = dedent(f"""Observation: {observation}""".strip())
        return thoughts

    def _should_continue(self, llm_resp: str) -> Tuple[bool, Optional[Dict]]:
        output: Dict = get_last_json(llm_resp, llm=self.llm, max_iterations=self.max_iterations)
        output_key_exists = bool(output and output.get(self.output_key))
        return (not output_key_exists, output)

    def _force_output(
        self, llm_resp: str, all_thoughts_and_obs: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Force the output once the max iterations are reached."""
        prompt = (
            "\n".join(all_thoughts_and_obs)
            + "Based on the previous action and observation, give me the output."
        )
        output = self.llm.run(prompt)
        cont, final_output = self._should_continue(output)
        if cont:
            prompt = (
                "\n".join(all_thoughts_and_obs)
                + f"Based on the previous action and observation, give me the output. {final_output}"
            )
            output = self.llm.run(prompt)
            cont, final_output = self._should_continue(output)
        if cont:
            raise OpenAGIException(
                f"LLM did not produce the expected output after {self.max_iterations} iterations."
            )
        return (cont, final_output)

    def auto_workers_assignment(self, query: str, description: str, task_lists: TaskLists):
        """
        Autonomously generates the Workers with the

        Args:
            query (str): The query to be processed.
            description (str): A description of the task.
            task_lists (TaskLists): The task lists to be processed.

        Returns:
            str: JSON of the list of Workers that needs to be executed
        """

        workers = []
        tools_list = get_tool_list()
        
        for action in self.actions:
            tools_list.append(action)

        worker_dict = {}
        main_task_list = TaskLists()
        while not task_lists.all_tasks_completed:
            cur_task = task_lists.get_next_unprocessed_task()
            print(cur_task)
            logging.info(f"**** Executing Task - {cur_task.name} [{cur_task.id}] ****")

            worker_config = cur_task.worker_config

            worker_instance = None
            if worker_config["role"] not in worker_dict:
                worker_instance = Worker(
                    role=worker_config["role"],
                    instructions=worker_config["instructions"],
                    llm=self.llm,
                    actions=self.get_supported_actions_for_worker(
                        worker_config["supported_actions"],tools_list
                    ),
                )
                worker_dict[worker_config["role"]] = worker_instance
            else:
                worker_instance = worker_dict[worker_config["role"]]
            workers.append(worker_instance)
            cur_task.worker_id = worker_instance.id
            main_task_list.add_task(cur_task)

        task_lists = main_task_list
        self.assign_workers(workers=workers)

        if self.workers:
            return self.worker_task_execution(
                query=query,
                description=description,
                task_lists=task_lists,
            )

    def single_agent_execution(self, query: str, description: str, task_lists: TaskLists):
        """
        Executes a single agent's tasks for the given query and description, updating the task lists and memory as necessary.

        Args:
            query (str): The query to be processed.
            description (str): A description of the task.
            task_lists (TaskLists): The task lists to be processed.

        Returns:
            str: The final result of the task execution.
        """
        all_thoughts_and_obs = []
        output = None
        previous_task_context = None

        while not task_lists.all_tasks_completed:
            iteration = 1
            max_iterations = self.max_iterations

            cur_task = task_lists.get_next_unprocessed_task()
            logging.info(f"**** Executing Task - {cur_task.name} [{cur_task.id}] ****")

            task_to_execute = f"{cur_task.name}. {cur_task.description}"
            agent_description = "Task executor"

            logging.debug("Provoking initial thought observation...")
            initial_thought_provokes = self._provoke_thought_obs(None)

            te_vars = dict(
                task_to_execute=task_to_execute,
                worker_description=agent_description,
                supported_actions=[action.cls_doc() for action in self.actions],
                thought_provokes=initial_thought_provokes,
                output_key=self.output_key,
                context=previous_task_context,
                max_iterations=max_iterations,
            )

            logging.debug("Generating base prompt...")
            base_prompt = WorkerAgentTaskExecution().from_template(te_vars)
            prompt = f"{base_prompt}\nThought:\nIteration: {iteration}\nActions:\n"

            logging.debug("Running LLM with prompt...")
            observations = self.llm.run(prompt)
            logging.info(f"LLM execution completed. Observations: {observations}")
            all_thoughts_and_obs.append(prompt)

            while iteration < max_iterations:
                logging.info(f"---- Iteration {iteration} ----")
                continue_flag, output = self._should_continue(observations)

                if not continue_flag:
                    logging.info(f"Task completed. Output: {output}")
                    break

                resp_json = get_last_json(observations)

                output = resp_json.get(self.output_key) if resp_json else None
                if output:
                    cur_task.result = output
                    cur_task.actions = te_vars["supported_actions"]
                    self.memory.update_task(cur_task)

                action_json = resp_json.get("action") if resp_json else None

                if action_json and not isinstance(action_json, list):
                    action_json = [action_json]

                if not action_json:
                    logging.warning(f"No action found in the output: {output}")
                    observations = f"Action: {action_json}\n{observations} Unable to extract action. Verify the output and try again."
                    all_thoughts_and_obs.append(observations)
                    iteration += 1
                elif action_json:
                    actions = get_act_classes_from_json(action_json)

                    for act_cls, params in actions:
                        params["previous_action"] = None  # Modify as needed
                        params["llm"] = self.llm
                        params["memory"] = self.memory
                        try:
                            logging.debug(f"Running action: {act_cls.__name__}...")
                            res = run_action(action_cls=act_cls, **params)
                            logging.info(f"Action '{act_cls.__name__}' completed. Result: {res}")
                        except Exception as e:
                            logging.error(f"Error running action: {e}")
                            observations = f"Action: {action_json}\n{observations}. {e} Try to fix the error and try again. Ignore if already tried more than twice"
                            all_thoughts_and_obs.append(observations)
                            iteration += 1
                            continue

                        observation_prompt = f"Observation: {res}\n"
                        all_thoughts_and_obs.append(observation_prompt)
                        observations = res

                    logging.debug("Provoking thought observation...")
                    thought_prompt = self._provoke_thought_obs(observations)
                    all_thoughts_and_obs.append(f"\n{thought_prompt}\nActions:\n")

                    prompt = f"{base_prompt}\n" + "\n".join(all_thoughts_and_obs)
                    logging.debug(f"\nSTART:{'*' * 20}\n{prompt}\n{'*' * 20}:END")
                    logging.debug("Running LLM with updated prompt...")
                    observations = self.llm.run(prompt)
                    iteration += 1
            else:
                if iteration == max_iterations:
                    logging.info("---- Forcing Output ----")
                    cont, final_output = self._force_output(observations, all_thoughts_and_obs)
                    if cont:
                        raise OpenAGIException(
                            f"LLM did not produce the expected output after {iteration} iterations for task {cur_task.name}"
                        )
                    output = final_output
                    cur_task.result = output
                    cur_task.actions = te_vars["supported_actions"]
                    self.memory.update_task(cur_task)
                    task_lists.add_completed_tasks(cur_task)

            previous_task_context = self.get_previous_task_contexts(task_lists)
            task_lists.add_completed_tasks(cur_task)

        logging.info("Finished Execution...")

        if self.output_format == OutputFormat.markdown:
            logging.info("Output Formatting...")
            output_formatter = FormatterAction(
                content=output,
                format_type=OutputFormat.markdown,
                llm=self.llm,
                memory=self.memory,
            )
            output = output_formatter.execute()

        logging.debug(f"Execution Completed for Session ID - {self.memory.session_id}")
        return output


    def run(self, query: str, description: str,planned_tasks: Optional[List[Dict]] = None):
        logging.info("Running Admin Agent...")
        logging.info(f"SessionID - {self.memory.session_id}")

        if self.memory.long_term and planned_tasks:
            logging.warning("Long Term Memory is not applicable for user given plan.")

        ltm = ["None"]
        bad_feedback = False
        bad_session = None
        if self.memory.long_term and not planned_tasks:
            logging.info("Retrieving similar queries from long term memory...")
            similar_sessions = self.memory.get_ltm(query)
            ltm = []
            for memory in similar_sessions:
                metadata = memory["metadata"]
                if memory["similarity_score"] >= self.memory.ltm_threshold:
                    if metadata["ans_feedback"]=='' and metadata["plan_feedback"]=='':
                        logging.info(f"Found a very similar query (similarity = {memory['similarity_score']} in long term memory without negative feedback, returning answer directly")
                        result = memory["document"]
                        # ask for feedback here and UPDATE the response
                        # write for case when threshold is crossed but negative feedback
                        session = SessionDict.from_dict(metadata)
                        self.save_ltm("update", session)
                        return result
                    else:
                        ltm.append(LTMFormatPrompt().base_prompt.format(**metadata))
                        bad_feedback = True
                        bad_session = SessionDict.from_dict(metadata)
                        break
                # ltm.append(LTMFormatPrompt().base_prompt.format(**metadata))
                # the above is commented because i think it is better to have a threshold on what gets retrieved
                # instead of relying on top k. This way we only retrieve one session though, but it should be a
                # good session.

        old_context = "\n\n".join(ltm)
        if not planned_tasks:
            planned_tasks = self.run_planner(query=query, description=description, long_term_context=old_context)


        logging.info("Tasks Planned...")
        logging.debug(f"{planned_tasks=}")

        task_lists: TaskLists = self._generate_tasks_list(planned_tasks=planned_tasks)

        self.memory.save_planned_tasks(tasks=list(task_lists.tasks.queue))

        if self.planner.autonomous:
            result = self.auto_workers_assignment(
                query=query, description=description, task_lists=task_lists
            )
        else:
            if self.workers:
                result = self.worker_task_execution(
                    query=query,
                    description=description,
                    task_lists=task_lists,
                )
            else:
                result = self.single_agent_execution(
                    query=query, description=description, task_lists=task_lists
                )
        # Human feedback part
        if self.memory.long_term:
            if bad_feedback:
                bad_session.plan = str(planned_tasks)
                bad_session.answer =  result
                self.save_ltm("update", bad_session)
            else:
                session = SessionDict(
                    query=query,
                    description=description,
                    plan=str(planned_tasks),
                    session_id=self.memory.session_id,
                    answer=result
                )
                self.save_ltm("add", session)
        return result

    def _can_task_execute(self, llm_resp: str) -> Union[bool, Optional[str]]:
        content: str = find_last_r_failure_content(text=llm_resp)
        if content:
            return False, content
        return True, content

    def get_supported_actions_for_worker(self, actions_list: List[str],tool_list: List[str]):
        """
        This function takes a list of action names (strings) and returns a list of class objects
        from the modules within the 'tools' folder that match these action names and inherit from BaseAction.

        :param actions_list: List of action names as strings.
        :return: List of matching class objects.
        """
        matching_classes = []
        #tool_list = get_tool_list()
        # Iterate through all modules in the tools package
        for action in tool_list:
            if action.__name__ in actions_list:
                matching_classes.append(action)

        return matching_classes

    def save_ltm(self, action_type: str, session: SessionDict):
        """
        Save a session to long-term memory by either adding or updating an existing session.

        :param action_type: Type of operation: 'add' or 'update'
        :param session: The SessionDict object containing session details
        """
        # Get feedback for plan and answer
        session.plan_feedback = self.input_action.execute(
            prompt=(
                f"Review the generated plan: \n{session.plan}\n"
                "If satisfied, press ENTER. \nOtherwise, describe the issue and suggest improvements:"
            )
        ).strip()

        session.ans_feedback = self.input_action.execute(
            prompt=(
                f"Review the generated answer: \n{session.answer}\n"
                "If satisfied, press ENTER. \nOtherwise, describe the issue and suggest improvements:"
            )
        ).strip()

        # Save or update based on the action_type
        if action_type == "add":
            self.memory.add_ltm(session)
            logging.info(f"Session added to long-term memory: {session}")
        elif action_type == "update":
            self.memory.update_ltm(session)
            logging.info(f"Session updated in long-term memory: {session}")
        else:
            raise ValueError("Invalid action_type. Use 'add' or 'update'.")

# From openagi/agent.py
def model_post_init(self, __context: Any) -> None:
        model = super().model_post_init(__context)

        if not self.llm:
            self.llm = get_default_llm()

        if not self.planner:
            self.planner = TaskPlanner(workers=self.workers)

        if not self.memory:
            self.memory = Memory()

        self.actions = self.actions or []

        default_actions = [MemoryRagAction]
        self.actions.extend(default_actions)

        return model

# From openagi/agent.py
def actions_validator(cls, act_clss):
        for act_cls in act_clss:
            if not issubclass(act_cls, BaseAction):
                raise ValueError(f"{act_cls} is not a subclass of BaseAction")
        return act_clss

# From openagi/agent.py
def assign_workers(self, workers: List[Worker]):
        if workers:
            for worker in workers:
                if not getattr(worker, "llm", False):
                    setattr(worker, "llm", self.llm)
                if not getattr(worker, "memory", False):
                    setattr(worker, "memory", self.memory)

        if not self.workers:
            self.workers = workers
        else:
            self.workers.extend(workers)

# From openagi/agent.py
def run_planner(self, query: str, description: str, long_term_context: str):
        if self.planner:
            if not getattr(self.planner, "llm", False):
                setattr(self.planner, "llm", self.llm)

            setattr(self.planner, "workers", self.workers)

        logging.info("Thinking...")
        actions_dict: List[BaseAction] = []

        for act in self.actions:
            actions_dict.append(act.cls_doc())

        workers_dict = []
        for worker in self.workers:
            workers_dict.append(worker.worker_doc())
            for action in worker.actions:
                actions_dict.append(action.cls_doc())

        return self.planner.plan(
            query=query,
            description=description,
            long_term_context=long_term_context,
            supported_actions=actions_dict,
            supported_workers=workers_dict,
        )

# From openagi/agent.py
def get_previous_task_contexts(self, task_lists: TaskLists):
        task_summaries = []
        logging.info("Retrieving completed task contexts...")
        t_list = task_lists.completed_tasks.queue
        for indx, task in enumerate(t_list):
            memory = run_action(
                action_cls=MemoryRagAction,
                task=task,
                llm=self.llm,
                memory=self.memory,
                query=task.id,
            )
            if memory and self.summarize_task_context:
                params = {
                    "past_messages": memory,
                    "llm": self.llm,
                    "memory": self.memory,
                    "instructions": "Include summary of all the thoughts, but include all the relevant points from the observations without missing any.",
                }
                memory = run_action(action_cls=SummarizerAction, **params)
                if not memory:
                    raise Exception("No memory returned after summarization.")
            task_summaries.append(f"\n{indx+1}. {task.name} - {task.description}\n{memory}")
        else:
            logging.warning("No Tasks to summarize.")
        if task_summaries:
            return "\n".join(task_summaries).strip()
        return "None"

# From openagi/agent.py
def worker_task_execution(self, query: str, description: str, task_lists: TaskLists):
        res = None

        while not task_lists.all_tasks_completed:
            cur_task = task_lists.get_next_unprocessed_task()
            worker = self._get_worker_by_id(cur_task.worker_id)
            res, task = worker.execute_task(
                cur_task,
                context=self.get_previous_task_contexts(task_lists=task_lists),
            )
            self.memory.update_task(task)
            task_lists.add_completed_tasks(task)

        logging.info("Finished Execution...")

        if self.output_format == OutputFormat.markdown and res:
            logging.info("Output Formatting...")
            output_formatter = FormatterAction(
                content=res,
                format_type=OutputFormat.markdown,
                llm=self.llm,
                memory=self.memory,
            )
            res = output_formatter.execute()
        logging.debug(f"Execution Completed for Session ID - {self.memory.session_id}")
        return res

# From openagi/agent.py
def auto_workers_assignment(self, query: str, description: str, task_lists: TaskLists):
        """
        Autonomously generates the Workers with the

        Args:
            query (str): The query to be processed.
            description (str): A description of the task.
            task_lists (TaskLists): The task lists to be processed.

        Returns:
            str: JSON of the list of Workers that needs to be executed
        """

        workers = []
        tools_list = get_tool_list()
        
        for action in self.actions:
            tools_list.append(action)

        worker_dict = {}
        main_task_list = TaskLists()
        while not task_lists.all_tasks_completed:
            cur_task = task_lists.get_next_unprocessed_task()
            print(cur_task)
            logging.info(f"**** Executing Task - {cur_task.name} [{cur_task.id}] ****")

            worker_config = cur_task.worker_config

            worker_instance = None
            if worker_config["role"] not in worker_dict:
                worker_instance = Worker(
                    role=worker_config["role"],
                    instructions=worker_config["instructions"],
                    llm=self.llm,
                    actions=self.get_supported_actions_for_worker(
                        worker_config["supported_actions"],tools_list
                    ),
                )
                worker_dict[worker_config["role"]] = worker_instance
            else:
                worker_instance = worker_dict[worker_config["role"]]
            workers.append(worker_instance)
            cur_task.worker_id = worker_instance.id
            main_task_list.add_task(cur_task)

        task_lists = main_task_list
        self.assign_workers(workers=workers)

        if self.workers:
            return self.worker_task_execution(
                query=query,
                description=description,
                task_lists=task_lists,
            )

# From openagi/agent.py
def single_agent_execution(self, query: str, description: str, task_lists: TaskLists):
        """
        Executes a single agent's tasks for the given query and description, updating the task lists and memory as necessary.

        Args:
            query (str): The query to be processed.
            description (str): A description of the task.
            task_lists (TaskLists): The task lists to be processed.

        Returns:
            str: The final result of the task execution.
        """
        all_thoughts_and_obs = []
        output = None
        previous_task_context = None

        while not task_lists.all_tasks_completed:
            iteration = 1
            max_iterations = self.max_iterations

            cur_task = task_lists.get_next_unprocessed_task()
            logging.info(f"**** Executing Task - {cur_task.name} [{cur_task.id}] ****")

            task_to_execute = f"{cur_task.name}. {cur_task.description}"
            agent_description = "Task executor"

            logging.debug("Provoking initial thought observation...")
            initial_thought_provokes = self._provoke_thought_obs(None)

            te_vars = dict(
                task_to_execute=task_to_execute,
                worker_description=agent_description,
                supported_actions=[action.cls_doc() for action in self.actions],
                thought_provokes=initial_thought_provokes,
                output_key=self.output_key,
                context=previous_task_context,
                max_iterations=max_iterations,
            )

            logging.debug("Generating base prompt...")
            base_prompt = WorkerAgentTaskExecution().from_template(te_vars)
            prompt = f"{base_prompt}\nThought:\nIteration: {iteration}\nActions:\n"

            logging.debug("Running LLM with prompt...")
            observations = self.llm.run(prompt)
            logging.info(f"LLM execution completed. Observations: {observations}")
            all_thoughts_and_obs.append(prompt)

            while iteration < max_iterations:
                logging.info(f"---- Iteration {iteration} ----")
                continue_flag, output = self._should_continue(observations)

                if not continue_flag:
                    logging.info(f"Task completed. Output: {output}")
                    break

                resp_json = get_last_json(observations)

                output = resp_json.get(self.output_key) if resp_json else None
                if output:
                    cur_task.result = output
                    cur_task.actions = te_vars["supported_actions"]
                    self.memory.update_task(cur_task)

                action_json = resp_json.get("action") if resp_json else None

                if action_json and not isinstance(action_json, list):
                    action_json = [action_json]

                if not action_json:
                    logging.warning(f"No action found in the output: {output}")
                    observations = f"Action: {action_json}\n{observations} Unable to extract action. Verify the output and try again."
                    all_thoughts_and_obs.append(observations)
                    iteration += 1
                elif action_json:
                    actions = get_act_classes_from_json(action_json)

                    for act_cls, params in actions:
                        params["previous_action"] = None  # Modify as needed
                        params["llm"] = self.llm
                        params["memory"] = self.memory
                        try:
                            logging.debug(f"Running action: {act_cls.__name__}...")
                            res = run_action(action_cls=act_cls, **params)
                            logging.info(f"Action '{act_cls.__name__}' completed. Result: {res}")
                        except Exception as e:
                            logging.error(f"Error running action: {e}")
                            observations = f"Action: {action_json}\n{observations}. {e} Try to fix the error and try again. Ignore if already tried more than twice"
                            all_thoughts_and_obs.append(observations)
                            iteration += 1
                            continue

                        observation_prompt = f"Observation: {res}\n"
                        all_thoughts_and_obs.append(observation_prompt)
                        observations = res

                    logging.debug("Provoking thought observation...")
                    thought_prompt = self._provoke_thought_obs(observations)
                    all_thoughts_and_obs.append(f"\n{thought_prompt}\nActions:\n")

                    prompt = f"{base_prompt}\n" + "\n".join(all_thoughts_and_obs)
                    logging.debug(f"\nSTART:{'*' * 20}\n{prompt}\n{'*' * 20}:END")
                    logging.debug("Running LLM with updated prompt...")
                    observations = self.llm.run(prompt)
                    iteration += 1
            else:
                if iteration == max_iterations:
                    logging.info("---- Forcing Output ----")
                    cont, final_output = self._force_output(observations, all_thoughts_and_obs)
                    if cont:
                        raise OpenAGIException(
                            f"LLM did not produce the expected output after {iteration} iterations for task {cur_task.name}"
                        )
                    output = final_output
                    cur_task.result = output
                    cur_task.actions = te_vars["supported_actions"]
                    self.memory.update_task(cur_task)
                    task_lists.add_completed_tasks(cur_task)

            previous_task_context = self.get_previous_task_contexts(task_lists)
            task_lists.add_completed_tasks(cur_task)

        logging.info("Finished Execution...")

        if self.output_format == OutputFormat.markdown:
            logging.info("Output Formatting...")
            output_formatter = FormatterAction(
                content=output,
                format_type=OutputFormat.markdown,
                llm=self.llm,
                memory=self.memory,
            )
            output = output_formatter.execute()

        logging.debug(f"Execution Completed for Session ID - {self.memory.session_id}")
        return output

# From openagi/agent.py
def get_supported_actions_for_worker(self, actions_list: List[str],tool_list: List[str]):
        """
        This function takes a list of action names (strings) and returns a list of class objects
        from the modules within the 'tools' folder that match these action names and inherit from BaseAction.

        :param actions_list: List of action names as strings.
        :return: List of matching class objects.
        """
        matching_classes = []
        #tool_list = get_tool_list()
        # Iterate through all modules in the tools package
        for action in tool_list:
            if action.__name__ in actions_list:
                matching_classes.append(action)

        return matching_classes

# From openagi/agent.py
def save_ltm(self, action_type: str, session: SessionDict):
        """
        Save a session to long-term memory by either adding or updating an existing session.

        :param action_type: Type of operation: 'add' or 'update'
        :param session: The SessionDict object containing session details
        """
        # Get feedback for plan and answer
        session.plan_feedback = self.input_action.execute(
            prompt=(
                f"Review the generated plan: \n{session.plan}\n"
                "If satisfied, press ENTER. \nOtherwise, describe the issue and suggest improvements:"
            )
        ).strip()

        session.ans_feedback = self.input_action.execute(
            prompt=(
                f"Review the generated answer: \n{session.answer}\n"
                "If satisfied, press ENTER. \nOtherwise, describe the issue and suggest improvements:"
            )
        ).strip()

        # Save or update based on the action_type
        if action_type == "add":
            self.memory.add_ltm(session)
            logging.info(f"Session added to long-term memory: {session}")
        elif action_type == "update":
            self.memory.update_ltm(session)
            logging.info(f"Session updated in long-term memory: {session}")
        else:
            raise ValueError("Invalid action_type. Use 'add' or 'update'.")


# From openagi/exception.py
class OpenAGIException(Exception):
    ...

# From openagi/exception.py
class ExecutionFailureException(Exception):
    """Task Execution Failed"""

# From openagi/exception.py
class LLMResponseError(OpenAGIException):
    """No useful Response found"""


# From actions/utils.py
def run_action(action_cls: str, memory, llm, **kwargs):
    """
    Runs the specified action with the provided keyword arguments.

    Args:
        action_cls (str): The class name of the action to be executed.
        **kwargs: Keyword arguments to be passed to the action class constructor.

    Returns:
        The result of executing the action.
    """
    logging.info(f"Running Action - {str(action_cls)}")
    kwargs["memory"] = memory
    kwargs["llm"] = llm
    action: BaseAction = action_cls(**kwargs)  # Create an instance with provided kwargs
    res = action.execute()
    return res

from openagi.llms.base import LLMBaseModel
from typing import ClassVar

# From actions/base.py
class BaseAction(BaseModel):
    """Base Actions class to be inherited by other actions, providing basic functionality and structure."""

    session_id: int = Field(default_factory=str, description="SessionID of the current run.")
    previous_action: Optional[Any] = Field(
        default=None,
        description="Observation or Result of the previous action that might needed to run the current action.",
    )
    llm: Optional[LLMBaseModel] = Field(
        description="LLM Model to be used.", default=None, exclude=True
    )
    memory: Optional[Memory] = Field(
        description="Memory that stores the results of the earlier tasks executed for the current objective.",
        exclude=True,
        default=None,
    )

    def execute(self):
        """Executes the action"""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def cls_doc(cls):
        default_exclude_doc_fields = ["llm", "memory", "session_id", "name", "description"]
        return {
            "cls": {
                "kls": cls.__name__,
                "module": cls.__module__,
                "doc": dedent(cls.__doc__).strip() if cls.__doc__ else "",
            },
            "params": {
                field_name: field.description
                for field_name, field in cls.model_fields.items()
                if field_name not in default_exclude_doc_fields
            },
        }

# From actions/base.py
class ConfigurableAction(BaseAction):
    config: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def set_config(cls, *args, **kwargs):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                cls.config.update(args[0])
            else:
                raise ValueError("If using positional arguments, a single dictionary must be provided.")
        cls.config.update(kwargs)

    @classmethod
    def get_config(cls, key: str, default: Any = None) -> Any:
        return cls.config.get(key, default)

# From actions/base.py
def cls_doc(cls):
        default_exclude_doc_fields = ["llm", "memory", "session_id", "name", "description"]
        return {
            "cls": {
                "kls": cls.__name__,
                "module": cls.__module__,
                "doc": dedent(cls.__doc__).strip() if cls.__doc__ else "",
            },
            "params": {
                field_name: field.description
                for field_name, field in cls.model_fields.items()
                if field_name not in default_exclude_doc_fields
            },
        }

# From actions/base.py
def set_config(cls, *args, **kwargs):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                cls.config.update(args[0])
            else:
                raise ValueError("If using positional arguments, a single dictionary must be provided.")
        cls.config.update(kwargs)

# From actions/base.py
def get_config(cls, key: str, default: Any = None) -> Any:
        return cls.config.get(key, default)

from openagi.prompts.summarizer import SummarizerPrompt

# From actions/compressor.py
class SummarizerAction(BaseAction):
    """Summarizer Action"""

    past_messages: Any = Field(
        ...,
        description="Messages/Data to be summarized",
    )

    def execute(self):
        summarizer: str = SummarizerPrompt.from_template({"past_messages": self.past_messages})
        return self.llm.run(summarizer)


# From actions/formatter.py
class FormatterAction(BaseAction):
    """Content Formatter Action"""

    content: Any = Field(..., description="Data/Content to be formatted.")
    format_type: str = Field(
        default="markdown",
        description="Type to which the content will be formatted to. It will be modified to the supported formats and returned. Supported Formats - markdown/plan-text",
    )

    def execute(self):
        return self.llm.run(
            f"Format and return the below response in {self.format_type} format without removing any content. You can rephrase if required.\n{self.content}"
        )


# From actions/console.py
class ConsolePrint(BaseAction):
    content: str = Field(
        ...,
        description="The content/data passed will be logged into the console using pprint.pprint() module.",
    )

    def execute(self):
        pprint(self.content)
        return self.content

from pathlib import Path

# From actions/files.py
class CreateFileAction(BaseAction):
    """
    Creates a new file with the specified content and directory structure.
    """

    filename: str = Field(..., description="Name of the file along with the directory.")
    parent_mkdir: bool = Field(
        default=True, description="Create parent directories of the file if not exist."
    )
    exist_ok: bool = Field(
        default=True,
        description="Do not raise error if any of the parent directories exists.",
    )
    file_content: str = Field(default="", description="String content of the file to insert")
    write_text_kargs: Optional[Dict] = Field(
        default=None, description="Kwargs to be passed to pathlib's write_text method"
    )

    def execute(self):
        output_file = Path(self.filename)
        print(f"Created file - {output_file.absolute()}")
        output_file.parent.mkdir(
            parents=self.parent_mkdir,
            exist_ok=self.exist_ok,
        )

        write_kwargs = {}
        if self.write_text_kargs:
            write_kwargs = {**write_kwargs}

        output_file.write_text(data=self.file_content, **write_kwargs)
        return self.file_content

# From actions/files.py
class WriteFileAction(BaseAction):
    """
    Executes the action to write the provided content to a file at the specified path.
    """

    filename: str = Field(..., description="Name of the file along with the directory.")
    file_content: str = Field(default="", description="String content of the file to insert")
    file_mode: str = Field(
        default="w",
        description="File mode to open the file with while using python's open() func. Defaults to 'w'",
    )

    def execute(self):
        output_file = Path(self.filename)
        logging.info(f"Writing file - {output_file.absolute()}")
        with open(output_file.absolute(), self.file_mode) as f:
            f.write(self.file_content)
        return self.file_content

# From actions/files.py
class ReadFileAction(BaseAction):
    """
    Reads the contents of a file specified by the `file_path` parameter.
    """

    file_path: str = Field(..., description="Name of the file along with the directory.")

    def execute(self):
        output_file = Path(self.file_path)
        logging.info(f"Reading file - {output_file.absolute()}")
        with open(output_file.absolute(), "r") as f:
            return f.read()


# From actions/human_input.py
class HumanCLIInput(BaseAction):
    ques_prompt: str = Field(
        default="Do you think this task is progressing as expected [y/n] or [yes/no]: ",
        description="question to be asked to human",
    )

    def execute(self, prompt=ques_prompt):
        response = input(f"Agent: {prompt}\nYou: ")
        return response

from queue import Queue
from openagi.tasks.task import Task

# From tasks/lists.py
class TaskLists:
    def __init__(self) -> None:
        self.tasks = Queue()
        self.completed_tasks = Queue()

    def add_task(self, task: Task) -> None:
        """Adds a Task instance to the queue."""
        self.tasks.put(task)

    def add_tasks(self, tasks: List[Dict[str, str]]):
        for task in tasks:
            task["name"] = task["task_name"]
            worker_config: Optional[Dict[str, str]] = None
            
            if all(key in task for key in ["role", "instruction", "worker_name", "supported_actions"]):
                worker_config = {
                    "role": task["role"],
                    "instructions": task["instruction"],
                    "name": task["worker_name"],
                    "supported_actions": task["supported_actions"]
                }
            task["worker_config"] = worker_config
            self.add_task(Task(**task))

    def get_tasks_queue(self) -> List:
        return self.tasks

    def get_tasks_lists(self):
        return [dict(task.model_fields.items()) for task in list(self.tasks.queue)]

    def get_next_unprocessed_task(self) -> Task:
        """Retrieves the next unprocessed task from the queue."""
        if not self.tasks.empty():
            return self.tasks.get_nowait()
        return None

    @property
    def all_tasks_completed(self) -> bool:
        """Checks if all tasks in the queue have been processed."""
        return self.tasks.empty()

    def add_completed_tasks(self, task: Task):
        self.completed_tasks.put(task)

# From tasks/lists.py
def add_task(self, task: Task) -> None:
        """Adds a Task instance to the queue."""
        self.tasks.put(task)

# From tasks/lists.py
def add_tasks(self, tasks: List[Dict[str, str]]):
        for task in tasks:
            task["name"] = task["task_name"]
            worker_config: Optional[Dict[str, str]] = None
            
            if all(key in task for key in ["role", "instruction", "worker_name", "supported_actions"]):
                worker_config = {
                    "role": task["role"],
                    "instructions": task["instruction"],
                    "name": task["worker_name"],
                    "supported_actions": task["supported_actions"]
                }
            task["worker_config"] = worker_config
            self.add_task(Task(**task))

# From tasks/lists.py
def get_tasks_queue(self) -> List:
        return self.tasks

# From tasks/lists.py
def get_tasks_lists(self):
        return [dict(task.model_fields.items()) for task in list(self.tasks.queue)]

# From tasks/lists.py
def get_next_unprocessed_task(self) -> Task:
        """Retrieves the next unprocessed task from the queue."""
        if not self.tasks.empty():
            return self.tasks.get_nowait()
        return None

# From tasks/lists.py
def all_tasks_completed(self) -> bool:
        """Checks if all tasks in the queue have been processed."""
        return self.tasks.empty()

# From tasks/lists.py
def add_completed_tasks(self, task: Task):
        self.completed_tasks.put(task)

from openagi.utils.helper import get_default_id

# From tasks/task.py
class Task(BaseModel):
    id: str = Field(default_factory=get_default_id)
    name: str = Field(..., description="Name of task being.")
    description: str = Field(..., description="Description of the individual task.")
    result: Optional[str] = Field(..., default_factory=str, description="Result of the task.")
    actions: Optional[str] = Field(
        ...,
        default_factory=str,
        description="Actions undertaken to acheieve the task. Usually set after the current task is executed.",
    )
    worker_id: Optional[str] = Field(
        description="WorkerId associated to accomplish the given task using supported actions.",
        default_factory=str,
    )
    worker_config: Optional[Dict[str, Any]] = Field(
        description="Stores workers configuration values"
    )
    @property
    def is_done(self):
        return bool(self.result)

    def set_result(self, result):
        self.result = result

# From tasks/task.py
def is_done(self):
        return bool(self.result)

# From tasks/task.py
def set_result(self, result):
        self.result = result

from openagi.llms.base import LLMConfigModel
from openagi.utils.yamlParse import read_from_env
from langchain_core.messages import HumanMessage
from langchain_sambanova import ChatSambaNovaCloud

# From llms/sambanova.py
class SambaNovaConfigModel(LLMConfigModel):
    """Configuration model for SambaNova."""
    
    sambanova_api_key: str
    base_url: str
    project_id: str
    model: str = "Meta-Llama-3.3-70B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.01
    streaming: bool = False

# From llms/sambanova.py
class SambaNovaModel(LLMBaseModel):
    """SambaNova implementation of the LLMBaseModel."""
    
    config: Any

    def load(self):
        """Initializes the SambaNova client with configurations."""
        self.llm = ChatSambaNovaCloud(
            base_url=self.config.base_url,
            project_id=self.config.project_id,
            api_key=self.config.sambanova_api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            streaming=self.config.streaming
        )
        return self.llm

    def run(self, input_data: str):
        """Processes input using SambaNova model."""
        if not self.llm:
            self.load()
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> SambaNovaConfigModel:
        """Loads configurations from environment variables."""
        return SambaNovaConfigModel(
            sambanova_api_key=read_from_env("SAMBANOVA_API_KEY", raise_exception=True),
            base_url=read_from_env("SAMBANOVA_BASE_URL", raise_exception=True),
            project_id=read_from_env("SAMBANOVA_PROJECT_ID", raise_exception=True),
            model=read_from_env("SAMBANOVA_MODEL", default="Meta-Llama-3.3-70B-Instruct"),
            temperature=float(read_from_env("SAMBANOVA_TEMPERATURE", default=0.7)),
            max_tokens=int(read_from_env("SAMBANOVA_MAX_TOKENS", default=1024)),
            top_p=float(read_from_env("SAMBANOVA_TOP_P", default=0.01)),
            streaming=bool(read_from_env("SAMBANOVA_STREAMING", default=False))
        )

# From llms/sambanova.py
def load(self):
        """Initializes the SambaNova client with configurations."""
        self.llm = ChatSambaNovaCloud(
            base_url=self.config.base_url,
            project_id=self.config.project_id,
            api_key=self.config.sambanova_api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            streaming=self.config.streaming
        )
        return self.llm

# From llms/sambanova.py
def load_from_env_config() -> SambaNovaConfigModel:
        """Loads configurations from environment variables."""
        return SambaNovaConfigModel(
            sambanova_api_key=read_from_env("SAMBANOVA_API_KEY", raise_exception=True),
            base_url=read_from_env("SAMBANOVA_BASE_URL", raise_exception=True),
            project_id=read_from_env("SAMBANOVA_PROJECT_ID", raise_exception=True),
            model=read_from_env("SAMBANOVA_MODEL", default="Meta-Llama-3.3-70B-Instruct"),
            temperature=float(read_from_env("SAMBANOVA_TEMPERATURE", default=0.7)),
            max_tokens=int(read_from_env("SAMBANOVA_MAX_TOKENS", default=1024)),
            top_p=float(read_from_env("SAMBANOVA_TOP_P", default=0.01)),
            streaming=bool(read_from_env("SAMBANOVA_STREAMING", default=False))
        )

from langchain_cohere import ChatCohere

# From llms/cohere.py
class CohereConfigModel(LLMConfigModel):
    """Configuration model for Cohere model"""

    cohere_api_key: str
    model_name:str = "command"

# From llms/cohere.py
class CohereModel(LLMBaseModel):
    """Cohere LLM implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Cohere LLM that runs model locally on CPU.
    """

    config: Any

    def load(self):
        """Initializes the Cohere instance with configurations."""
        self.llm = ChatCohere(
            model = self.config.model_name,
            cohere_api_key = self.config.cohere_api_key,
            temperature = 0.1
        )
        return self.llm

    def run(self, input_data: str):
        """Runs the Cohere model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Cohere LLM service.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> CohereConfigModel:
        """Loads the Cohere configurations from a YAML file.

        Returns:
            An instance of CohereConfigModel with loaded configurations.
        """
        return CohereConfigModel(
            model_name = read_from_env("COHERE_MODEL",raise_exception=True),
            cohere_api_key = read_from_env("COHERE_API_KEY",raise_exception=True)
        )

from langchain_ollama.chat_models import ChatOllama

# From llms/ollama.py
class OllamaConfigModel(LLMConfigModel):
    """Configuration model for Ollama model"""

    model_name:str = "mistral"

# From llms/ollama.py
class OllamaModel(LLMBaseModel):
    """Ollama LLM implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Ollama LLM that runs model locally on CPU.
    """

    config: Any

    def load(self):
        """Initializes the Ollama instance with configurations."""
        self.llm = ChatOllama(
            model = self.config.model_name,
            temperature=0   
        )
        return self.llm

    def run(self, input_data: str):
        """Runs the Ollama model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Ollama LLM service.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> OllamaConfigModel:
        """Loads the Ollama configurations from a YAML file.

        Returns:
            An instance of OllamaConfigModel with loaded configurations.
        """
        return OllamaConfigModel(
            model_name = read_from_env("OLLAMA_MODEL",raise_exception=True),
        )

from abc import abstractmethod

# From llms/base.py
class LLMConfigModel(BaseModel):
    """Base configuration model for all LLMs.

    This class can be extended to include more fields specific to certain LLMs.
    """

    class Config:
        protected_namespaces = ()

    pass

# From llms/base.py
class LLMBaseModel(BaseModel):
    """Abstract base class for language learning models.

    Attributes:
        config: An instance of LLMConfigModel containing configuration.
        llm: Placeholder for the actual LLM instance, to be defined in subclasses.
    """

    config: Any
    llm: Any = None

    @abstractmethod
    def load(self):
        """Initializes the LLM instance with configurations."""
        pass

    @abstractmethod
    def run(self, input_data: Any):
        """Interacts with the LLM service using the provided input.

        Args:
            input_data: The input to process by the LLM. The format can vary.

        Returns:
            The result from processing the input data through the LLM.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_env_config():
        """Loads configuration values from a YAML file."""
        pass

# From llms/base.py
class Config:
        protected_namespaces = ()

from openai import OpenAI
from openai._exceptions import AuthenticationError

# From llms/xai.py
class XAIConfigModel(LLMConfigModel):
    """Configuration model for Grok X-AI"""

    xai_api_key: str
    model_name: str = "grok-beta"
    base_url: str = "https://api.x.ai/v1"
    system_prompt: str = "You are an AI assistant. Use the supplied tools to assist the user."

# From llms/xai.py
class XAIModel(LLMBaseModel):
    """XAI- GROK service implementation of the LLMBaseModel.

    This class implements the specific logic required to work with XAI service.
    """

    config: Any
    system_prompt: str = "You are an AI assistant"

    def load(self):
        """Initializes the XAI instance with configurations."""
        self.llm = OpenAI(
            api_key = self.config.xai_api_key,
            base_url = self.config.base_url
        )
        return self.llm

    def run(self, prompt : Any):
        """Runs the XAI model with the provided input text.

        Args:
            input_text: The input text to process.

        Returns:
            The response from XAI service.
        """
        logging.info(f"Running LLM - {self.__class__.__name__}")
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        try:
            chat_completion = self.llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{self.system_prompt}",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
                       ],
            model=self.config.model_name
            )
        except AuthenticationError:
            raise OpenAGIException("Authentication failed. Please check your XAI_API_KEY.")
        return chat_completion.choices[0].message.content

    @staticmethod
    def load_from_env_config() -> XAIConfigModel:
        """Loads the XAI configurations from a YAML file.

        Returns:
            An instance of XAIConfigModel with loaded configurations.
        """
        return XAIConfigModel(
            xai_api_key=read_from_env("XAI_API_KEY", raise_exception=True),
        )

from langchain_community.llms import HuggingFaceHub

# From llms/hf.py
class HuggingFaceConfigModel(LLMConfigModel):
    """Configuration model for Hugging Face."""
    api_token: str
    model_name: str = "huggingfaceh4/zephyr-7b-beta"
    temperature: float = 0.1
    max_new_tokens: int = 512

# From llms/hf.py
class HuggingFaceModel(LLMBaseModel):
    """Hugging Face service implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Hugging Face service.
    """

    config: Any

    def load(self):
        """Initializes the GroqModel instance with configurations."""
        self.llm = HuggingFaceHub(
            huggingfacehub_api_token = self.config.api_token,
            repo_id= self.config.model_name, 
            model_kwargs={"temperature": self.config.temperature,
                          "max_new_tokens":self.config.max_new_tokens,
                          "repetition_penalty":1.2}
        )
        return self.llm
    
    def run(self, input_data: str):
        """Runs the HuggingFace model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from HuggingFace model.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> HuggingFaceConfigModel:
        """Loads the Hugging Face configurations from a YAML file.

        Returns:
            An instance of HuggingFaceConfigModel with loaded configurations.
        """
        return HuggingFaceConfigModel(
            api_token = read_from_env("HUGGINGFACE_ACCESS_TOKEN",raise_exception=True),
            model_name=read_from_env("HUGGINGFACE_MODEL", raise_exception=True),
            temperature=read_from_env("TEMPERATURE",raise_exception=True),
            max_new_tokens= read_from_env("MAX_NEW_TOKENS",raise_exception=True)
        )

from langchain_openai import ChatOpenAI

# From llms/openai.py
class OpenAIConfigModel(LLMConfigModel):
    """Configuration model for OpenAI."""

    model_name: str = "gpt-4o"
    openai_api_key: str

# From llms/openai.py
class OpenAIModel(LLMBaseModel):
    """OpenAI service implementation of the LLMBaseModel.

    This class implements the specific logic required to work with OpenAI service.
    """

    config: Any

    def load(self):
        """Initializes the OpenAI instance with configurations."""
        self.llm = ChatOpenAI(
            openai_api_key=self.config.openai_api_key,
            model_name=self.config.model_name,
        )
        return self.llm

    def run(self, input_text: str):
        """Runs the OpenAI model with the provided input text.

        Args:
            input_text: The input text to process.

        Returns:
            The response from OpenAI service.
        """
        logging.info(f"Running LLM - {self.__class__.__name__}")
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_text)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> OpenAIConfigModel:
        """Loads the OpenAI configurations from a YAML file.

        Returns:
            An instance of OpenAIConfigModel with loaded configurations.
        """
        return OpenAIConfigModel(
            openai_api_key=read_from_env("OPENAI_API_KEY", raise_exception=True),
        )

from langchain_groq import ChatGroq

# From llms/groq.py
class GroqConfigModel(LLMConfigModel):
    """Configuration model for Groq Chat model."""

    groq_api_key: str
    model_name: str = "mixtral-8x7b-32768"
    temperature: float = 0.1

# From llms/groq.py
class GroqModel(LLMBaseModel):
    """Chat Groq Model implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Chat Groq Model.
    """

    config: Any

    def load(self):
        """Initializes the GroqModel instance with configurations."""
        self.llm = ChatGroq(
            model_name = self.config.model_name,
            groq_api_key = self.config.groq_api_key,
            temperature = self.config.temperature
        )
        return self.llm
    
    def run(self, input_data: str):
        """Runs the Chat Groq model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Groq model with low inference latency.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content
    
    @staticmethod
    def load_from_env_config() -> GroqConfigModel:
        """Loads the GroqModel configurations from a env file.

        Returns:
            An instance of GroqConfigModel with loaded configurations.
        """
        return GroqConfigModel(
            groq_api_key=read_from_env("GROQ_API_KEY", raise_exception=True),
            model_name = read_from_env("GROQ_MODEL",raise_exception=True),
            temperature=read_from_env("GROQ_TEMP",raise_exception=True)
        )

from langchain_mistralai import ChatMistralAI

# From llms/mistral.py
class MistralConfigModel(LLMConfigModel):
    """Configuration model for Mistral."""

    mistral_api_key: str
    model_name: str = "mistral-large-latest"
    temperature: float = 0.1

# From llms/mistral.py
class MistralModel(LLMBaseModel):
    """Mistral service implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Mistral service.
    """

    config: Any

    def load(self):
        """Initializes the Mistral instance with configurations."""
        self.llm = ChatMistralAI(
           model = self.config.model_name,
           temperature = self.config.temperature,
           api_key = self.config.mistral_api_key
        )
        return self.llm

    def run(self, input_text: str):
        """Runs the Mistral model with the provided input text.

        Args:
            input_text: The input text to process.

        Returns:
            The response from Mistral service.
        """
        logging.info(f"Running LLM - {self.__class__.__name__}")
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_text)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> MistralConfigModel:
        """Loads the Mistral configurations from a YAML file.

        Returns:
            An instance of MistralConfigModel with loaded configurations.
        """
        return MistralConfigModel(
            mistral_api_key=read_from_env("MISTRAL_API_KEY", raise_exception=True),
        )

from langchain_anthropic import ChatAnthropic

# From llms/claude.py
class ChatAnthropicConfigModel(LLMConfigModel):
    """
    Configuration model Anthropic model. This provides opus, sonnet SOTA models
    """
    anthropic_api_key: str
    temperature: float = 0.5
    model_name: str = "claude-3-5-sonnet-20240620"

# From llms/claude.py
class ChatAnthropicModel(LLMBaseModel):
    """
    Define the Claude LLM from Anthropic using Langchain LLM integration
    """
    config: Any

    def load(self):
        """Initializes the ChatAnthropic instance with configurations."""
        self.llm = ChatAnthropic(
            model_name = self.config.model_name,
            api_key = self.config.anthropic_api_key,
            temperature = self.config.temperature
        )
        return self.llm

    def run(self, input_data: str):
        """
        Runs the Chat Anthropic model with the provided input text.
        Args:
            input_data: The input text to process.
        Returns:
            The response from Anthropic - Claude LLM.
        """

        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        
        message = HumanMessage(content=input_data)
        response = self.llm([message])
        return response.content

    @staticmethod
    def load_from_env_config() -> ChatAnthropicConfigModel:
        """Loads the ChatAnthropic configurations from a env file.

        Returns:
            An instance of ChatAnthropicConfigModel with loaded configurations.
        """
        return ChatAnthropicConfigModel(
            anthropic_api_key = read_from_env("ANTHROPIC_API_KEY", raise_exception=True),
            model_name = read_from_env("CLAUDE_MODEL_NAME",raise_exception=False),
            temperature = read_from_env("TEMPERATURE",raise_exception=False)
        )

from langchain_google_genai import ChatGoogleGenerativeAI

# From llms/gemini.py
class GeminiConfigModel(LLMConfigModel):
    """Configuration model for Gemini Chat model."""

    google_api_key: str
    model_name: str = "gemini-pro"
    temperature: float = 0.1

# From llms/gemini.py
class GeminiModel(LLMBaseModel):
    """Chat Gemini Model implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Chat Google Generative - Gemini Model.
    """

    config: Any

    def load(self):
        """Initializes the GeminiModel instance with configurations."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key = self.config.google_api_key,
            model = self.config.model_name,
            temperature= self.config.temperature
        )
        return self.llm
    
    def run(self, input_data: str):
        """Runs the Chat Gemini model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Gemini model with low inference latency.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content
    
    @staticmethod
    def load_from_env_config() -> GeminiConfigModel:
        """Loads the GeminiModel configurations from a env file.

        Returns:
            An instance of GeminiConfigModel with loaded configurations.
        """
        return GeminiConfigModel(
            google_api_key = read_from_env("GOOGLE_API_KEY", raise_exception=True),
            model_name = read_from_env("Gemini_MODEL",raise_exception=False),
            temperature=read_from_env("Gemini_TEMP",raise_exception=False)
        )

from langchain_openai import AzureChatOpenAI

# From llms/azure.py
class AzureChatConfigModel(LLMConfigModel):
    """Configuration model for Azure Chat OpenAI."""

    base_url: str
    deployment_name: str
    model_name: str
    openai_api_version: str
    api_key: str

# From llms/azure.py
class AzureChatOpenAIModel(LLMBaseModel):
    """Azure's OpenAI service implementation of the LLMBaseModel.

    This class implements the specific logic required to work with Azure's OpenAI service.
    """

    config: Any

    def load(self):
        """Initializes the AzureChatOpenAI instance with configurations."""
        self.llm = AzureChatOpenAI(
            azure_deployment=self.config.deployment_name,
            model_name=self.config.model_name,
            openai_api_version=self.config.openai_api_version,
            openai_api_key=self.config.api_key,
            azure_endpoint=self.config.base_url,
        )
        return self.llm

    def run(self, input_data: str):
        """Runs the Azure Chat OpenAI model with the provided input text.

        Args:
            input_data: The input text to process.

        Returns:
            The response from Azure's OpenAI service.
        """
        if not self.llm:
            self.load()
        if not self.llm:
            raise ValueError("`llm` attribute not set.")
        message = HumanMessage(content=input_data)
        resp = self.llm([message])
        return resp.content

    @staticmethod
    def load_from_env_config() -> AzureChatConfigModel:
        """Loads the AzureChatOpenAI configurations from a YAML file.

        Returns:
            An instance of AzureChatConfigModel with loaded configurations.
        """
        return AzureChatConfigModel(
            base_url=read_from_env("AZURE_BASE_URL", raise_exception=True),
            deployment_name=read_from_env("AZURE_DEPLOYMENT_NAME", raise_exception=True),
            model_name=read_from_env("AZURE_MODEL_NAME", raise_exception=True),
            openai_api_version=read_from_env("AZURE_OPENAI_API_VERSION", raise_exception=True),
            api_key=read_from_env("AZURE_OPENAI_API_KEY", raise_exception=True),
        )

from openagi.prompts.base import BasePrompt

# From planner/base.py
class BasePlanner(BaseModel):
    human_intervene: bool = Field(
        default=True,
        description="If human internvention is required or not.",
    )
    input_action: Optional[BaseAction] = Field(
        description="If `human_intervene` is enabled, which action to be performed.",
    )
    prompt: BasePrompt = Field(description="Prompt to be used")

    def _extract_task_from_response(llm_response: str):
        raise

    def human_clarification(self, response: str) -> bool:
        """Whether to Ask clarifying questions"""
        raise NotImplementedError("Subclasses must implement this method.")

    def plan(self, query: str, description: str, long_term_context: str, supported_actions: List[BaseAction],*args,
        **kwargs,) -> Dict:
        raise NotImplementedError("Subclasses must implement this method.")

# From planner/base.py
def human_clarification(self, response: str) -> bool:
        """Whether to Ask clarifying questions"""
        raise NotImplementedError("Subclasses must implement this method.")

# From planner/base.py
def plan(self, query: str, description: str, long_term_context: str, supported_actions: List[BaseAction],*args,
        **kwargs,) -> Dict:
        raise NotImplementedError("Subclasses must implement this method.")



import json
from openagi.exception import LLMResponseError
from openagi.planner.base import BasePlanner
from openagi.prompts.constants import CLARIFIYING_VARS
from openagi.prompts.task_clarification import TaskClarifier
from openagi.prompts.task_creator import AutoTaskCreator
from openagi.prompts.task_creator import SingleAgentTaskCreator
from openagi.prompts.task_creator import MultiAgentTaskCreator

# From planner/task_decomposer.py
class TaskPlanner(BasePlanner):
    human_intervene: bool = Field(
        default=False, description="If human internvention is required or not."
    )
    input_action: Optional[HumanCLIInput] = Field(
        default_factory=HumanCLIInput,
        description="If `human_intervene` is enabled, which action to be performed.",
    )
    prompt: Optional[BasePrompt] = Field(
        description="Prompt to be used",
        default=None,
    )
    workers: Optional[List[Worker]] = Field(
        default=None,
        description="List of workers to be used.",
    )
    llm: Optional[LLMBaseModel] = Field(default=None, description="LLM Model to be used")
    retry_threshold: int = Field(
        default=3, description="Number of times to retry the task if it fails."
    )
    autonomous: bool = Field(
        default=False, description="Autonomous will self assign role and instructions and divide it among the workers"
    )
    """
    def get_prompt(self) -> None:
        if not self.prompt:
            if self.workers:
                self.prompt = MultiAgentTaskCreator(workers=self.workers)
            else:
                self.prompt = SingleAgentTaskCreator()
        logging.info(f"Using prompt: {self.prompt.__class__.__name__}")
        return self.prompt
    """
    def get_prompt(self) -> BasePrompt:
        if not self.prompt:
            if self.autonomous:
                self.prompt = AutoTaskCreator()
            else:
                if self.workers:
                    self.prompt = MultiAgentTaskCreator(workers=self.workers) 
                else:
                    self.prompt = SingleAgentTaskCreator()
                    
        logging.info(f"Using prompt: {self.prompt.__class__.__name__}")
        return self.prompt

    def _extract_task_from_response(self, llm_response: str) -> Union[str, None]:
        """
        Extracts the last JSON object from the given LLM response string.

        Args:
            llm_response (str): The LLM response string to extract the JSON from.

        Returns:
            Union[str, None]: The last JSON object extracted from the response, or None if no JSON was found.
        """
        return get_last_json(llm_response)

    def human_clarification(self, planner_vars) -> Dict:
        """
        Handles the human clarification process during task planning.

        This method is responsible for interacting with the human user to clarify any
        ambiguities or missing information in the task planning process. It uses a
        TaskClarifier prompt to generate a question for the human, and then waits for
        the human's response to update the planner variables accordingly.

        The method will retry the clarification process up to `self.retry_threshold`
        times before giving up and returning the current planner variables.

        Args:
            planner_vars (Dict): The current planner variables, which may be updated
                based on the human's response.

        Returns:
            Dict: The updated planner variables after the human clarification process.
        """

        logging.info(f"Initiating Human Clarification. Make sure to clarify the questions, if not just type `I dont know` to stop")
        chat_history = []
    
        while True:
            clarifier_vars = {
                **planner_vars,
                "chat_history": "\n".join(chat_history)
            }
            clarifier = TaskClarifier.from_template(variables=clarifier_vars)
            
            response = self.llm.run(clarifier)
            parsed_response = get_last_json(response, llm=self.llm)
            question = parsed_response.get("question", "").strip()
            
            if not question:
                return planner_vars

            # set the ques_prompt to question in input_action
            # self.input_action.ques_prompt = question
            human_input = self.input_action.execute(prompt=question)
            planner_vars["objective"] += f" {human_input}"
            
            # Update chat history
            chat_history.append(f"Q: {question}")
            chat_history.append(f"A: {human_input}")
            
            # Check for unwillingness to continue
            if any(phrase in human_input.lower() for phrase in ["don't know", "no more questions", "that's all", "stop asking"]):
                return planner_vars

    def extract_ques_and_task(self, ques_prompt):
        """
        Extracts the question to be asked to the human and the remaining task from the given question prompt.

        Args:
            ques_prompt (str): The question prompt containing the question to be asked to the human and the remaining task.

        Returns:
            Tuple[str, str]: The task and the question to be asked to the human.
        """
        start = CLARIFIYING_VARS["start"]
        end = CLARIFIYING_VARS["end"]
        # pattern to find question to be asked to human
        regex = rf"{start}(.*?){end}"

        # Find all matches in the text
        matches = re.findall(regex, ques_prompt)

        # remove <clarify from human>...</clarify from human> part from the prompt
        task = re.sub(regex, "", ques_prompt)
        if not matches:
            return None, None

        question = matches[-1]
        if question and question.strip():
            f"OpenAGI: {question}\nYou: "
        return task, question

    def plan(
        self,
        query: str,
        description: str,
        long_term_context : str,
        supported_actions: List[Dict],
        *args,
        **kwargs,
    ) -> Dict:
        """
        Plans a task by querying a large language model (LLM) and extracting the resulting tasks.

        Args:
            query (str): The objective or query to plan for.
            description (str): A description of the task or problem to solve.
            supported_actions (List[Dict]): A list of dictionaries describing the actions that can be taken to solve the task.
            *args: Additional arguments to pass to the LLM.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            Dict: A dictionary containing the planned tasks.
            :param supported_actions:
            :param query:
            :param description:
            :param long_term_context:
        """
        planner_vars = dict(
            objective=query,
            task_descriptions=description,
            supported_actions=supported_actions,
            previous_context=long_term_context,
            *args,
            **kwargs,
        )

        if self.human_intervene:
            planner_vars = self.human_clarification(planner_vars)

        prompt_template = self.get_prompt()

        prompt: str = prompt_template.from_template(variables=planner_vars)
        resp = self.llm.run(prompt)

        tasks = self._extract_task_with_retry(resp, prompt)

        if not tasks:
            raise LLMResponseError("Note: This not a error => No tasks was planned in the Planner response. Tweak the prompt and actions, then try again")

        print(f"\n\nTasks: {tasks}\n\n")
        return tasks

    def _extract_task_with_retry(self, llm_response: str, prompt: str) -> Dict:
        """
        Attempts to extract a task from the given LLM response, retrying up to a specified threshold if the response is not valid JSON.

        Args:
            llm_response (str): The response from the language model.
            prompt (str): The prompt used to generate the LLM response.

        Returns:
            Dict: The extracted task, or raises an exception if the task could not be extracted after multiple retries.

        Raises:
            LLMResponseError: If the task could not be extracted after multiple retries.
        """
        retries = 0
        while retries < self.retry_threshold:
            try:
                resp = self._extract_task_from_response(llm_response=llm_response)
                logging.debug(f"\n\nExtracted Task: {resp}\n\n")
                return resp
            except json.JSONDecodeError:
                retries += 1
                logging.info(
                    f"Retrying task extraction {retries}/{self.retry_threshold} due to an error parsing the JSON response."
                )
                llm_response = self.llm.run(prompt)

        raise LLMResponseError("Failed to extract tasks after multiple retries.")

# From planner/task_decomposer.py
def get_prompt(self) -> BasePrompt:
        if not self.prompt:
            if self.autonomous:
                self.prompt = AutoTaskCreator()
            else:
                if self.workers:
                    self.prompt = MultiAgentTaskCreator(workers=self.workers) 
                else:
                    self.prompt = SingleAgentTaskCreator()
                    
        logging.info(f"Using prompt: {self.prompt.__class__.__name__}")
        return self.prompt

# From planner/task_decomposer.py
def extract_ques_and_task(self, ques_prompt):
        """
        Extracts the question to be asked to the human and the remaining task from the given question prompt.

        Args:
            ques_prompt (str): The question prompt containing the question to be asked to the human and the remaining task.

        Returns:
            Tuple[str, str]: The task and the question to be asked to the human.
        """
        start = CLARIFIYING_VARS["start"]
        end = CLARIFIYING_VARS["end"]
        # pattern to find question to be asked to human
        regex = rf"{start}(.*?){end}"

        # Find all matches in the text
        matches = re.findall(regex, ques_prompt)

        # remove <clarify from human>...</clarify from human> part from the prompt
        task = re.sub(regex, "", ques_prompt)
        if not matches:
            return None, None

        question = matches[-1]
        if question and question.strip():
            f"OpenAGI: {question}\nYou: "
        return task, question


# From memory/sessiondict.py
class SessionDict(BaseModel):
    session_id: str
    query: str
    description: str
    answer: str
    plan: str
    plan_feedback: str = "NA"
    ans_feedback: str = "NA"

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Class method to initialize an instance from a dictionary."""
        return cls(**input_dict)

# From memory/sessiondict.py
def from_dict(cls, input_dict: dict):
        """Class method to initialize an instance from a dictionary."""
        return cls(**input_dict)

from pydantic import ConfigDict

# From storage/base.py
class BaseStorage(BaseModel):
    """Base Storage class to be inherited by other storages, providing basic functionality and structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(title="BaseStorage", description="Name of the Storage.")

    def save_document(self, id, document, metadata):
        """Save documents to the with metadata."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_document(self, id, document, metadata):
        raise NotImplementedError("Subclasses must implement this method.")

    def delete_document(self, id):
        raise NotImplementedError("Subclasses must implement this method.")

    def query_documents(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_kwargs(cls, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

# From storage/base.py
def save_document(self, id, document, metadata):
        """Save documents to the with metadata."""
        raise NotImplementedError("Subclasses must implement this method.")

# From storage/base.py
def update_document(self, id, document, metadata):
        raise NotImplementedError("Subclasses must implement this method.")

# From storage/base.py
def delete_document(self, id):
        raise NotImplementedError("Subclasses must implement this method.")

# From storage/base.py
def query_documents(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

# From storage/base.py
def from_kwargs(cls, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

import tempfile
import chromadb
from chromadb import HttpClient
from chromadb import PersistentClient
from openagi.storage.base import BaseStorage

# From storage/chroma.py
class ChromaStorage(BaseStorage):
    name: str = Field(default="ChromaDB Storage")
    client: chromadb.ClientAPI
    collection: chromadb.Collection

    @classmethod
    def get_default_persistent_path(cls):
        path = Path(tempfile.gettempdir()) / "openagi"
        return str(path.absolute())

    @classmethod
    def from_kwargs(cls, **kwargs):
        if kwargs.get("host", None) and kwargs.get("port", None):
            _client = HttpClient(host=kwargs["host"], port=kwargs["port"])
        else:
            persit_pth = kwargs.get("persist_path", cls.get_default_persistent_path())
            _client = PersistentClient(path=persit_pth)
            logging.info(f"Using Chroma persistent client with path: {persit_pth}")

        _collection = _client.get_or_create_collection(kwargs.get("collection_name"))
        logging.debug(f"Collection: Name - {_collection.name}, ID - {_collection.id}")
        return cls(client=_client, collection=_collection)

    def save_document(self, id, document, metadata):
        """Create a new document in the ChromaDB collection."""

        resp = self.collection.add(ids=id, documents=document, metadatas=metadata)
        return resp

    def update_document(self, id, document, metadata):
        """Update an existing document in the ChromaDB collection."""
        # if not isinstance(document, list):
        #     document = [document]
        # if not isinstance(metadata, list):
        #     metadata = [metadata]
        self.collection.update(ids=[id], documents=document, metadatas=metadata)
        logging.info("Document updated successfully.")

    def delete_document(self, id):
        """Delete a document from the ChromaDB collection."""
        self.collection.delete(ids=[id])
        logging.debug("Document deleted successfully.")

    def query_documents(self, **kwargs):
        """Query the ChromaDB collection for relevant documents based on the query."""
        results = self.collection.query(**kwargs)
        logging.debug(f"Queried results: {results}")
        return results

# From storage/chroma.py
def get_default_persistent_path(cls):
        path = Path(tempfile.gettempdir()) / "openagi"
        return str(path.absolute())

import importlib

# From utils/extraction.py
def force_json_output(resp_txt: str, llm) -> str:
    """
    Forces proper JSON output format in first attempt.
    """
    prompt = """
        You are a JSON formatting expert. Your task is to process the input and provide a valid JSON output.
        
        FOLLOW THESE INSTRUCTIONS to convert:
        - Output must be ONLY a JSON object wrapped in ```json code block
        - Do not include any explanations, comments, or additional text in your response. The output needs be in JSON only. 
        
        Convert this INPUT to proper JSON:
        INPUT: {resp_txt}
        Output only the JSON:
        """.strip()

    prompt = prompt.replace("{resp_txt}", resp_txt)
    return llm.run(prompt)

# From utils/extraction.py
def get_last_json(
    text: str, llm: Optional[LLMBaseModel] = None, max_iterations: int = 5
) -> Optional[Dict]:
    """
    Extracts valid JSON from text with improved reliability.
    """
    # More precise JSON block pattern
    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    matches = re.findall(pattern, text, re.MULTILINE)
    
    if matches:
        try:
            last_json = matches[-1].strip()
            last_json = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', last_json)
            last_json = re.sub(r'\s+', ' ', last_json)
            return json.loads(last_json)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {str(e)}", exc_info=True)
            if llm:
                text = force_json_output(last_json, llm)
                return get_last_json(text, None, max_iterations)
    
    if llm:
        for iteration in range(1, max_iterations + 1):
            try:
                text = force_json_output(text, llm)
                return get_last_json(text, None, max_iterations)
            except Exception as e:
                logging.error(f"Attempt {iteration} failed: {str(e)}", exc_info=True)
                if iteration == max_iterations:
                    raise OpenAGIException(
                        f"Failed to extract valid JSON after {max_iterations} attempts. Last error: {str(e)}"
                    )
    return None

# From utils/extraction.py
def get_act_classes_from_json(json_data) -> List[Tuple[str, Optional[Dict]]]:
    """
    Extracts the Action class names and parameters from a JSON block.

    Args:
        json_data (List[Dict]): A list of dictionaries containing the class and parameter information.

    Returns:
        List[Tuple[type, Optional[Dict]]]: A list of tuples containing the Action class and its initialization parameters.
    """
    actions = []

    for item in json_data:
        # Extracting module and class name
        module_name = item["cls"]["module"]
        class_name = item["cls"]["kls"]

        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Get the class from the module
        cls = getattr(module, class_name)

        # Extracting parameters for class initialization
        params = item["params"]

        # Storing the instance in the list
        actions.append((cls, params))

    return actions

# From utils/extraction.py
def find_last_r_failure_content(text):
    """
    Finds the content of the last <r_failure> tag in the given text.

    Args:
        text (str): The text to search for the <r_failure> tag.

    Returns:
        str or None: The content of the last <r_failure> tag, or None if no matches are found.
    """
    pattern = r"<r_failure>(.*?)</r_failure>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        last_match = matches[-1]
        return last_match.group(1)
    else:
        return None

# From utils/extraction.py
def extract_str_variables(template):
    """
    Extracts all variable names from a given template string.

    The function uses a regular expression to find all placeholders within curly braces in the template string, and returns a list of the extracted variable names.

    Args:
        template (str): The template string to extract variables from.

    Returns:
        list[str]: A list of variable names extracted from the template.
    """
    # This regular expression will find all placeholders within curly braces
    pattern = r"\{(\w+)\}"
    matches = re.findall(pattern, template)
    return matches

import inspect
from openagi.actions.tools import ddg_search
from openagi.actions.tools import document_loader
from openagi.actions.tools import searchapi_search
from openagi.actions.tools import serp_search
from openagi.actions.tools import serper_search
from openagi.actions.tools import webloader
from openagi.actions.tools import youtubesearch
from openagi.actions.tools import exasearch
from openagi.actions import files
from openagi.actions import formatter
from openagi.actions import human_input
from openagi.actions import compressor
from openagi.actions import console
from openagi.actions import obs_rag

# From utils/tool_list.py
def get_tool_list():
    """
    Dynamically retrieves all classes from the specified modules and returns them in a list.
    Only includes classes that are subclasses of a specific base class (if needed).

    :return: List of class objects from the specified modules.
    """
    class_list = []

    for module in modules:
        # Inspect the module for classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Optionally, filter by a specific base class, e.g., BaseAction
            # if issubclass(obj, BaseAction) and obj is not BaseAction:
            class_list.append(obj)  # Append the class itself (not an instance)

    return class_list

from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# From utils/llmTasks.py
def extract_json_from_string(text_block):
    # Load JSON string into Python object
    try:
        json_extracted = text_block.replace("```json", "").replace("```", "")
        logging.debug(f" the extracted string:: {json_extracted}")
        ret = json.loads(json_extracted)
        return ret
    except Exception as e:
        print(f"Error - please make one more attempt to run the usecase {e}")
        logging.error(f" the extracted string:: has non json {json_extracted} ::{e}")
        exit()

# From utils/llmTasks.py
def tools_handler(
    tools: List,
    task_input: Dict,
    llm: LLMBaseModel,
):
    model_prefix = """You are a smart assistant which is capable of smartly answering about how to solve a given problem in sequential tasks using the available tools.
                    The available tools are provided to you and you are supposed to answer What all tools are required to solve the problem and in which order with all of their parameters.
            """

    tool_prefix = """
        Tool describing format:
        [
            {
                "category" : "The category of the tool, defines what the tool can do in a basic manner.",
                "tool_name" : "The tool name",
                "description": "Description of the tool",
                "args": {"arg1": "arg1"},
                "cls": {
                    "kls": "Class name",
                    "module": "module name"
                },
                "output": "Output schema"
            }
        ]


        """

    tools_db = f"""
            The available tools:
            {tools}
            """
    task_desc = """
                    Task Statement:
            The Task statement has 5 parameters, and you must pay attention to all of them and give the best output.
            The 5 parameters are:
            "role" : This describes the role of the Language Model. This role tells you about your interest while looking for answers. For example if you are a private investment investor, then you will look for stock related information of different companies in recent timeframe using a search engine to properly answer the given task problem.
            "goal" : This is the goal which the final answer should achieve.
            "backstory" : This describes your role in a more detailed manner.
            "task" : The real task objective which needs to be solved. You are supposed to solve this problem by giving out sequential steps to solve the entire problem. 
            "instructions" : These are additional instructions which you have to follow while solving the task. It may be given or may not be given.
            "Tools_List" : These are the list of tools which you need to use. You are supposed to tell how to use these tools with all their paramters to solve the original problem.
            "OrderOfExecution" : This parameter tells you, whether you have to follow exact and strict order as given in the tools list or not while solving the main problem at hand. If the value is True we use the tools list in a strict manner, other wise a versatile manner.
            """

    task_input_str = ""

    for k, v in task_input.items():
        task_desc += f"\n{k}: {v}"

    example = (
        """
        Refer to some examples with the input information and the output information, you are supposed to answer the answer in this very format
        [
            {
                "role" : "Intelligent Question-answering Assistant",
                "goal" : "Answer the question with Best and very high accuracy with no mistake.",
                "backstory" : "You are an question-answering Assistant which has access to all the tools mentioned, like search and calculate, you are supposed to use thos to get the final answer.",
                "task" : "Where does the singer Arijit Singh belong to?",
                "instructions" : "",
                "Tools_list" : ["ScrapeAndSearchWebsite, SearchInternetGoogle, SearchInternetDuckDuck, getYoutubeSearchResults"],
                "OrderOfExecution" : true,
            }

            Tools:
            """
        + str(tools_db)
        + """
            Answer:
            ```json
                [
                    {
                        "category": "Search",
                        "tool_name": "DuckDuckGoSearch Tool",
                        "args": {"search_str": "trends in 2H 2023 onwards"},
                        "cls": {
                            "kls": "DuckDuckGoSearhTool",
                            "module": "openagi.tools.integrations.duckducksearch",
                        },
                        "output": "",
                    },
                    {
                        "category": "Search",
                        "tool_name": "Youtube Tool",
                        "args": {"search_str": "Trending in 2H 2023 onwards"},
                        "cls": {
                            "kls": "YoutubeSearchTool",
                            "module": "openagi.tools.integrations.youtubesearch",
                        },
                        "output": "",
                    },

                ]

            ```
        ]"""
    )

    suffix = "Provide a JSON-formatted response with the correct tool names and parameter values, without explanations, to solve the given problem accurately. Ensure the JSON format is error-free and follows the specified structure. Do not give any explanation paragraph, just give the required json answer. I want to use the final output in my code, so make sure it is in such a manner."

    final_prompt = f"""
    {model_prefix}
    {tool_prefix}
    {tools_db}
    {task_desc}
    "Task Input: {task_input_str}\n"
    {example}
    {suffix}
    """

    resp = llm.run(final_prompt)
    json_resp_tool = extract_json_from_string(resp)

    tools_resp = []
    for tool in json_resp_tool:
        cls_name = tool["cls"]["kls"]
        mod_name = tool["cls"]["module"]
        tool_cls = getattr(importlib.import_module(mod_name), cls_name)
        params = tool["args"]
        tool_cls = tool_cls()
        setattr(tool_cls, "llm", llm)
        tool_obj = tool_cls._run(**params)
        tool_resp = {
            "tool_name": tool["tool_name"],
            "output": tool_obj,
        }
        tools_resp.append(tool_resp)

    # Concatenate tools_resp into a string
    # TODO: Make it return any kind of datatypes. Right now it supports strings only.
    tool_resp_str = ""
    for tool in tools_resp:
        logging.info("getting tool output for: {tool['tool_name']}")
        tool_resp_str += f"{tool['tool_name']}: {tool['output']}\n"

    return tool_resp_str

# From utils/llmTasks.py
def getEmail(inputString, role, backstory, goal, task, llm):
    template = "As a {role}, your primary objective is to {goal}. Your assigned task involves composing an email based on the provided summary. Here's some background information to guide you: {backstory}. Your challenge is to craft a detailed email within the given context. Please ensure that your email remains focused on the following context: {context}. Keep the email concise and relevant to the provided information."
    prompt = PromptTemplate(
        input_variables=["goal", "role", "task", "backstory", "context"],
        template=template,
    )
    prompt.format(context=inputString, role=role, backstory=backstory, goal=goal, task=task)
    chain = LLMChain(llm=llm, prompt=prompt)
    inputs = {
        "role": role,
        "backstory": backstory,
        "task": task,
        "goal": goal,
        "context": inputString,  # Assuming the inputString is the context
    }
    blog = chain.run(inputs)
    print(f"the blog is  {blog}")
    return blog

# From utils/llmTasks.py
def getSummary(inputString, role, backstory, goal, task, llm):
    template = "As a {role}, your primary objective is to {goal}. Your assigned task involves performing the following {task}. Here's some background information to guide you: {backstory}. Your challenge is to craft a detailed summary within the given context. Please ensure that your summary remains focused on the following context:{context}. Keep the summary concise and relevant to the provided information.\n"
    prompt = PromptTemplate(
        input_variables=["goal", "role", "task", "backstory", "context"],
        template=template,
    )
    prompt.format(context=inputString, role=role, backstory=backstory, goal=goal, task=task)
    chain = LLMChain(llm=llm, prompt=prompt)
    inputs = {
        "role": role,
        "backstory": backstory,
        "task": task,
        "goal": goal,
        "context": inputString,  # Assuming the inputString is the context
    }
    blog = chain.run(inputs)
    print(f"output of getSummary {blog}")
    return blog

# From utils/llmTasks.py
def llm_chain(role, backstory, goal, task, llm: LLMBaseModel, input_string):
    template = "As a {role}, your primary objective is to {goal}. Your assigned task involves performing the following {task} to the best of your ability. Here's some background information to guide you: {backstory}. Also use context: {context} to provide answer for the given task. Keep the answer concise and relevant to the provided information."
    prompt = PromptTemplate(
        input_variables=["goal", "role", "task", "backstory", "context"], template=template
    )
    prompt.format(role=role, backstory=backstory, goal=goal, task=task, context=input_string)
    chain = LLMChain(llm=llm.llm, prompt=prompt)
    inputs = {
        "role": role,
        "backstory": backstory,
        "task": task,
        "goal": goal,
        "context": input_string,
    }
    code = chain.run(inputs)
    return code

# From utils/llmTasks.py
def getReview(inputString, role, backstory, goal, task, llm):
    template = "As a {role}, your primary objective is to {goal}. Your assigned task involves performing the following {task}. Here's some background information to guide you: {backstory}. Your challenge is to perform the task within the given context:{context}. Keep the answer relevant to the provided information.\n"
    prompt = PromptTemplate(
        input_variables=["goal", "role", "task", "backstory", "context"],
        template=template,
    )
    prompt.format(context=inputString, role=role, backstory=backstory, goal=goal, task=task)
    chain = LLMChain(llm=llm, prompt=prompt)
    inputs = {
        "role": role,
        "backstory": backstory,
        "task": task,
        "goal": goal,
        "context": inputString,  # Assuming the inputString is the context
    }
    review = chain.run(inputs)
    return review

# From utils/llmTasks.py
def handleLocalLLMTask(inputString, role, backstory, goal, task, llm):
    sysMsg = role + " " + backstory + " " + goal + " " + task
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1236/v1", api_key="not-needed")

    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=[
            {"role": "system", "content": sysMsg},
            {"role": "user", "content": inputString},
        ],
        temperature=0.7,
    )

    return completion.choices[0].message.content

# From utils/llmTasks.py
def getfromLocalLLM(inputString, role, backstory, goal, task, llm):
    llm_1 = Ollama(model="llama2")
    template = "As a {role}, your primary objective is to {goal}. Your assigned task involves performing the following {task}. Here's some background information to guide you: {backstory}. Your challenge is to perform the task within the given context:{context}. Keep the answer relevant to the provided information.\n"
    prompt = PromptTemplate(
        input_variables=["goal", "role", "task", "backstory", "context"], template=template
    )
    prompt.format(context=inputString, role=role, backstory=backstory, goal=goal, task=task)
    chain = LLMChain(llm=llm_1, prompt=prompt)
    inputs = {
        "role": role,
        "backstory": backstory,
        "task": task,
        "goal": goal,
        "context": inputString,  # Assuming the inputString is the context
    }  # inputstring with localllm
    review = chain.run(inputs)
    return review

# From utils/llmTasks.py
def handleLLMTask(inputString, role, backstory, goal, task, llm):
    logging.info(f"Running handleLLMTask:: {llm}")
    return llm_chain(
        role=role,
        backstory=backstory,
        goal=goal,
        task=task,
        llm=llm,
        input_string=inputString,
    )

from uuid import uuid4
from openagi.llms.openai import OpenAIModel

# From utils/helper.py
def get_default_llm():
    config = OpenAIModel.load_from_env_config()
    return OpenAIModel(config=config)

# From utils/helper.py
def get_default_id():
    return uuid4().hex


# From utils/yamlParse.py
def read_from_env(attr_name, raise_exception=False):
    attr_value = os.environ.get(attr_name)
    if not attr_value and raise_exception:
        raise ValueError(f"Unable to get config {attr_name}")
    return attr_value


# From prompts/summarizer.py
class SummarizerPrompt(BasePrompt):
    base_prompt: str = summarizer_prompt


# From prompts/base.py
class BasePrompt(BaseModel):
    name: str = Field(default="BasePrompt", description="Name of the prompt.")
    description: str = Field(
        default="BasePrompt class to be used by other actions that get created.",
        description="Description of the prompt.",
    )
    base_prompt: str = Field(default_factory=str, description="Base prompt to be used.")

    def get_prompt(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def prompt_variables(cls):
        return {
            field_name: field.field_info.description
            for field_name, field in cls.model_fields.items()
        }

    @classmethod
    def from_template(cls, variables: Dict):
        x = cls(**variables)
        for k, v in variables.items():
            placeholder = "{" + f"{k}" + "}"
            x.base_prompt = x.base_prompt.replace(placeholder, f"{v}")
        return x.base_prompt

# From prompts/base.py
def prompt_variables(cls):
        return {
            field_name: field.field_info.description
            for field_name, field in cls.model_fields.items()
        }

# From prompts/base.py
def from_template(cls, variables: Dict):
        x = cls(**variables)
        for k, v in variables.items():
            placeholder = "{" + f"{k}" + "}"
            x.base_prompt = x.base_prompt.replace(placeholder, f"{v}")
        return x.base_prompt

from openagi.prompts.constants import FAILURE_VARS

# From prompts/execution.py
class TaskExecutor(BasePrompt):
    objective: str = Field(..., description="Final objective")
    all_tasks: List[Dict] = Field(
        ..., description="List of tasks to be executed that was generated earlier"
    )
    current_task_name: str = Field(..., description="Current task name to be executed.")
    current_description: str = Field(..., description="Current task name to be executed.")
    previous_task: Optional[str] = Field(..., description="Previous task, description & result.")
    supported_actions: List[Dict] = Field(
        ...,
        description="Supported Actions that can be used to acheive the current task.",
    )
    base_prompt: str = task_execution


# From prompts/ltm.py
class LTMFormatPrompt(BasePrompt):
    base_prompt: str = ltm_prompt


# From prompts/task_creator.py
class SingleAgentTaskCreator(BasePrompt):
    base_prompt: str = single_agent_task_creation

# From prompts/task_creator.py
class MultiAgentTaskCreator(SingleAgentTaskCreator):
    base_prompt: str = worker_task_creation

# From prompts/task_creator.py
class AutoTaskCreator(BasePrompt):
    base_prompt: str = auto_task_creator



# From prompts/worker_task_execution.py
class WorkerAgentTaskExecution(BasePrompt):
    base_prompt: str = WORKER_TASK_EXECUTION


# From prompts/task_clarification.py
class TaskClarifier(BasePrompt):
    base_prompt: str = TASK_CLARIFICATION_PROMPT

from openagi.actions.base import ConfigurableAction
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader

# From tools/document_loader.py
class TextLoaderTool(ConfigurableAction):
    """Load content from a text file.
    
    This action loads and processes content from .txt files, combining
    metadata and content into a single context string.
    """
    
    def execute(self) -> str:
        file_path: str = self.get_config('filename')
        loader = TextLoader(file_path=file_path)
        documents = loader.load()
        
        if not documents:
            return ""
            
        page_content = documents[0].page_content
        source = documents[0].metadata["source"]
        return f"{source} {page_content}"

# From tools/document_loader.py
class PDFLoaderTool(ConfigurableAction):
    """Load content from a PDF file.
    
    This action loads and processes content from .pdf files, combining
    metadata and content into a single context string.
    """
    
    def execute(self) -> str:
        file_path: str = self.get_config('filename')
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()
        
        if not documents:
            return ""
            
        page_content = documents[0].page_content
        source = documents[0].metadata["source"]
        return f"{source} {page_content}"

# From tools/document_loader.py
class CSVLoaderTool(ConfigurableAction):
    """Load content from a CSV file.
    
    This action loads and processes content from .csv files, combining
    row numbers and content into a formatted string representation.
    """
    
    def execute(self) -> str:
        file_path: str = self.get_config('filename')
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        
        content_parts = []
        for idx, doc in enumerate(documents):
            row_content = doc.page_content
            row_number = doc.metadata["row"]
            content_parts.append(f"row_no {row_number}: {row_content}")
            
        return "".join(content_parts)

from duckduckgo_search import DDGS

# From tools/ddg_search.py
class DuckDuckGoSearch(ConfigurableAction):
    """Use this Action to search DuckDuckGo for a query."""

    name: str = Field(
        default_factory=str,
        description="DuckDuckGoSearch Action to search over duckduckgo using the query.",
    )
    description: str = Field(
        default_factory=str,
        description="This action is used to search for words, documents, images, videos, news, maps and text translation using the DuckDuckGo.com search engine.",
    )

    query: Any = Field(
        default_factory=str,
        description="User query, a string, to fetch web search results from DuckDuckGo",
    )

    max_results: int = Field(
        default=10,
        description="Total results, in int, to be executed from the search. Defaults to 10.",
    )

    def _get_ddgs(self):
        return DDGS()

    def execute(self):
        if self.max_results > 15:
            logging.info("Over threshold value... Limiting the Max results to 15")
            self.max_results = 15
        
        result = self._get_ddgs().text(
            self.query,
            max_results=self.max_results,
        )
        return json.dumps(result)

import warnings
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# From tools/speech_tool.py
class ElevenLabsTTS(ConfigurableAction):
    """Use this Action to generate lifelike speech using ElevenLabs' text-to-speech API."""

    text: Any = Field(
        default_factory=str,
        description="Text input that needs to be converted to speech.",
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="The ID of the voice to be used for speech synthesis.",
    )
    model_id: str = Field(
        default="eleven_multilingual_v2",
        description="The model ID used for text-to-speech conversion.",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="The output format of the generated audio.",
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""),
        description="API key for ElevenLabs' authentication.",
    )

    def execute(self):
        logging.info(f"Generating speech for text: {self.text}")
        
        if not self.api_key:
            warnings.warn(
                "ElevenLabs API key is missing. Please provide it as a parameter or set it in the .env file.",
                DeprecationWarning,
                stacklevel=2
            )
            return json.dumps({"error": "ElevenLabs API key is missing. Please provide it as a parameter or set it in the .env file."})
        
        client = ElevenLabs(api_key=self.api_key)
        try:
            audio = client.text_to_speech.convert(
                text=self.text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format,
            )
            play(audio)
            return json.dumps({"success": "Audio played successfully."})
        
        except Exception as e:
            logging.error(f"Error generating speech: {str(e)}")
            return json.dumps({"error": f"Failed to generate speech: {str(e)}"})

import yt_dlp
from youtube_search import YoutubeSearch

# From tools/youtubesearch.py
class YouTubeSearchTool(ConfigurableAction):
    """Youtube Search Tool"""

    query: str = Field(
        ..., description="Keyword required to search the video content on YouTube"
    )
    max_results: Any = Field(
        default=5,
        description="Total results, an integer, to be executed from the search. Defaults to 5",
    )

    def execute(self):
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'force_generic_extractor': True,
            'format': 'best'
        }
        results = YoutubeSearch(self.query, max_results=self.max_results)
        response = results.to_dict()
        context = ""
        for ids in response:
            url = "https://youtube.com/watch?v="+ids['id']
            context += f"Title: {ids['title']}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                description = info_dict.get('description', None)
                context += f"Description: {description} \n\n"
        return context

from langchain_community.document_loaders import WebBaseLoader

# From tools/webloader.py
class WebBaseContextTool(ConfigurableAction):
	"""
	Use this Action to extract actual context from a Webpage. The WebBaseContextTool class provides a way to load and optionally summarize the content of a webpage, returning the metadata and page content as a context string.
    If a url seems to be failing for more than once, ignore it and move forward.
	"""

	link: str = Field(
		default_factory=str,
		description="Extract context for the Agents from the Web Search through web page",
	)
	can_summarize: bool = Field(
		default=True,
		description="Indicates whether the action can summarize the content before returning. Uses lightweight summarization. Defaults to true.",
	)

	def _split_into_sentences(self, text):
		"""Split text into sentences using simple regex"""
		text = re.sub(r'\s+', ' ', text)
		sentences = re.split(r'[.!?]+', text)
		return [s.strip() for s in sentences if len(s.strip()) > 10]

	def _calculate_word_freq(self, sentences):
		"""Calculate word frequency across all sentences"""
		words = ' '.join(sentences).lower().split()
		return Counter(words)

	def _score_sentence(self, sentence, word_freq):
		"""Score a sentence based on word frequency and length"""
		words = sentence.lower().split()
		score = sum(word_freq[word] for word in words)
		return score / (len(words) + 1)

	def _get_summary(self, text, num_sentences=6):
		"""Create a simple summary by selecting top scoring sentences"""
		sentences = self._split_into_sentences(text)
		if not sentences:
			return text
			
		word_freq = self._calculate_word_freq(sentences)
		
		scored_sentences = [
			(self._score_sentence(sentence, word_freq), i, sentence)
			for i, sentence in enumerate(sentences)
		]
		
		top_sentences = sorted(scored_sentences, reverse=True)[:num_sentences]
		ordered_sentences = sorted(top_sentences, key=lambda x: x[1])
		
		return ' '.join(sentence for _, _, sentence in ordered_sentences)

	def execute(self):
		loader = WebBaseLoader(self.link)
		data = loader.load()
		metadata = data[0].metadata["title"]
		page_content = data[0].page_content
		if page_content:
			page_content = page_content.strip()
		if self.can_summarize:
			logging.info(f"Summarizing the page {self.link}...")
			page_content = self._get_summary(page_content)
		context = metadata + page_content
		return context

import http.client

# From tools/serper_search.py
class SerperSearch(ConfigurableAction):
    """Google Serper.dev Search Tool"""
    query: str = Field(..., description="User query to fetch web search results from Google")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        if 'SERPER_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated and will be removed in a future version. "
                "Please use SerperSearch.set_config(api_key='your_key') instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(api_key=os.environ['SERPER_API_KEY'])

    def execute(self):
        api_key = self.get_config('api_key')
        
        if not api_key:
            if 'SERPER_API_KEY' in os.environ:
                api_key = os.environ['SERPER_API_KEY']
                warnings.warn(
                    "Using environment variables for API keys is deprecated and will be removed in a future version. "
                    "Please use SerperSearch.set_config(api_key='your_key') instead of setting environment variables.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                raise OpenAGIException("API KEY NOT FOUND. Use SerperSearch.set_config(api_key='your_key') to set the API key.")

        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": self.query})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        result = json.loads(data)
        
        meta_data = ""
        for info in result.get("organic", []):
            meta_data += f"CONTEXT: {info.get('title', '')} \ {info.get('snippet', '')}\n"
            meta_data += f"Reference URL: {info.get('link', '')}\n\n"
            
        return meta_data.strip()

from lumaai import LumaAI

# From tools/luma_ai.py
class LumaLabsTool(ConfigurableAction):
    """Luma Labs Tool for generating AI images and videos.
    
    This action uses the Luma Labs API to generate images or videos based on text prompts.
    Supports various features including image generation, video generation, and camera motions.
    Requires an API key to be configured before use.
    """
    
    prompt: str = Field(..., description="Text prompt to generate image or video content")
    mode: str = Field(
        default="image",
        description="Mode of operation: 'image' or 'video'"
    )
    aspect_ratio: str = Field(
        default="16:9",
        description="Aspect ratio (1:1, 3:4, 4:3, 9:16, 16:9, 9:21, 21:9)"
    )
    model: str = Field(
        default="photon-1",
        description="Model to use (photon-1, photon-flash-1, ray-2 for video)"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        if 'LUMAAI_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated. "
                "Please use LumaLabsTool.set_config(api_key='your_key') instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(api_key=os.environ['LUMAAI_API_KEY'])

    def execute(self) -> str:
        api_key: str = self.get_config('api_key')
        if not api_key:
            if 'LUMAAI_API_KEY' in os.environ:
                api_key = os.environ['LUMAAI_API_KEY']
                warnings.warn(
                    "Using environment variables for API keys is deprecated. "
                    "Please use LumaLabsTool.set_config(api_key='your_key') instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                raise OpenAGIException("API KEY NOT FOUND. Use LumaLabsTool.set_config(api_key='your_key') to set the API key.")

        client = LumaAI(auth_token=api_key)

        try:
            if self.mode == "image":
                generation = client.generations.image.create(
                    prompt=self.prompt,
                    aspect_ratio=self.aspect_ratio,
                    model=self.model
                )
            else:
                generation = client.generations.create(
                    prompt=self.prompt,
                    aspect_ratio=self.aspect_ratio,
                    model="ray-2" if self.model == "ray-2" else "photon-1"
                )

            completed = False
            while not completed:
                generation = client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    raise OpenAGIException(f"Generation failed: {generation.failure_reason}")
                time.sleep(2)

            if self.mode == "image":
                result_url = generation.assets.image
                file_extension = "jpg"
            else:
                result_url = generation.assets.video
                file_extension = "mp4"

            response = requests.get(result_url, stream=True)
            file_name = f'{generation.id}.{file_extension}'
            with open(file_name, 'wb') as file:
                file.write(response.content)

            return f"""Generation completed successfully!
                   Mode: {self.mode}
                   File saved as: {file_name}
                   Prompt: {self.prompt}
                   URL: {result_url}"""

        except Exception as e:
            raise OpenAGIException(f"Error in Luma Labs generation: {str(e)}")


# From tools/wikipedia_search.py
class WikipediaSearch(ConfigurableAction):
	"""Use this Action to search Wikipedia for a query."""

	name: str = Field(
		default_factory=str,
		description="WikipediaSearch Action to search Wikipedia using the query.",
	)
	description: str = Field(
		default_factory=str,
		description="This action is used to search and retrieve information from Wikipedia articles.",
	)

	query: str = Field(
		...,
		description="User query to fetch information from Wikipedia",
	)

	max_results: int = Field(
		default=3,
		description="Maximum number of sentences to return from the Wikipedia article. Defaults to 3.",
	)

	def execute(self):
		try:
			# Search Wikipedia
			search_results = wikipedia.search(self.query)
			
			if not search_results:
				return json.dumps({"error": "No results found"})
			
			# Get the first (most relevant) page
			try:
				page = wikipedia.page(search_results[0])
				summary = wikipedia.summary(search_results[0], sentences=self.max_results)
				
				result = {
					"title": page.title,
					"summary": summary,
					"url": page.url
				}
				
				return json.dumps(result)
			
			except wikipedia.DisambiguationError as e:
				# Handle disambiguation pages
				return json.dumps({
					"error": "Disambiguation error",
					"options": e.options[:5]  # Return first 5 options
				})
				
		except Exception as e:
			logging.error(f"Error in Wikipedia search: {str(e)}")
			return json.dumps({"error": str(e)})

from exa_py import Exa

# From tools/exasearch.py
class ExaSearch(ConfigurableAction):
    """Exa Search tool for querying and retrieving information.
    
    This action uses the Exa API to perform searches and retrieve relevant content
    based on user queries. Requires an API key to be configured before use.
    """
    query: str = Field(..., description="User query or question")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        if 'EXA_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated and will be removed in a future version. "
                "Please use ExaSearch.set_config(api_key='your_key') instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(api_key=os.environ['EXA_API_KEY'])


    def execute(self) -> str:
        api_key: str = self.get_config('api_key')
        if not api_key:
            if 'EXA_API_KEY' in os.environ:
                api_key = os.environ['EXA_API_KEY']
                warnings.warn(
                    "Using environment variables for API keys is deprecated and will be removed in a future version. "
                    "Please use ExaSearch.set_config(api_key='your_key') instead of setting environment variables.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                raise OpenAGIException("API KEY NOT FOUND. Use ExaSearch.set_config(api_key='your_key') to set the API key.")
            
        exa = Exa(api_key=api_key)
        results = exa.search_and_contents(
            self.query,
            text={"max_characters": 512},
        )
        
        content_parts = []
        for result in results.results:
            content_parts.append(result.text.strip())

        content = "".join(content_parts)
        return (
            content.replace("<|endoftext|>", "")
                  .replace("NaN", "")
        )

from serpapi import GoogleSearch

# From tools/serp_search.py
class GoogleSerpAPISearch(ConfigurableAction):
    """Google Serp API Search Tool"""
    query: str = Field(
        ..., description="User query of type string used to fetch web search results from Google."
    )
    max_results: Any = Field(
        default=10,
        description="Total results, an integer, to be executed from the search. Defaults to 10",
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        if 'GOOGLE_SERP_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated and will be removed in a future version. "
                "Please use GoogleSerpAPISearch.set_config(api_key='your_key') instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            # Automatically migrate the environment variable to config
            self.set_config(api_key=os.environ['GOOGLE_SERP_API_KEY'])
    
    def execute(self):
        api_key = self.get_config('api_key')
        
        if not api_key:
            if 'GOOGLE_SERP_API_KEY' in os.environ:
                api_key = os.environ['GOOGLE_SERP_API_KEY']
                warnings.warn(
                    "Using environment variables for API keys is deprecated and will be removed in a future version. "
                    "Please use GoogleSerpAPISearch.set_config(api_key='your_key') instead of setting environment variables.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                raise OpenAGIException("API KEY NOT FOUND. Use GoogleSerpAPISearch.set_config(api_key='your_key') to set the API key.")

        search_dict = {
            "q": self.query,
            "hl": "en",
            "gl": "us",
            "num": self.max_results,
            "api_key": api_key,
        }
        logging.debug(f"{search_dict=}")
        search = GoogleSearch(search_dict)
        
        max_retries = 3
        retries = 1
        result = None
        
        while retries < max_retries and not result:
            try:
                result = search.get_dict()
            except TypeError:
                logging.error("Error during GoogleSearch.", exc_info=True)
                continue
            retries += 1
            
        if not result:
            raise OpenAGIException(f"Unable to generate result for the query {self.query}")
            
        logging.debug(result)
        logging.info(f"NOTE: REMOVE THIS BEFORE RELEASE:\n{result}\n")
        
        if error := result.get("error", NotImplemented):
            raise OpenAGIException(
                f"Error while running action {self.__class__.__name__}: {error}"
            )
            
        meta_data = ""
        for info in result.get("organic_results", []):
            meta_data += f"CONTEXT: {info.get('title', '')} \ {info.get('snippet', '')}\n"
            meta_data += f"Reference URL: {info.get('link', '')}\n\n"
            
        return meta_data.strip()

from Bio import Entrez

# From tools/pubmed_tool.py
class PubMedSearch(ConfigurableAction):
	"""PubMed Search tool for querying biomedical literature.
	
	This action uses the Bio.Entrez module to search PubMed and retrieve
	scientific articles based on user queries. Requires an email address
	to be configured for NCBI's tracking purposes.
	"""
	
	query: str = Field(..., description="Search query for PubMed")
	max_results: int = Field(
		default=5,
		description="Maximum number of results to return (default: 5)"
	)
	sort: str = Field(
		default="relevance",
		description="Sort order: 'relevance', 'pub_date', or 'first_author'"
	)

	def execute(self) -> str:
		email: str = self.get_config('email')
		if not email:
			raise OpenAGIException(
				"Email not configured. Use PubMedSearch.set_config(email='your_email@example.com')"
			)

		Entrez.email = email
		
		try:
			# Search PubMed
			search_handle = Entrez.esearch(
				db="pubmed",
				term=self.query,
				retmax=self.max_results,
				sort=self.sort
			)
			search_results = Entrez.read(search_handle)
			search_handle.close()

			if not search_results["IdList"]:
				return "No results found for the given query."

			# Fetch details for found articles
			ids = ",".join(search_results["IdList"])
			fetch_handle = Entrez.efetch(
				db="pubmed",
				id=ids,
				rettype="medline",
				retmode="text"
			)
			
			results = fetch_handle.read()
			fetch_handle.close()

			# Process and format results
			formatted_results = (
				f"Found {len(search_results['IdList'])} results for query: {self.query}\n\n"
				f"{results}"
			)

			return formatted_results

		except Exception as e:
			return f"Error searching PubMed: {str(e)}"

import arxiv

# From tools/arxiv_search.py
class ArxivSearch(ConfigurableAction):
    """
    Arxiv Search is a tool used to search articles in Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, and Statistics
    """
    query: str = Field(..., description="User query or question")
    max_results: int = Field(10, description="Total results, in int, to be executed from the search. Defaults to 10.")

    def execute(self):
        search = arxiv.Search(
        query = self.query,
        max_results = self.max_results,
                              )
        client = arxiv.Client()
        results = client.results(search)
        meta_data = ""
        for result in results:
            meta_data += f"title : {result.title}\n "
            meta_data += f"summary : {result.summary}\n "
            meta_data += f"published : {result.published}\n "
            meta_data += f"authors : {result.authors}\n "
            meta_data += f"pdf_url : {result.pdf_url}\n "
            meta_data += f"entry_id : {result.entry_id}\n\n "
        return meta_data.strip()

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# From tools/unstructured_io.py
class UnstructuredPdfLoaderAction(ConfigurableAction):
    """
    Use this Action to extract content from PDFs including metadata.
    Returns a list of dictionary with keys 'type', 'element_id', 'text', 'metadata'.
    """

    def execute(self):
        file_path = self.get_config('filename')    
        logging.info(f"Reading file {file_path}")
        
        elements = partition_pdf(file_path, extract_images_in_pdf=True)

        chunks = chunk_by_title(elements)

        dict_elements = []
        for element in chunks:
            dict_elements.append(element.to_dict())

        with open("ele.txt", "w") as f:
            f.write(str(dict_elements))

        return str(dict_elements)

import yfinance

# From tools/yahoo_finance.py
class YahooFinanceTool(ConfigurableAction):
	"""Yahoo Finance tool for fetching stock market data.
	
	This action uses the yfinance library to retrieve financial information
	about stocks, including current price, historical data, and company info.
	"""
	
	symbol: str = Field(..., description="Stock symbol to look up (e.g., 'AAPL' for Apple)")
	info_type: str = Field(
		default="summary",
		description="Type of information to retrieve: 'summary', 'price', 'history', or 'info'"
	)
	period: Optional[str] = Field(
		default="1d",
		description="Time period for historical data (e.g., '1d', '5d', '1mo', '1y')"
	)

	def execute(self) -> str:
		try:
			stock = yf.Ticker(self.symbol)
			
			if self.info_type == "summary":
				info = stock.info
				return (
					f"Company: {info.get('longName', 'N/A')}\n"
					f"Current Price: ${info.get('currentPrice', 'N/A')}\n"
					f"Market Cap: ${info.get('marketCap', 'N/A')}\n"
					f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
					f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}"
				)
				
			elif self.info_type == "price":
				return f"Current price of {self.symbol}: ${stock.info.get('currentPrice', 'N/A')}"
				
			elif self.info_type == "history":
				history = stock.history(period=self.period)
				if history.empty:
					return f"No historical data available for {self.symbol}"
				
				latest = history.iloc[-1]
				return (
					f"Historical data for {self.symbol} (last entry):\n"
					f"Date: {latest.name.date()}\n"
					f"Open: ${latest['Open']:.2f}\n"
					f"High: ${latest['High']:.2f}\n"
					f"Low: ${latest['Low']:.2f}\n"
					f"Close: ${latest['Close']:.2f}\n"
					f"Volume: {latest['Volume']}"
				)
				
			elif self.info_type == "info":
				info = stock.info
				return (
					f"Company Information for {self.symbol}:\n"
					f"Industry: {info.get('industry', 'N/A')}\n"
					f"Sector: {info.get('sector', 'N/A')}\n"
					f"Website: {info.get('website', 'N/A')}\n"
					f"Description: {info.get('longBusinessSummary', 'N/A')}"
				)
				
			else:
				return f"Invalid info_type: {self.info_type}. Supported types are: summary, price, history, info"
				
		except Exception as e:
			return f"Error fetching data for {self.symbol}: {str(e)}"

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# From tools/dalle_tool.py
class DallEImageGenerator(ConfigurableAction):
    """Use this Action to generate images using DALL·E."""

    name: str = Field(
        default_factory=str,
        description="DallEImageGenerator Action to generate an image using OpenAI's DALL·E model.",
    )
    description: str = Field(
        default_factory=str,
        description="This action is used to create images based on textual descriptions using the DALL·E model.",
    )

    query: Any = Field(
        default_factory=str,
        description="User query, a string, describing the image to be generated.",
    )

    def execute(self):
        logging.info(f"Generating image for prompt: {self.query}")
        if 'OPENAI_API_KEY' not in os.environ:
            warnings.warn(
                "Dall-E expects an OPENAI_API_KEY. Please add it to your environment variables.",
                UserWarning,
                stacklevel=2
            )
            return json.dumps({"error": "Dall-E requires an OPENAI_API_KEY. Please add it to your environment variables."})
        
        try:
            # Use the query directly without the LLM chain
            dalle_wrapper = DallEAPIWrapper()
            result = dalle_wrapper.run(self.query)
            return json.dumps(result)

        except Exception as e:
            logging.error(f"Error generating image: {str(e)}")
            return json.dumps({"error": f"Failed to generate image: {str(e)}"})

from tavily import TavilyClient

# From tools/tavilyqasearch.py
class TavilyWebSearchQA(ConfigurableAction):
    """
    Tavily Web Search QA is a tool used when user needs to ask the question in terms of query to get response
    """
    query: str = Field(..., description="User query or question")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        if 'TAVILY_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated and will be removed in a future version. "
                "Please use TavilyWebSearchQA.set_config(api_key='your_key') instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(api_key=os.environ['TAVILY_API_KEY'])

    def execute(self):
        api_key = self.get_config('api_key')
        
        if not api_key:
            if 'TAVILY_API_KEY' in os.environ:
                api_key = os.environ['TAVILY_API_KEY']
                warnings.warn(
                    "Using environment variables for API keys is deprecated and will be removed in a future version. "
                    "Please use TavilyWebSearchQA.set_config(api_key='your_key') instead of setting environment variables.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                raise OpenAGIException("API KEY NOT FOUND. Use TavilyWebSearchQA.set_config(api_key='your_key') to set the API key.")
        
        client = TavilyClient(api_key=api_key)
        response = client.qna_search(query=self.query)
        return response

import base64
from langchain_community.document_loaders.github import GithubFileLoader

# From tools/github_search_tool.py
class OpenAGIGithubFileLoader(GithubFileLoader):
    def get_file_paths(self) -> List[Dict]:
        base_url = (
            f"{self.github_api_url}/repos/{self.repo}/git/trees/" f"{self.branch}?recursive=1"
        )
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()
        all_files = response.json()["tree"]
        
        """ one element in all_files
        {
            'path': '.github', 
            'mode': '040000', 
            'type': 'tree', 
            'sha': '89a2ae046e8b59eb96531f123c0c6d4913885df1', 
            'url': 'https://github.com/api/v3/repos/shufanhao/langchain/git/trees/89a2ae046e8b59eb96531f123c0c6d4913885dxxx'
        }
        """
        required_files = [
            f for f in all_files if not (self.file_filter and not self.file_filter(f["path"]))
        ]
        return required_files

    def get_file_content_by_path(self, path: str) -> str:
        base_url = f"{self.github_api_url}/repos/{self.repo}/contents/{path}"
        #print(base_url)
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()

        content_encoded = response.json()["content"]
        return base64.b64decode(content_encoded).decode("utf-8")

# From tools/github_search_tool.py
class GitHubFileLoadAction(BaseAction):
    """
    #Use this Action to extract specific extension files from GitHub.
    """

    repo: str = Field(
        default_factory=str,
        description="Repository name- Format: username/repo e.g., aiplanethub/openagi",
    )
    directory:str = Field(
        default_factory=str,
        description="File directory that contains the supporting files i.e., src/openagi/llms",
    )
    extension: str = Field(
        default_factory = ".txt",
        description="File extension to extract the data from. eg: `.py`, `.md`",
    )


    def execute(self):
        access_token = os.environ.get("GITHUB_ACCESS_TOKEN")

        loader = OpenAGIGithubFileLoader(
            repo=self.repo,
            access_token=access_token,
            github_api_url="https://api.github.com",
            branch="main",
            file_filter=lambda files: files.startswith(self.directory) and files.endswith(self.extension),
        )

        data = loader.load()
        response = []
        for doc in data:
            response.append(f"{doc.page_content}\nMetadata{doc.metadata}")

        return "\n\n".join(response)

# From tools/github_search_tool.py
def get_file_paths(self) -> List[Dict]:
        base_url = (
            f"{self.github_api_url}/repos/{self.repo}/git/trees/" f"{self.branch}?recursive=1"
        )
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()
        all_files = response.json()["tree"]
        
        """ one element in all_files
        {
            'path': '.github', 
            'mode': '040000', 
            'type': 'tree', 
            'sha': '89a2ae046e8b59eb96531f123c0c6d4913885df1', 
            'url': 'https://github.com/api/v3/repos/shufanhao/langchain/git/trees/89a2ae046e8b59eb96531f123c0c6d4913885dxxx'
        }
        """
        required_files = [
            f for f in all_files if not (self.file_filter and not self.file_filter(f["path"]))
        ]
        return required_files

# From tools/github_search_tool.py
def get_file_content_by_path(self, path: str) -> str:
        base_url = f"{self.github_api_url}/repos/{self.repo}/contents/{path}"
        #print(base_url)
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()

        content_encoded = response.json()["content"]
        return base64.b64decode(content_encoded).decode("utf-8")

import praw

# From tools/reddit.py
class RedditSearch(ConfigurableAction):
    """
    Reddit Search Tool to search and retrieve posts/comments from Reddit using PRAW
    """
    query: str = Field(..., description="User query to search Reddit")
    subreddit: Optional[str] = Field(
        default=None,
        description="Specific subreddit to search in. If None, searches across all subreddits"
    )
    sort: str = Field(
        default="relevance",
        description="Sort method for results: 'relevance', 'hot', 'top', 'new', or 'comments'"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return (1-25)"
    )
    include_comments: bool = Field(
        default=False,
        description="Whether to include top comments in the results"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()
    
    def _check_deprecated_usage(self):
        required_keys = ['client_id', 'client_secret', 'user_agent']
        env_keys = {
            'REDDIT_CLIENT_ID': 'client_id',
            'REDDIT_CLIENT_SECRET': 'client_secret',
            'REDDIT_USER_AGENT': 'user_agent'
        }
        
        config_missing = any(not self.get_config(key) for key in required_keys)
        env_vars_present = any(key in os.environ for key in env_keys)
        
        if config_missing and env_vars_present:
            warnings.warn(
                "Using environment variables for Reddit credentials is deprecated and will be removed in a future version. "
                "Please use RedditSearch.set_config() instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(**{
                conf_key: os.environ.get(env_key)
                for env_key, conf_key in env_keys.items()
                if env_key in os.environ
            })

    def _init_reddit_client(self) -> praw.Reddit:
        """Initialize and return PRAW Reddit client"""
        client_id = self.get_config('client_id')
        client_secret = self.get_config('client_secret')
        user_agent = self.get_config('user_agent')
        
        if not all([client_id, client_secret, user_agent]):
            raise OpenAGIException(
                "Reddit credentials not found. Use RedditSearch.set_config() to set:"
                "\n- client_id: Your Reddit API client ID"
                "\n- client_secret: Your Reddit API client secret"
                "\n- user_agent: A unique identifier for your application"
            )
        
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def _format_submission(self, submission: praw.models.Submission, include_comments: bool) -> str:
        """Format a submission into a string with optional comments"""
        result = (
            f"TITLE: {submission.title}\n"
            f"SUBREDDIT: r/{submission.subreddit.display_name}\n"
            f"CONTENT: {submission.selftext if submission.is_self else '[External Link]'}\n"
            f"SCORE: {submission.score}\n"
            f"URL: https://reddit.com{submission.permalink}\n"
        )
        
        if include_comments:
            submission.comments.replace_more(limit=0)
            top_comments = submission.comments[:3]  # Get top 3 comments
            if top_comments:
                result += "TOP COMMENTS:\n"
                for comment in top_comments:
                    result += f"- {comment.body[:200]}... (Score: {comment.score})\n"
        
        return result + "\n"

    def execute(self) -> str:
        # Validate limit
        self.limit = max(1, min(25, self.limit))
        
        # Initialize Reddit client
        reddit = self._init_reddit_client()
        
        # Perform search
        if self.subreddit:
            search_results = reddit.subreddit(self.subreddit).search(
                self.query,
                sort=self.sort,
                limit=self.limit
            )
        else:
            search_results = reddit.subreddit("all").search(
                self.query,
                sort=self.sort,
                limit=self.limit
            )

        # Format results
        formatted_results = []
        for submission in search_results:
            formatted_results.append(
                self._format_submission(submission, self.include_comments)
            )

        if not formatted_results:
            return "No results found."
            
        return "\n".join(formatted_results).strip()

from googlesearch import search

# From tools/google_search_tool.py
class GoogleSearchTool(ConfigurableAction):
    """
    Google Search is a tool used for scraping the Google search engine. Extract information from Google search results.
    """
    query: str = Field(..., description="User query or question ")

    max_results: int = Field(
        default=10,
        description="Total results, in int, to be executed from the search. Defaults to 10. The limit should be 10 and not execeed more than 10",
    )

    lang: str = Field(
        default="en",
        description = "specify the langauge for your search results."
    )

    def execute(self):
        if self.max_results > 15:
            logging.info("Over threshold value... Limiting the Max results to 15")
            self.max_results = 15
        
        context = ""
        search_results = search(self.query,num_results=self.max_results,lang=self.lang,advanced=True)
        for info in search_results:
            context += f"Title: {info.title}. Description: {info.description}. URL: {info.url}"
        
        return context

from urllib.parse import urlencode

# From tools/searchapi_search.py
class SearchApiSearch(ConfigurableAction):
    """SearchApi.io provides a real-time API to access search results from Google (default), Google Scholar, Bing, Baidu, and other search engines."""
    query: str = Field(
        ..., description="User query of type string used to fetch web search results from a search engine."
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._check_deprecated_usage()

    def _check_deprecated_usage(self):
        if 'SEARCHAPI_API_KEY' in os.environ and not self.get_config('api_key'):
            warnings.warn(
                "Using environment variables for API keys is deprecated and will be removed in a future version. "
                "Please use SearchApiSearch.set_config(api_key='your_key', engine='google') instead of setting environment variables.",
                DeprecationWarning,
                stacklevel=2
            )
            self.set_config(api_key=os.environ['SEARCHAPI_API_KEY'], engine='google')

    def execute(self):
        base_url = "https://www.searchapi.io/api/v1/search"
        api_key = self.get_config('api_key')
        engine = self.get_config('engine', 'google')  # Default to google if not set

        search_dict = {
            "q": self.query,
            "engine": engine,
            "api_key": api_key
        }

        logging.debug(f"{search_dict=}")

        url = f"{base_url}?{urlencode(search_dict)}"
        response = requests.request("GET", url)
        json_response = response.json()

        organic_results = json_response.get("organic_results", [])

        meta_data = ""
        for organic_result in organic_results:
            meta_data += f"CONTEXT: {organic_result['title']} \ {organic_result['snippet']}"
            meta_data += f"Reference URL: {organic_result['link']}\n"
        return meta_data

