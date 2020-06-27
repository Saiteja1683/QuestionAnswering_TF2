
import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from c_text_word_2_number import text2int


nlp = spacy.load('en', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
stopword_list.extend('&')
stopword_list.extend('@')
stopword_list.extend('#')
stopword_list.extend('$')
stopword_list.extend('%')
stopword_list.extend('^')
stopword_list.extend(';')

units = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
        'hundred', 'thousand', 'million', 'billion', 'trillion']

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # remove special characters    
        if special_char_removal:
            doc = remove_special_characters(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

def word_2_numbers(context_line):
    tokens = nltk.word_tokenize(context_line[0])
    nums_idx = []
    idx = 0
    for n,word in enumerate(tokens):
        if word in stopword_list:
            tokens.remove(word)
        if word in units:
            nums_idx.append(n)

    dup = nums_idx.copy()
    dic = {}
    for id in nums_idx:
        if id+1 not in dup:
            continue
        else:
            if id+2 not in dup and len(dup)>2:
                new_word = tokens[id]+' '+tokens[id+1]
                dup.remove(id)
                dup.remove(id+1)
                dic[id] = new_word
                
            else:
                if id+3 not in dup and len(dup)>3:
                    new_word = tokens[id]+' '+tokens[id+1]+' '+tokens[id+2]
                    dup.remove(id)
                    dup.remove(id+1)
                    dup.remove(id+2)
                    dic[id] = new_word
                    
                else:    
                    if id+4 not in dup and len(dup)>4:
                        new_word = tokens[id]+' '+tokens[id+1]+' '+tokens[id+2]+' '+tokens[id+3]
                        dup.remove(id)
                        dup.remove(id+1)
                        dup.remove(id+2)
                        dup.remove(id+3)
                        dic[id] = new_word
                        
    idx_values = dic.keys()
    new_context = tokens.copy()
    for i in idx_values:
        new_context[i] = dic[i]
        skip_length = len(dic[i].split())
        for removee in range(i+1,i+skip_length):
            try:
                new_context.remove(new_context[removee])
            except:
                continue

    return new_context



# cont_lines = [line.rstrip('\n') for line in open('./data/data/contexts.txt')]
# answer = [line.rstrip('\n') for line in open('./data/data/answers.txt')]



# pre = 0

# for i in range(0,252):
#     print(i)
#     test_line = cont_lines[i]
#     norm_line = normalize_corpus([test_line], text_lemmatization=False, stopword_removal=True, text_lower_case=True)
#     print('***************')
#     processed_tokens = word_2_numbers(norm_line)
#     final_context = []
#     for toks in processed_tokens:
#         try:
#             final_context.append(text2int(toks))
#         except:
#             final_context.append(toks)

#     ans = answer[i].lower()
#     print(ans)
    
#     print(final_context)
#     # for w in final_context:
#     #     if w==str(ans):
#     #         print(w,ans)
#     #     else:
#     #         print(ans,'not present')


#     print(type(final_context))
#     if ans in final_context:
#         print(ans,'present')
#         pre = pre+1
#     else:
#         ans_lis_1 = ans.split('.')
#         ans_lis_2 = ans.split(' ')
#         print(ans_lis_2)
#         #print(ans_lis_1)
#         #if ans_lis_1[0] in final_context:
#         #    print(ans_lis_1,'1111')
#         for k in range(len(final_context)):
#             #print(w)
#             if final_context[k]==ans_lis_2[0]:
#                 print('ci')
    
# print(pre,'hahahah')
    
    

    