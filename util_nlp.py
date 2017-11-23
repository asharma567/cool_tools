def NMF(corpus, n_topics, n_top_words):
    from sklearn import decomposition
    from time import time


    #timing the clustering process
    t0 = time()
    tfidf = instantiate_tfv()

    #NMF for grouped comments
    tfidf_vectorized_corpus = tfidf.fit_transform(corpus)

    vocab = tfidf.get_feature_names()
    nmf = decomposition.NMF(n_components=n_topics).fit(tfidf_vectorized_corpus)

    print("done in %0.3fs." % (time() - t0))
    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join( [vocab[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))        

def instantiate_tfv(ngrams=(1,3), top_n_features=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    '''
    
    '''
    tfidf = TfidfVectorizer(
                        analyzer='word',
                        stop_words=set(stopwords.words('english')),
                        sublinear_tf=True,
                        ngram_range=ngrams,
                        smooth_idf=True,
                        max_features=top_n_features
                        )
    return tfidf


def clean_html(html):
    import re
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()

def unescape(text):
    import re, htmlentitydefs
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)



def tokenize_and_normalize(document):
    #this isn't ideal from an nlp standpoint because it affects normalization
    from nltk import tokenize
    from nltk.corpus import stopwords
    from string import punctuation
    
    html_encoding_removed = clean_html(unescape(document))
    special_chars_removed = _special_char_translation(html_encoding_removed)
    abbrev_removed = multiple_replace(special_chars_removed)

    stops_removed_doc = _remove_stop_words (abbrev_removed)

    from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer

    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    
    punc_removed = ''.join([char for char in stops_removed_doc if char not in set(punctuation)])    
    stripped_lemmatized = map(wordnet.lemmatize, punc_removed.split())
    stripped_lemmatized_stemmed = map(snowball.stem, stripped_lemmatized)
    
    return ' '.join([word for word in stripped_lemmatized_stemmed if len(word) > 1])

def _special_char_translation(doc):
    from unidecode import unidecode
    return ' '.join([unidecode(word) for word in doc.split()])

def _remove_stop_words(doc):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    return ' '.join([word for word in doc.split() if word.lower() not in stopwords])



def normalize(document, post_normalization_stop_words={}):
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer

    SNOWBALL = SnowballStemmer('english')
    WORDNET = WordNetLemmatizer()
    WHITE_SPACE = ' '

    decoded_doc = _special_char_translation(document)
    abbreviations_removed_doc = _multiple_replace(decoded_doc)
    stops_removed_doc = _remove_stop_words (abbreviations_removed_doc)
    punc_removed = ''.join([char for char in stops_removed_doc if char not in set(punctuation)])    
    
    stripped_lemmatized = map(WORDNET.lemmatize, punc_removed.split())
    stripped_lemmatized_stemmed = map(SNOWBALL.stem, stripped_lemmatized)
    
    return WHITE_SPACE.join([word for word in stripped_lemmatized_stemmed if word not in post_normalization_stop_words])

ABREVIATIONS_DICT = {
    "'m":' am',
    "'ve":' have',
    "'ll":" will",
    "'d":" would",
    "'s":" is",
    "'re":" are",
    "  ":" ",
    "' s": " is",
    
    #debatable between and/or
    "/":" and "
}

def _multiple_replace(text, adict=ABREVIATIONS_DICT):
    import re
    '''
    Does a multiple find/replace
    '''
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text.lower())

def _special_char_translation(doc):    
    return ' '.join([unidecode(word) for word in doc.split()])
    
def _remove_stop_words(doc):
    STOPWORDS_SET = set(stopwords.words('english'))
    return ' '.join([word for word in doc.split() if word.lower() not in STOPWORDS_SET])




def get_titles_in_corpus(list_of_ngrams, corpus):
    output=[]
    
    for ngram in list_of_ngrams:
        for title in corpus:
            if ngram in title:
                output.append(title)
        
    
    return set(output)

def str_to_countidf_array(input_str, corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    '''
    This is a modification to tfidf. It maintains the scarcity weighting of IDF: log(N/df) but it yields 
    a wordcount not relative to the document size. So if there's a search: 'some word' the size of the 
    actual search term document length has no diminishing affects for searching the underlying terms.

    It just takes away the normalization by document length in tf. That's the only diff versus tfidf.

    todos:
    -----
    needs to be modularized
    '''

    # compute the idfs from corpus
    tfv = TfidfVectorizer(
                    norm=None,
                    analyzer='word',
                    sublinear_tf=True,
                    ngram_range=(1,1),
                    smooth_idf=True,
                )

    cnt_vect = CountVectorizer(
            analyzer='word',
            ngram_range=(1,1)
        )

    cnt_vect.fit(corpus)
    tfv.fit(corpus)
    
    #get these for analysis sake
    vocab_w_idf = dict(zip(tfv.get_feature_names(), tfv.idf_))
    
    #check if columns are aligned
    non_matches = [item for item in zip(cnt_vect.get_feature_names(), tfv.get_feature_names()) if  item[0] != item[1]]
    
    if non_matches:
        print non_matches
        raise Exception("vocabs don't match")    

    #grab indices for the word counts
    #motivation being that it's a sparse matrix
    input_arr = cnt_vect.transform([input_str]).todense()
    input_arr[input_arr != 0]
    _, indices_for_counts = input_arr.nonzero()

    #indices for counts
    counts = input_arr[0, indices_for_counts]
    idf_weights = tfv.idf_[indices_for_counts]

    #multiply the word counts and idf weights
    ctidf_arr = [ct * idf for ct, idf in zip(counts.A.tolist()[0], idf_weights.tolist())]

    #make an array
    zeroes_arr = np.zeros(input_arr.shape)
    zeroes_arr[:, indices_for_counts] = ctidf_arr

    return zeroes_arr
