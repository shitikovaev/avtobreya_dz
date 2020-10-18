import re
import string
import os
import nltk

nltk.download('punkt')

dir, script = os.path.split(os.path.realpath(__file__))


def getText(filename):
    with open(dir + filename, "r") as f:
        text = f.read()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("“", "", text)
    text = re.sub("”", "", text)
    text = re.sub("–", "", text)
    return text


def natashaTag(text):
    from natasha import (
        Segmenter,
        NewsEmbedding,
        NewsMorphTagger,
        Doc
    )
    segmenter = Segmenter()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    result = ""
    for item in doc.tokens:
        result += item.text + " [" + str(item.pos) + "] "
    return result


def pyMorphyTag(text):
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    words = nltk.word_tokenize(text)
    result = ""
    for word in words:
        parsed = morph.parse(word)[0]
        result += word + " [" + str(parsed.tag.POS) + "] "
    return result


def myStemTag(text):
    import pymystem3
    myStem = pymystem3.Mystem()
    parsed = myStem.analyze(text)
    result = ""
    for item in parsed:
        try:
            temp = item["analysis"]
            tag = str(item["analysis"][0]["gr"].split(",")[0])
            tag = tag.split("=")[0]
            result += item["text"] + " [" + tag + "] "
        except:
            pass
    return result


def flairTag(text):
    from flair.data import Sentence
    from flair.models import SequenceTagger
    sentences = nltk.tokenize.sent_tokenize(text)
    tagger = SequenceTagger.load('pos')
    result = ""
    for sentence in sentences:
        sentence = Sentence(sentence)
        tagger.predict(sentence)
        sentence.clear_embeddings()
        result += sentence.to_tagged_string()
    result = re.sub("<", "[", result)
    result = re.sub(">", "]", result)
    return result


def spacyTag(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    result = ""
    for i, s in enumerate(doc.sents):
        for t in s:
            if t.pos_ != 'SPACE' and t.pos_ != 'PUNCT':
                result += t.text + " [" + str(t.pos_) + "] "
    return result


def nltkTag(text):
    nltk_text = nltk.word_tokenize(text)
    nltk.download('averaged_perceptron_tagger')
    result = ""
    for item in nltk.pos_tag(nltk_text):
        result += item[0] + " [" + str(item[1]) + "] "
    return result


def unifyTags(text):
    pos = {}
    pos["noun"] = ["NN", "NNS", "S", "NOUN", "NNP", "PROPN"]
    pos["verb"] = ["VBD", "VB", "MD", "VBP", "V", "VERB", "INFN", "VBZ", "AUX", "VBN"]
    pos["adj"] = ["JJ", "A", "ADJ", "ADJF", "ADJS"]
    pos["pronoun"] = ["PRP", "WP", "PRP$", "ADVPRO", "APRO", "SPRO", "PRON", "NPRO", "WDT"]
    pos["article"] = ["DT", "DET"]
    pos["adverb"] = ["WRB", "ADV", "ADVB", "RB"]
    pos["conj"] = ["CONJ", "CC", "CCONJ", "SCONJ"]
    pos["num"] = ["NUMR", "CD", "NUM"]
    pos["prep"] = ["ADP", "IN", "PR", "RP", "PREP"]
    pos["clitic"] = ["PART", "PRCL"]
    pos["interjection"] = ["UH"]

    words = text.split(" ")
    res = []
    for word in words:
        if word.startswith("["):
            word = re.sub("\[", "", word)
            word = re.sub("]", "", word)
            for part in pos:
                if word in pos[part]:
                    res.append(part)
                    break

        else:
            res.append(word)

    return " ".join(res)


def checkAccuracy(machineTagged, tagged):
    machineWords = machineTagged.split(" ")
    words = tagged.split(" ")
    i = 0
    correct = 0
    print(machineWords)
    print(words)
    while i < len(words):
        if words[i] != machineWords[i]:
            print("ERROR!!!")
            print(words[i], machineWords[i])
        i += 1
        if words[i] == machineWords[i]:
            correct += 1
        i += 1
    return correct * 2 / len(words)


text = getText("/rus.txt")
tagged_text = getText("/rus_tagged.txt")
print("NATASHA")
print(checkAccuracy(unifyTags(natashaTag(text)), tagged_text))
print()


print("PYMORPHY")
print(checkAccuracy(unifyTags(pyMorphyTag(text)), tagged_text))
print()


print("MYSTEM")
print(checkAccuracy(unifyTags(myStemTag(text)), tagged_text))
print()

text = getText("/eng.txt")
tagged_text = getText("/eng_tagged.txt")
print("FLAIR")
print(checkAccuracy(unifyTags(flairTag(text)), tagged_text))
print()

print("SPACY")
print(checkAccuracy(unifyTags(spacyTag(text)), tagged_text))
print()


print("NLTK")
print(checkAccuracy(unifyTags(nltkTag(text)), tagged_text))
print()

