import nltk
import pickle
from fuzzywuzzy import fuzz

def model_prediction(model,X):
    pred = model.predict(X) # making prediciotns
    scores = model.predict_marginals(X) # getting scores
    
    return pred,scores


def append_result(result, name, value, isLineItem, LineItemSeq, bbox, index,score):
    result.append({
        "Name": name,
        "Value": value,
        "IsLineItem": isLineItem,
        "LineItemSequence": LineItemSeq,
        "bounding_box": bbox,
        "offsets":index ,
        "ConfidenceScore": score})
    return result

def postprocessing(data,pred,scores):
    result = []
    arr_char_offset = []
    arr = data.split()
    start = 0
    end = 0
    for i in range(0,len(arr),1):
        word = arr[i]
        end = end+len(word)
        end = end+1
        if i==0:
            start = 0
        else:
            start = arr_char_offset[i-1][2]
            
        arr_char_offset.append([word,start,end])

    res = ""
    count = 0
    flag = 0
    conf = 0
    for i in range(0,len(arr),1):
        ans = pred[0][i]
        score = scores[0][i][ans]
        word = arr[i]
        
        if ans!="O":
            if flag==0:
                start = arr_char_offset[i][1]
                # print(arr_char_offset[i][1],arr_char_offset[i][2])
                flag = 1
                
            res = res+word+" "
            conf+=score # adding all the socres
            count+=1 # adding the count 
            label = ans # getting the label name
            end = arr_char_offset[i][2]
            # print(arr_char_offset[i][1],arr_char_offset[i][2])
            
        else:
            if count!=0:
                res = res.strip()
                if start==end:
                    end = start+1
                    result = append_result(result, label, res, False, "", [], [{"EndOffset":end,"StartOffset":start}],conf/count)
                else:
                    result = append_result(result, label, res, False, "", [], [{"EndOffset":end-1,"StartOffset":start}],conf/count)
            
            flag = 0
            count = 0
            conf = 0
            res="" 

    return result


def data_processing(data):
    ans = pos_tagger(data)
    X = [sent2features(ans)]
    return X

def pos_tagger(sentence):
    arr = []
    arr_split = sentence.split()
    tagged = nltk.pos_tag(arr_split) 
    return tagged


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    
    if i > 4:
        word5 = sent[i-5][0]
        postag5 = sent[i-5][1]
        features.update({
            '-5:word.lower()': word5.lower(),
            '-5:word.istitle()': word5.istitle(),
            '-5:word.isupper()': word5.isupper(),
            '-5:postag': postag5,
            '-5:postag[:2]': postag5[:2],
        })
    if i > 3:
        word4 = sent[i-4][0]
        postag4 = sent[i-4][1]
        features.update({
            '-4:word.lower()': word4.lower(),
            '-4:word.istitle()': word4.istitle(),
            '-4:word.isupper()': word4.isupper(),
            '-4:postag': postag4,
            '-4:postag[:2]': postag4[:2],
        })
    if i > 2:
        word3 = sent[i-3][0]
        postag3 = sent[i-3][1]
        features.update({
            '-3:word.lower()': word3.lower(),
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:postag': postag3,
            '-3:postag[:2]': postag3[:2],
        })
    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
        })
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
      
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
        })  
    
    if i < len(sent)-3:
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        features.update({
            '+3:word.lower()': word3.lower(),
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:postag': postag3,
            '+3:postag[:2]': postag3[:2],
        })  
    
    if i < len(sent)-4:
        word4 = sent[i+4][0]
        postag4 = sent[i+4][1]
        features.update({
            '+4:word.lower()': word4.lower(),
            '+4:word.istitle()': word4.istitle(),
            '+4:word.isupper()': word4.isupper(),
            '+4:postag': postag4,
            '+4:postag[:2]': postag4[:2],
        })  
    
    if i < len(sent)-5:
        word5 = sent[i+5][0]
        postag5 = sent[i+5][1]
        features.update({
            '+5:word.lower()': word5.lower(),
            '+5:word.istitle()': word5.istitle(),
            '+5:word.isupper()': word5.isupper(),
            '+5:postag': postag5,
            '+5:postag[:2]': postag5[:2],
        }) 
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]


def append_result(result, name, value, isLineItem, LineItemSeq, bbox, index,score):
    result.append({
        "Name": name,
        "Value": value,
        "IsLineItem": isLineItem,
        "LineItemSequence": LineItemSeq,
        "bounding_box": [bbox],
        "offsets":[index] ,
        "ConfidenceScore": score})
    return result




