# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:43:11 2016

@author: robert
"""

#TWEAK TIME

import pickle
import os
import re
# use nltk.download() to download stopwords corpus if not yet




# Iterable to be passed to word2vec class as sentences.
# reads sentences one by one from picke dump



#stops = set(stopwords.words('russian'))

stops = ['чтоб', 'между', 'какой', 'без', 'но', 'чуть', 'для', 'не', 'куда',
            'себя', 'всего', 'даже', 'был', 'кто', 'уж', 'только', 'с', 'быть',
            'теперь', 'много', 'по', 'надо', 'когда', 'этого',
            'три', 'и', 'опять', 'или', 'под', 'более', 'эти', 'бы', 'чем',
            'совсем', 'сам', 'раз', 'хоть', 'нибудь', 'него', 'уже', 'сейчас', 
            'никогда', 'о', 'ни', 'можно', 'ли', 'потому', 'тем', 'будто', 
            'в', 'перед', 'так', 'два', 'ничего', 'а', 'почти', 'может',
            'было', 'эту', 'их', 'нет', 'впрочем', 'им', 'во', 'лучше',
             'до', 'про', 'вот', 'после', 'что', 'зачем', 'иногда', 
            'ее', 'другой', 'больше', 'тоже', 'еще', 'от', 'у', 'потом', 'всю',
            'над', 'этой', 'за', 'если',  'ж', 'там', 'есть',
            'через', 'из', 'как', 'на', 'чтобы', 'такой', 'том',
            'да', 'этом', 'хорошо', 'к', 'при', 'были', 'себе', 
            'чего',  'ней', 'то',  'вам', 'один', 'вдруг', 'со', 
            'тогда', 'будет', 'разве', 'нельзя', 'наконец', 'ведь', 'здесь',
            'тот', 'какая', 'этот', 'же', 'где', 'ну', 'конечно',  
            'того', 'тут', 'была',  'всегда', 'свою', 'об', 'всех']

# futureStops = ['кто', 'что']


# Looping over the corpus and generating pickle dump file that would give off
# context pairs one by one

def createContext(root_directory):
    
    pickleDump = open('/run/media/robert/1TB-1/linuxfolder/pythonworks/contDumpFinal', 'ab')
    dumpCounter = 0

# walking the corpus dir
# files walked linewise


    for root, dirs, files in os.walk(root_directory):
            for fname in filter(lambda fname: fname.endswith('.conll'), files):
                

                document = open(os.path.join(root, fname), 'r')
                print('Opened document ' + fname)
                
                
                wordCounter = -1
                sentDict = {}
                sentCash = []
                for line in document:
 
                    if len(line)<5:
                        continue
                    line = line.lower()
                    line = line.split()
                                        # Creating cash dictionary for sentence

                    wordCounter += 1
                    if wordCounter < int(line[0]):

                        if re.match('[A-Za-zА-Яа-я]+$', line[2]) != None:
                            sentDict.update({line[0]:{'word':line[2],'ref':line[6]}})

                            
                        else:
                            sentDict.update({line[0]:{'word':None,'ref':line[6]}})

                            
                    else:
                        wordCounter = 0
                                            # Creating a sentence (context pair) to be passed to word2vec later
                        for slot in sentDict:
                            if sentDict[slot]['word'] == None:
                                continue
                            if sentDict[slot]['word'] in stops:
                                
                                continue
                            sentCash.append(sentDict[slot]['word']) # append target word if it is okay
       # looking into word that's higher in hyerarchy
                            if (sentDict[slot]['ref'] != 0 and sentDict[slot]['ref'] != '0'):
                                wordRef = sentDict[slot]['ref']
                                refCounter = 0
                                while refCounter < 10:
                                    refCounter += 1
                    
                    #cycling through dependent word chain until good word fould or 10 tries
                                    

                                    try:
                                        if sentDict[wordRef]['word'] in stops:
  
                                            wordRef = sentDict[wordRef]['ref']
                                       
                                        else:
                                            refCounter = 10
        
                                            try:
                                        
                                                sentCash.append(sentDict[sentDict[slot]['ref']]['word'])
    #                                            print(sentCash)
                                                
                                            except:
                                                continue
                                    except:
                                        pass
        # looking into dependent words
        # cycling through all words in a sentence again
                            for slot2 in sentDict:
                                if sentDict[slot2]['ref'] == slot:
                                    if sentDict[slot2]['word'] != None:
                                        if re.match('[A-Za-zА-Яа-я]+$', sentDict[slot2]['word']) != None:
                                            if sentDict[slot2]['word'] not in stops:
                                                sentCash.append(sentDict[slot2]['word'])
                                        # if okay, stop here
#
                                    if (sentDict[slot2]['word'] == None) or (sentDict[slot2]['word'] in stops):
                                        checkedSlot = slot2
                                        slotCounter = 0
                                        while slotCounter < 10:
#                                            print('SPASITE2')
                                            slotCounter += 1
                                            for slot3 in sentDict:
                                                if sentDict[slot3]['ref'] == checkedSlot:
                                                    
                                                    if (sentDict[slot3]['word'] == None) or (sentDict[slot3]['word'] in stops):
#                                                        print(str(sentDict[slot3]['word']) + ' is  BAD WORD FROM SECOND CYCLE!')
                                                        checkedSlot = slot3
                                                        slotCounter += 1
                                                    else:
 #                                                       print(sentDict[slot3]['word'] +  ' is a GOOD WORD FROM SECOND CYCLE!')
                                                        sentCash.append(sentDict[slot3]['word'])
                                                        slotCounter = 10
                        # veryfying no stopwords slipped
                            for k in filter(lambda k: k in stops, sentCash):
                                sentCash.remove(k)
                            if len(sentCash) > 1:
#                                print('Dumping.....')
                                pickle.dump(sentCash,pickleDump)
                        #pickling to a file
                                dumpCounter += 1
                            sentCash = []
                        sentDict = {}
                        if re.match('[A-Za-zА-Яа-я]+$', line[2]) != None:
                            sentDict.update({line[0]:{'word':line[2],'ref':line[6]}})
                        else:
                            sentDict.update({line[0]:{'word':None,'ref':line[6]}})

    pickleDump.close()
    return(dumpCounter)
                            

                    
                            
                        
                    
                    
                                    
            
            
            
            
      
