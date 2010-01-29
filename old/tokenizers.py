##vhodne metody tokenizace --
##    pro kod: nahrazeni promennych, ruzne jazyky (aspon c++, java, prolog)...
##    pro text: ruzne jazyky - ang ok, cestina spolu s lemmatizaci
##    jine typy textovych dat?

import re
import unicodedata
import sys

UNIS = [unichr(i) for i in xrange(sys.maxunicode)]
PAT_ANUM = re.compile('\w+', re.UNICODE)
PAT_HTML = re.compile('<.*?>', re.UNICODE)
PAT_AB = re.compile('[%s]+' % ''.join(ch for ch in UNIS if unicodedata.category(ch)[0] == 'L'), re.UNICODE)

class TokenizerFactory:
    """public class; all requests for tokenizer constructions should go through here"""
    
    def createTokenizer(self, contentType='text', encoding='utf8', toLowercase=False):
        if contentType == 'text':
            return TextTokenizer(encoding, toLowercase)
        elif contentType == 'code':
            return CodeTokenizer(encoding, toLowercase)
        elif contentType == 'math':
            return AlphanumMathTokenizer(encoding, toLowercase)
        elif contentType == 'alphabet':
            return AlphabetTokenizer(encoding, toLowercase)
        elif contentType == 'alphanum':
            return AlphanumTokenizer(encoding, toLowercase)
        elif contentType == 'alphanum_nohtml':
            return AlphanumTokenizerNoHTML(encoding, toLowercase)
        else:
            raise "unknown tokenizer type"

class TextTokenizer:
##        TODO: nejaky tokenizer, ktery nahrazuje cislovky atd vyssi sem. kategorii (nejaky dummy token NUM nebo tak)
    
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        """Tokenize string into list of strings."""
        import re, unicodedata

        tokens    = []
        positions = []
        if isinstance(string, unicode):
            unicode_string = string
        else:
            unicode_string = unicode(string, self.encoding)

        if self.toLowercase:
            unicode_string = unicode_string.lower()

        # for all non-whitespace sequences
        for non_ws_matcher in re.finditer("\S+", unicode_string, re.U):
            non_ws    = non_ws_matcher.group()
            start_pos = non_ws_matcher.start()
            subtok    = []

            # for all alphanumeric or nonalphanumeric sequences
            for token_matcher in re.finditer("(\w+|\W+)", non_ws, re.U):
                token = token_matcher.group()
                if unicodedata.category(token[0]) in ("Ll", "Lu", "Nd") or token[0] == "_":
                    # if alphanumeric use whole token
                    subtok.append(token)
                    positions.append(start_pos + token_matcher.start())
                else:
                    # if nonalphanumeric split token to sequences of
                    # cooccurring characters
                    for subtoken_matcher in re.finditer("(.)\\1*", token):
                        subtok.append(subtoken_matcher.group())
                        positions.append(start_pos + token_matcher.start() + subtoken_matcher.start())

            tokens += subtok

        #return [x.encode('utf8') for x in tokens]
        if returnPositions:
            return (tokens, positions)
        else:
            return tokens

class AlphanumTokenizer:
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        if returnPositions:
            raise Exception("not implemented yet")
        if not isinstance(string, unicode):
            string = unicode(string, self.encoding)

        if self.toLowercase:
            string = string.lower()
        return re.findall(PAT_ANUM, string)

class AlphabetTokenizer:
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        if returnPositions:
            raise Exception("not implemented yet")
        if not isinstance(string, unicode):
            string = unicode(string, self.encoding)

        if self.toLowercase:
            string = string.lower()
        return re.findall(PAT_AB, string)

class AlphanumTokenizerNoHTML:
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        if returnPositions:
            raise Exception("not implemented yet")
        if not isinstance(string, unicode):
            string = unicode(string, self.encoding)
        string = re.sub(PAT_HTML, "", string)
        if self.toLowercase:
            string = string.lower()
        return re.findall(PAT_ANUM, string)

class AlphanumMathTokenizer:
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        if returnPositions:
            raise Exception("not implemented yet")
        if not isinstance(string, unicode):
            string = unicode(string, self.encoding)
        if self.toLowercase:
            string = string.lower()
        s = string.split('$')
        result = []
        for i in xrange(len(s)):
            if i % 2 == 0:
                result.extend(re.findall(PAT_ANUM, s[i]))
            else:
                exp = s[i].strip()
                if exp.find('\n') < 0:
                    if exp:
                        result.append('$' + exp + '$')
        return result
         
class CodeTokenizer:
    
    def __init__(self, encoding = 'utf8', toLowercase = False):
        self.encoding = encoding
        self.toLowercase = toLowercase

    def tokenize(self, string, returnPositions = False):
        """Tokenize string into list of strings."""
##        TODO: kolik domenove znalosti? pro kazdy jazyk zvlast? promenne -> dummy symbol, cisla -> dummy symbol atd
        raise "CodeTokenizer not implemented yet"
    
