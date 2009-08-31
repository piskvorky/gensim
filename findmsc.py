import urllib2
import logging
import re
import time
from Article import areValidMSCs

ZENTRALBLATT_DELAY = 0.0
PAT_ZB = re.compile("(<a.*?>)(.*?)(</a>)")

last_time = 0

def findmsc(url):
    global last_time
    """decide what kind of url this is, load the document and parse out the list of MSCs"""
    try:
        if url.find('zentralblatt') >= 0 and time.time() - last_time < ZENTRALBLATT_DELAY:
            time.sleep(time.time() - last_time)
        response = urllib2.urlopen(url)            
        last_time = time.time()
    except urllib2.URLError, e:
        if hasattr(e, 'reason'):
            logging.error('failed to reach server: %s (%s)' % (e.reason, url))
        elif hasattr(e, 'code'):
            logging.error('failed to get response: %s (%s)' % (e.code, url))
        return None
    result = []
    if url.find('zentralblatt') >= 0:
        html = response.read()
        mscline = html[html.find('<i>MSC') :]
        mscline = mscline[ : mscline.find('</dd>')]
        mscs = [match[1] for match in PAT_ZB.findall(mscline)]
        if not areValidMSCs(mscs):
            logging.warning("suspicious MSC=%s for %s" % (mscs, url))
        else:
            result.extend(mscs)
    elif url.find('ams.org') >= 0:
        for line in response:
            if line.strip().startswith('<strong>MR'):
                while not line.strip().startswith('</strong>'):
                    line = response.next()
                mscline = response.next().strip()
                while len(mscline) > 0:
                    if mscline.startswith('('):
                        mscs = mscline[1 : -1].split() # get rid of () for secondary MSCs
                    else:
                        mscs = ['*' + mscline] # prepend * to primary MSC area, ala ZentralBlatt
                    if not areValidMSCs(mscs):
                        logging.warning("suspicious MSC=%s for %s" % (mscs, url))
                    else:
                        result.extend(mscs)
                    mscline = response.next().strip()
    return result

if __name__ == "__main__":
    _urlzb = "http://www.zentralblatt-math.org/zmath/en/search/?q=an:0102.10201"
    urlzb = "http://www.zentralblatt-math.org/zmath/en/search/?q=an:0119.18101"
    _urlmr = "http://www.ams.org/mathscinet-getitem?mr=1500810"
    urlmr = "http://www.ams.org/mathscinet-getitem?mr=0123188"

    print '======================\n'
    print findmsc(_urlzb)    
    print findmsc(urlzb)
    print '======================\n'    
    print findmsc(_urlmr)
    print findmsc(urlmr)    