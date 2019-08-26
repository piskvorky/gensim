#!/bin/bash
pip install --upgrade pip
pip install --timeout=60 --trusted-host 28daf2247a33ed269873-7b1aad3fab3cc330e1fd9d109892382a.r6.cf2.rackcdn.com --find-links http://28daf2247a33ed269873-7b1aad3fab3cc330e1fd9d109892382a.r6.cf2.rackcdn.com/ "$@"
