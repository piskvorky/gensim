#!/usr/bin/env python

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


from __future__ import print_function
import collections
import os
import sys
import time
import numpy as np

import psutil


if not hasattr(psutil.Process, "cpu_num"):
    sys.exit("platform not supported")


def clean_screen():
    if psutil.POSIX:
        os.system('clear')
    else:
        os.system('cls')


def main():
    loads = []
    while True:
        # header
        clean_screen()
        cpus_percent = psutil.cpu_percent(percpu=True)
        loads.append(sum(cpus_percent))

        perc25 = np.percentile(loads, 25)
        perc50 = np.median(loads)
        perc75 = np.percentile(loads, 75)
        perc90 = np.percentile(loads, 90)
        perc95 = np.percentile(loads, 95)
        perc99 = np.percentile(loads, 99)
        avg = np.mean(loads)

        print("25% perc : {:.2f}".format(perc25))
        print("50% perc : {:.2f}".format(perc50))
        print("75% perc : {:.2f}".format(perc75))
        print("90% perc : {:.2f}".format(perc90))
        print("95% perc : {:.2f}".format(perc95))
        print("99% perc : {:.2f}".format(perc99))
        print("avg perc : {:.2f}".format(avg))

        time.sleep(1)


if __name__ == '__main__':
    main()