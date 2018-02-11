#!/bin/csh
        tcov -x cm.profile cs*.c >& /dev/null
        echo -n "statments not yet tested: "
        ./covs > covs.out
        grep "#####" *tcov | wc -l
