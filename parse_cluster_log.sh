#!/bin/bash


LOGFN=$1

cat $LOGFN | grep Loss | awk -F ' ' '{print $9}' > /tmp/$LOGFN.$$
perl -pe 's/\x1b\[[^m]+m//g' /tmp/$LOGFN.$$ > /tmp/$LOGFN.$$.1
cat -n /tmp/$LOGFN.$$.1 | awk -F ' ' '{print $1 ", " $2}' > $LOGFN.loss.aux
/bin/rm -rf /tmp/$LOGFN.*

