#!/bin/bash
#$1:the start index of cluster eg；ssd32-32
#$2:the end index of cluster 
#$3:target port

#global variable
pid=""

get_pid(){
#get pid of port
#$1 ip
#$2 target port
#$3 pid
re=`ssh $1 "netstat -anp|grep "$2`
tem=${re%%/*}
pid=${tem##* }
if [ "$pid" == "" ]
then
    echo "the port:"$2" of "$1" is free"
else
    echo "the port:"$2" of "$1" has been occpuied by process："$pid
    echo "kill pid:"$pid
    ssh $1 kill $pid
    echo "clean"
fi
}

for((step=$1;step<=$2;step++));
do
    if [ $step == 35 ]
    then
	re=`netstat -anp|grep $3`
	tem=${re%%/*}
	pid=${tem##* }
	if [ "$pid" != "" ]
	then
	    kill $pid
	fi
    else
	get_pid "ssd"$step $3
    fi
done
echo "clean up"
