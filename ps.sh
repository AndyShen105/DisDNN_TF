#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# $3 is the optimizer of model
# $4 is the targted_accuracy of model
# $5 is the tensorflow port
# $6 is the filepath of result
# ps.sh run in ssd35

get_ps_conf(){
    ps=""
    for(( i=35; i > 35-$1; i-- ))
    do
        ps="$ps,ssd$i:"$2
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=35-$1; i > 35-$1-$2; i-- ))
    do
        worker="$worker,ssd$i:"$3
    done
    worker=${worker:1}
};

for(( i=0; i<$2; i++ ))
do
{
    echo "0">temp$i
}
done

echo "release port occpied!"
./kill_cluster_pid.sh 26 35 $5

get_ps_conf $1 $5
echo $ps
get_worker_conf $1 $2 $5
echo $worker

#rm -rf ./result/$1"-"$2
#mkdir ./result/$1"-"$2
flag_arr[0]=0
for(( i=35; i>35-$1-$2; i-- ))
do
{
    if [ $i == 35 ]
    then
	source /root/anaconda2/envs/tensorflow/bin/activate
        python /root/code/disDNN/disDNN.py $ps $worker --job_name=ps --task_index=0
    else
	ssh ssd$i "source activate tensorflow"
        n=`expr 35 - $1`
        if [ $i -gt $n ]
        then
            index=`expr 35 - $i`
            ssh ssd$i python /root/code/disDNN/disDNN.py $ps $worker --job_name=ps --task_index=$index
        else
            index=`expr 35 - $1 - $i`
	    ssh ssd$i python /root/code/disDNN/disDNN.py $ps $worker --job_name=worker --task_index=$index --targted_accuracy=$4 --optimizer=$3 #>> /root/DMLcode/result/$6
            echo "worker"$index" complated"
	    echo "1">temp$index
	fi
    fi
}&
done
flag=0
while true
do
    for(( i=0; i<$2; i++ ))
    do
    {   
	tem=`cat temp$i`
	flag=`expr $tem + $flag`
	echo $flag
    }
    done	
    if [ $flag == $2 ]
    then
    	./kill_cluster_pid.sh 26 35 $5
	break
    fi
done 
rm -f temp*
echo "work done"
