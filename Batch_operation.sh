#!/bin/bash
#$1 id_start eg:ssd35's id is 35
#$2 id_end
#$3 cmd
for(( i=$1; i > $2; i-- ))
    do
        ssh ssd$i $3
    done
    

