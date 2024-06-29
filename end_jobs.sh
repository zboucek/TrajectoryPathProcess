# Submit all configs for given worlds
start_i=19620546
end_i=19620819
for i in $( seq $start_i $end_i )
do
   echo "end job for config"$i
   qdel $i".meta-pbs.metacentrum.cz"
done
