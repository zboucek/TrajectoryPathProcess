# Submit all configs for given worlds
num_worlds=7
num_configs=41
for WORLD in $( seq 0 $num_worlds )
do
   for CONFIG in $( seq 0 $num_configs )
   do
      echo "job for config" $WORLD $CONFIG
      qsub -v i_world=$WORLD,i_config=$CONFIG job.sh
   done
done
