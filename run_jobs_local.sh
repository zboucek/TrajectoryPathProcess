# Submit all configs for given world 
world="simple"

num_worlds=0
num_configs=2
for WORLD in $( seq 0 $num_worlds )
do
   for CONFIG in $( seq 0 $num_configs )
   do
      echo "job for config" $WORLD $CONFIG
      sh job.sh $WORLD $CONFIG
   done
done
