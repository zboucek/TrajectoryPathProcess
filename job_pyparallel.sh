#!/bin/bash
#PBS -N trajectory_planning_jobs
##PBS -l select=2:ncpus=1 -l place=exclhost # request whole host allocated to this job (without cpu and mem limit control)
##PBS -l select=84:place=group=cluster # 84 cpus on one cluster
#PBS -l select=1:ncpus=32:mem=100gb:scratch_local=6gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m e

export OMP_NUM_THREADS=$PBS_NUM_PPN # set it equal to PBS variable PBS_NUM_PPN (number of CPUs in a chunk)

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
DATADIR="/storage/plzen1/home/$LOGNAME/disertace/pyparallel-first/traj_plan"

# # Load the config file
# config_file=$(ls configs/config_${world}_${config}.json)
# config_json=$(cat ${config_file})

# worlds=['simple', 'simple2', 'orchard', 'columns', 'random_spheres',
#         'forest', 'random_columns', 'walls']
WORLD='0'

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# checks if scratch directory is created
if [ ! -d "$SCRATCHDIR" ] ; then echo "Scratch directory is not created!" 1>&2; exit 1; fi

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
cp -r $DATADIR/. $SCRATCHDIR

cd $SCRATCHDIR

if [ ! -d "$SCRATCHDIR/data_traj" ] ; then echo "data_traj is not in Scratch directory!" 1>&2; exit 1; fi

# spusteni aplikace - samotny vypocet
# singularity exec --bind $SCRATCHDIR:/data $SCRATCHDIR/traj_container2.sif python3 /data/run_python_jobs.py
singularity exec --bind $SCRATCHDIR:/data traj_container2.sif python3 /data/run_python_jobs_parallel.py --world_name $WORLD

# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
# TODO pro ostry beh dat CLEAN_SCRATCH=false
cd $SCRATCHDIR
if [ ! -d "$SCRATCHDIR/data_traj" ] ; then echo "data_traj is not in Scratch directory!" 1>&2; exit 1; fi
if [ ! -d "$DATADIR/data_traj" ] ; then echo "data_traj is not in Scratch directory!" 1>&2; exit 1; fi
cp -r $SCRATCHDIR/data_traj/. $DATADIR/data_traj || export CLEAN_SCRATCH=false