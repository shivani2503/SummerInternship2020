#!/bin/bash
#SBATCH --job-name=2xjobs
#SBATCH --account=NTDD0004
#SBATCH --ntasks=36
#SBATCH --ntasks-per-node=36
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --constraint=skylake


export SCRIPT=./bin/ncar_samurai_petsc_run_casper.sh
export COMPILER=pgi
export CONFIG=36th
export CODE=bnlsv4

$SCRIPT ~/beltrami.xml >& log.$COMPILER.$CONFIG.beltrami.full.$CODE
mv timing.0 timing.$COMPILER.$CONFIG.beltrami.full.$CODE

#./bin/ncar_samurai_run.sh ~/supercell12x.xml >& log.$COMPILER.$CONFIG.supercell12x.new3
#mv timing.0 timing.$COMPILER.$CONFIG.supercell12x.new3
#$SCRIPT ~/supercell.xml >& log.$COMPILER.$CONFIG.supercell.full.$CODE
#mv timing.0 timing.$COMPILER.$CONFIG.supercell.full.$CODE
#$SCRIPT ~/supercell2x.xml >& log.$COMPILER.$CONFIG.supercell2x.full.$CODE
#mv timing.0 timing.$COMPILER.$CONFIG.supercell2x.full.$CODE

#./bin/ncar_samurai_run.sh ~/hurricane12x.xml >& log.$COMPILER.$CONFIG.hurricane12x.full.$CODE
#mv timing.0 timing.$COMPILER.$CONFIG.hurricane12x.full.$CODE
#$SCRIPT ~/hurricane.xml >& log.$COMPILER.$CONFIG.hurricane.full.$CODE
#mv timing.0 timing.$COMPILER.$CONFIG.hurricane.full.$CODE
#$SCRIPT ~/hurricane2x.xml >& log.$COMPILER.$CONFIG.hurricane2x.full.$CODE
#mv timing.0 timing.$COMPILER.$CONFIG.hurricane2x.full.$CODE
$SCRIPT ~/hurricane_4panel-modD.xml >& log.$COMPILER.$CONFIG.hurricane_4panel-modD.full.$CODE
mv timing.0 timing.$COMPILER.$CONFIG.hurricane_4panel-modD.full.$CODE


