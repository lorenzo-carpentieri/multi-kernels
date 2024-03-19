# rimouovere la cartella di build
# compilare con ASYNCH e HIDE ad ON
# eseguire 30 volte
# compilare con AYNCH ON e HIDE OFF
# eseguire 30 volte

# num of runs
NUM_RUNS=10
# create the path to build directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build

if [ ! -d "$SCRIPT_DIR/logs" ]; then
    # Create the directory
    mkdir -p "$SCRIPT_DIR/logs"
    echo "Directory created: $SCRIPT_DIR/logs"
else
    echo "Directory already exists: $SCRIPT_DIR/logs"
    rm -rf $SCRIPT_DIR/logs/*
fi

# asynch + hiding
# clean the build directory
rm -rf $BUILD_DIR/*
cd $BUILD_DIR
../script_mpi_freq/compile_asynch_hiding.sh 
make -j 


for ((i=0; i<$NUM_RUNS;i++));
do
    echo Run $i 
    mpirun -n 4 ./mpi_app_geom_vec_add >> ../logs/mpi_app_geom_vec_add_asynch_hiding.log
done
# asynch + no_hiding
rm -rf $BUILD_DIR/*
../script_mpi_freq/compile_asynch_no_hiding.sh
make -j 

for ((i=0; i<$NUM_RUNS;i++));
do
    echo Run $i 
    mpirun -n 4 ./mpi_app_geom_vec_add >> ../logs/mpi_app_geom_vec_add_asynch_no_hiding.log
done

# synch + hiding

rm -rf $BUILD_DIR/*
cd $BUILD_DIR
../script_mpi_freq/compile_synch_hiding.sh 
make -j 


for ((i=0; i<$NUM_RUNS;i++));
do
    echo Run $i 
    mpirun -n 4 ./mpi_app_geom_vec_add >> ../logs/mpi_app_geom_vec_add_synch_hiding.log
done

# synch + no_hiding

rm -rf $BUILD_DIR/*
../script_mpi_freq/compile_synch_no_hiding.sh
make -j 

for ((i=0; i<$NUM_RUNS;i++));
do
    echo Run $i 
    mpirun -n 4 ./mpi_app_geom_vec_add >> ../logs/mpi_app_geom_vec_add_synch_no_hiding.log
done




