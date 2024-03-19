echo ASYNCH + HIDING 
python3 ./parse_mpi_out.py ./logs/mpi_app_geom_vec_add_asynch_hiding.log
echo ASYNCH + NO_HIDING
python3 ./parse_mpi_out.py ./logs/mpi_app_geom_vec_add_asynch_no_hiding.log
echo SYNCH + HIDING
python3 ./parse_mpi_out.py ./logs/mpi_app_geom_vec_add_synch_hiding.log
echo SYNCH + NO_HIDING
python3 ./parse_mpi_out.py ./logs/mpi_app_geom_vec_add_synch_no_hiding.log