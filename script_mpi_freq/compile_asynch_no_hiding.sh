export SYSMAN=
cmake \
    -DCMAKE_CXX_COMPILER="/opt/intel/oneapi/compiler/2024.0/bin/icpx" \
    -DCMAKE_CXX_FLAGS="-O3 -Wno-deprecated" \
    -DDPCPP_WITH_LZ_BACKEND=ON \
    -DLZ_ARCH="pvc" \
    -DENABLED_SYNERGY=ON \
    -DSYNERGY_LZ_SUPPORT=ON \
    -DENABLED_TIME_EVENT_PROFILING=ON \
    -DSYNERGY_DEVICE_PROFILING=ON \
    -DSYNERGY_USE_PROFILING_ENERGY=ON \
    -DWITH_MPI_ASYNCH=ON \
    -DENABLE_FREQ_CHANGE_MPI_HIDING=OFF \
    ..
# -DMPI_C=/leonardo/prod/spack/05/install/0.21/linux-rhel8-icelake/gcc-8.5.0/intel-oneapi-mpi-2021.10.0-6ff7efudnkj4xa7o4j67mxatygwzfwqc/mpi/2021.10.0/bin/mpicc \
# -DMPI_CXX=/leonardo/prod/spack/05/install/0.21/linux-rhel8-icelake/gcc-8.5.0/intel-oneapi-mpi-2021.10.0-6ff7efudnkj4xa7o4j67mxatygwzfwqc/mpi/2021.10.0/bin/mpicxx \

