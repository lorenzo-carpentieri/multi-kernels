Specify the following CMAKE variable:
- WITH_MPI_ASYNCH= ON/OFF : To enable MPI_Ixxx call. We handle the frequency change in two different ways:
    1. freq. change inside a thread, followed by a synch MPI_xxx function (WITH_MPI_ASYNCH=OFF and WITH_THREAD_FREQ_CHANGE=ON)
    2. freq. change done by another process (WITH_MPI_ASYNCH=OFF and WITH_PROCESS_FREQ_CHANGE=ON)
    3. freq. change done by another process (WITH_MPI_ASYNCH=ON and WITH_PROCESS_FREQ_CHANGE=ON) (Maybe not necessary in this case)
NOTE: we do not need another thread as the sycl kernel that we launch for changin the frequency is asynchronous
- ENABLE_FREQ_CHANGE_MPI_HIDING= ON/OFF : To hide the frequency change during MPI function calls. When is off the frequency change is done befeore kernel execution otherwise is done during MPI function call with the approaches describen above.
