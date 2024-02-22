## File Description

Each `.conf` file has the name of each kernel and the approach to use in order to perform frequency scaling.

```
APP|PHASE|KERNEL
kernel_name1 <freq1> <KEEP|NO_KEEP>
kernel_name2 <freq2> <KEEP|NO_KEEP>
...
kernel_nameN <freqN> <KEEP|NO_KEEP>
```
The `KEEP` option is used with **PHASE** approach and means that at the next execution of that kernel the frequency will be changed again. In the **APP** and **KERNEL** approach is not going to be used.

The `NO_KEEP` option is the default one for **PHASE** frequency scaling, and consist of avoid setting the frequency at the next execution of that kernel. In the **APP** and **KERNEL** approach is not going to be used.