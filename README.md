## jl_exp_deconv

jl_exp_deconv deconvolutes experimental mixture spectra via a model that is trained on experimental pure-component spectra.

### Module code

We place the module code in a file called `jl_exp_deconv.py` in directory called
`jl_exp_deconv`. ### Project Data

To get access to the data location run the following commands

    import os.path as op
    import jl_exp_deconv as exp_deconv
    data_path = op.join(exp_deconv.__path__[0], 'data')

### Windows executable

A Windows executable .bat file that provides a user interface via Command Prompt along with instructions is provided in the scripts directory.
