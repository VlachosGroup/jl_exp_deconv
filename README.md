## jl_exp_deconv

### Organization of the  project

The project has the following structure:

    jl_exp_deconv/
      |- README.md
      |- jl_exp_deconv/
         |- __init__.py
         |- jl_exp_deconv.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...

jl_exp_deconv deconvolutes experimental mixture spectra via a model that is trained on experimental pure-component spectra.

### Module code

We place the module code in a file called `jl_exp_deconv.py` in directory called
`jl_exp_deconv`. ### Project Data

To get access to the data location run the following commands

    import os.path as op
    import jl_exp_deconv as exp_deconv
    data_path = op.join(exp_deconv.__path__[0], 'data')


### Testing

We use the ['pytest'](http://pytest.org/latest/) library for
testing. The `py.test` application traverses the directory tree in which it is
issued, looking for files with the names that match the pattern `test_*.py`
(typically, something like our `jl_exp_deconv/tests/test_jl_exp_deconv.py`). Within each
of these files, it looks for functions with names that match the pattern
`test_*`. Each function in the module would has a corresponding test
(e.g. `test_transform_data`). We use end-to end testing to  check that particular values in the code evaluate to
the same values over time. This is sometimes called 'regression testing'. We
have one such test in `jl_exp_deconv/tests/test_jl_exp_deconv.py` called
`test_params_regression`.

We use use the the `numpy.testing` module (which we
import as `npt`) to assert certain relations on arrays and floating point
numbers. This is because `npt` contains functions that are specialized for
handling `numpy` arrays, and they allow to specify the tolerance of the
comparison through the `decimal` key-word argument.

To run the tests on the command line, change your present working directory to
the top-level directory of the repository (e.g. `/Users/lansford/code/jl_exp_deconv`),
and type:

    py.test jl_exp_deconv

This will exercise all of the tests in your code directory.

### Installation

For installation and distribution we will use the python standard
library `distutils` module. This module uses a `setup.py` file to
figure out how to install your software on a particular system. For a
small project such as this one, managing installation of the software
modules and the data is rather simple.

A `jl_exp_deconv/version.py` contains all of the information needed for the
installation and for setting up the [PyPI
page](https://pypi.python.org/pypi/jl_exp_deconv) for the software. 
### Scripts
A scripts directory that provides examples is provided

### Git Configuration

Currently there are two files in the repository which help working
with this repository, and which you could extend further:

- `.gitignore` -- specifies intentionally untracked files (such as
  compiled `*.pyc` files), which should not typically be committed to
  git (see `man gitignore`)
- `.mailmap` -- if any of the contributors used multiple names/email
  addresses or his git commit identity is just an alias, you could
  specify the ultimate name/email(s) for each contributor, so such
  commands as `git shortlog -sn` could take them into account (see
  `git shortlog --help`)
