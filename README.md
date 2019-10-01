## jl_exp_deconv
[![Build Status](https://travis-ci.org/uwescience/jl_exp_deconv.svg?branch=master)](https://travis-ci.org/uwescience/jl_exp_deconv)

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

We have also provided a `Makefile` that allows you to run the tests with more
verbose and informative output from the top-level directory, by issuing the
following from the command line:

    make test

### Styling

Some projects include `flake8` inside their automated tests, so that every pull
request is examined for code cleanliness.

In this project, we have run `flake8` most (but not all) files, on
most (but not all) checks:

```
flake8 --ignore N802,N806 `find . -name *.py | grep -v setup.py | grep -v /doc/`
```

This means, check all .py files, but exclude setup.py and everything in
directories named "doc". Do all checks except N802 and N806, which enforce
lowercase-only names for variables and functions.

The `Makefile` contains an instruction for running this command as well:

    make flake8

### Documentation

We follow the [numpy docstring
standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt),
which specifies in detail the inputs/outputs of every function, and specifies
how to document additional details, such as references to scientific articles,
notes about the mathematics behind the implementation, etc.

To document `jl_exp_deconv` we use the [sphinx documentation
system](http://sphinx-doc.org/). You can follow the instructions on the sphinx
website, and the example [here](http://matplotlib.org/sampledoc/) to set up the
system, but we have also already initialized and commited a skeleton
documentation system in the `docs` directory, that you can build upon.

Sphinx uses a `Makefile` to build different outputs of your documentation. For
example, if you want to generate the HTML rendering of the documentation (web
pages that you can upload to a website to explain the software), you will type:

	make html

This will generate a set of static webpages in the `doc/_build/html`, which you
can then upload to a website of your choice.

Alternatively, [readthedocs.org](https://readthedocs.org) (careful,
*not* readthedocs.**com**) is a service that will run sphinx for you,
and upload the documentation to their website. To use this service,
you will need to register with RTD. After you have done that, you will
need to "import your project" from your github account, through the
RTD web interface. To make things run smoothly, you also will need to
go to the "admin" panel of the project on RTD, and navigate into the
"advanced settings" so that you can tell it that your Python
configuration file is in `doc/conf.py`:

![RTD conf](https://github.com/uwescience/jl_exp_deconv/blob/master/doc/_static/RTD-advanced-conf.png)

 http://jl_exp_deconv.readthedocs.org/en/latest/


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
