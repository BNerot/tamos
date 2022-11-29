# tamos

## What does "tamos" stand for?
"tamos" stands for Thermal Architectures Modelling and Optimization Software.

## What is it?
`tamos` is a non-GUI tool that aims at facilitating the study of thermal architectures.
A thermal architecture is a set of energy production, storage and distribution components that operate together in order to meet some energy demands.

## How does it work? 
`tamos` performs the optimal sizing and operation of various energy components gathered in energy hubs.
It relies on the description of these components using the Mixed-Integer Linear Programming formalism (MILP), i.e. mathematical programming.

## What are the main software components?
`tamos` heavily relies on the `docplex` api, that eases the formulation of MILP problems.
Once declared, the problems are solved using the Cplex solver. It can also be exported to `.LP` and `.MPS` formats and be solved using non-proprietary solvers. 

##  Installation notes
The packaged version of `tamos`is available on PyPi. Please run:
`pip install tamos`


## Where is project hosted?
Sources are managed on GitHub: https://github.com/BNerot/tamos/tree/main/src/tamos 
The file `Batch analysis.ipynb` is only on GitHub. It provides the user with an easy way to analyse large volumes of results.

## Is it difficult to use?
Please follow one of the two examples in `examples` as a quick start guide. 
You can also find a web version of the documentation in `docs/build/html`. Once this directory is downloaded, please open 'index.html'. 

## Copyright
The code is distributed under an Apache-2.0 license. 
Most of the development work was done in the context of a PhD thesis. 
This thesis was funded by two French institutions:
- Commissariat à l'Energie Atomique: https://www.cea.fr/english
- Agence de la transition écologique: https://www.ademe.fr/en/frontpage/
