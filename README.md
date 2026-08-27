### Some important points to note: 

1) There are two folders: heat_sink and cantilever.
2) There are 4 methods under test in the ADMM paper namely : ADMM-R&R, ADMM-MIP, Relax&Round (uses FENiCS backend), Kernel filtering(uses convex linearization)
3) The ADMM methods solve two subproblems (subproblem 1: PDE+penalty , subproblem2: TV+penalty)
4) Subproblem 1 is convexified using the ConLin approach, subproblem 2 is Quadractic constraint optimization.
5) Different backends can be configured for solving subproblem 2 in ADMM (see the file admm_config.cfg in respective admm_colin folders)
6) If you want to run one instance of each of the 4 methods, please run these files:

> python heat_sink/admm_colin/admm_run.py

> python heat_sink/oc_method/oc_r_sweep.py --r-start 2 --r-end 2 --num-r 1

> python heat_sink/relax_and_round_smooth/fenics_model.py

NOTE : The oc_r_sweep is meant to run a sweep over r values so please set r-start, r-end to same values and num-r to 1.
NOTE : To run the MIP version of ADMM set these two parameters in the config file:
> use_mip = false ;
> backend = gurobi

7) To run the perimeter sweep, please run the following files

> python heat_sink/admm_colin/admm_perimeter_adaptive_sweep.py

> python heat_sink/oc_method/oc_perimeter_adaptive_sweep.py

> python heat_sink/relax_and_round_smooth/fenics_perimeter_adaptive_sweep.py

8) After completion of each instance, the results are stored in respective directories,
files with names {mesh_dim}.h5 are created.

9) In each method, there is a notebook file to help visualize all the history and results.

### Please check the config files in admm folders for details about parameters, all the algorithm related parameters are in admm_config.cfg.

### Once the tests are run the complete run history files are saved in respective folders of alpha values:  0.1, 0.01, for oc method, it creates folders with r values.

### To visualize results please run the cells in notebook admm_viz.ipynb and admm_colin_viz.ipynb, a few examples:

*create admm object*

> admm = ADMM(alpha=0.01, dim=64, base_dir="run_data_admm_gurobi")

*all scalar valued objects are accessed via*
> admm.trial(0).series.objective

> admm.trial(0).series.infeasibility

*all vector valued objects are accessed via*
> admm.trial(0).iters.control

> admm.trial(0).iters.control_cont

*metadata accessed via*
> admm.trial(0).meta

*For deterministic solver there is only a single trial* 

*The notebook admm_oc_colin_comparison.ipynb has code to generate pareto plots in the paper*

:):
