from functools import wraps
from pathlib import Path
from pandas import DataFrame
from tamos.data_IO import ResultsExport


class AdvSolve:
    _all_kinds = ["Eco", "Exergy", "CO2"]
    def __init__(self,
                 MILPModel,
                 working_dir,
                 skip_existing=True,
                 **kwargs
                 ):
        """
        AdvSolve is a wrapper that ease the advanced solving of MILP models.

        Attribute `last_binaries` contains the ResultsExport instances of of the latest solved models.
        These all are results of `MILPModel` yet following different solving configurations.

        Parameters
        ----------
        MILPModel : MILPModel
            A `MILPModel` instance whose `declare_variables` and `declare_constraints_and_KPIs`
            methods have already been called.
            `MILPModel` is used in every call to this instance methods.
        working_dir : str or path-like
            Directory the results are read from (see `skip_existing`) and written to.
        skip_existing : bool, optional, True

            * If True and if there exists a ResultsExport on disk having the same name than `MILPModel`,
              existing results are loaded instead of solving again `MILPModel`.
            * If False, `MILPModel` is solved in all cases.

        kwargs : optional
            Keywords arguments.
            Some are passed to the `solve` method of `MILPModel` instance. These are:

                * MIP_gap: float, default 1e-4
                * threads: int, default 0
                * timelimit: int, default 43200

            Others are passed to `ResultsExport`. These are:

                * get_LP: bool, default False
                * get_MPS: bool, default False
                * csv_precision: int, default 3
                * replace_inverted_TVP: bool, default False

            `parent_working_dir` of ResultsExport will be ignored since `working_dir` has the same use.

        Notes
        -----
        1. Each MILP solving is performed using the Cplex optimizer.
        2. AdvSolve does not perform model declaration.

        """
        self._working_dir = Path(working_dir)
        self._working_dir.mkdir(parents=True, exist_ok=True)
        self._skip_existing = skip_existing
        self._MILPModel = MILPModel
        self._base_name = self._MILPModel.name
        self._last_binaries = {}
        self._kwargs = kwargs

    def _clean_last_binaries(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            self._last_binaries = {}
            res = func(self, *args, **kwargs)
            return res
        return new_func

    @property
    def _ResultsExport_kwargs(self):
        return {k: v for k, v in self.kwargs.items()
                if k in ["get_LP", "get_MPS", "csv_precision", "replace_inverted_TVP"]}

    @property
    def _solve_kwargs(self):
        return {k: v for k, v in self.kwargs.items()
                if k in ["MIP_gap", "threads", "timelimit"]}

    @property
    def skip_existing(self):
        """
        * If True and if there exists a ResultsExport on disk having the same name than `MILPModel`,
          existing results are loaded instead of solving again `MILPModel`.
        * If False, `MILPModel` is solved in all cases.

        """
        return self._skip_existing

    @property
    def working_dir(self):
        """
        Directory the results are read from (see `skip_existing`) and written to.

        """
        return self._working_dir

    @property
    def MILPModel(self):
        """
        Used in every call to this instance methods.

        """
        return self._MILPModel

    @property
    def kwargs(self):
        """
        Keywords arguments.
        Some are passed to the `solve` method of `MILPModel` instance. These are:

            * MIP_gap: float, default 1e-4
            * threads: int, default 0
            * timelimit: int, default 43200

        Others are passed to `ResultsExport`. These are:

            * get_LP: bool, default False
            * get_MPS: bool, default False
            * csv_precision: int, default 3
            * replace_inverted_TVP: bool, default False

        `parent_working_dir` of ResultsExport will be ignored since `working_dir` has the same use.

        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, **kwargs):
        self._kwargs = kwargs

    @property
    def last_binaries(self):
        """
        The ResultsExport instances of the latest solved models.
        dict {str: ResultsExport}

        """

        return self._last_binaries

    def _load_existing_results(self):
        if self._skip_existing:
            try:
                name = self._MILPModel.name
                path = self._working_dir / Path(rf"{name}/results {name}")
                pp = ResultsExport.load_object(path)
                print("Found existing file: ", pp)
                return pp
            except FileNotFoundError as e:
                print(e)

    @classmethod
    def _write_results(self, results):
        results.write_all()
        results.dump_object()

    def _solve(self, kind, write=True):
        pp = self._load_existing_results()
        exists = pp is not None
        if not exists:
            self._MILPModel.solve(kind=kind, **self._solve_kwargs)
            pp = ResultsExport(self._MILPModel, parent_working_dir=self.working_dir, **self._ResultsExport_kwargs)
            if write:
                self._write_results(pp)
        self._last_binaries[pp._name] = pp
        return pp, exists

    def _clean_metadata(self, ct_kind=None):
        # does not remove solutions pool within Cplex
        for kind in self._all_kinds:
            self._MILPModel.description[f"{kind} quota (%)"] = "N.D"
        self._MILPModel.description["Effective kind"] = "N.D"
        self._MILPModel.name = self._base_name
        if ct_kind:
            self._MILPModel.remove_max_KPI_constraint(ct_kind)

    @_clean_last_binaries
    def solve_front(self,
                    main_kind,
                    aux_kind,
                    mask_weighted_average,
                    ):
        """
        Performs several `MILPModel` solvings given epsilon constraints to obtain a Pareto front of solutions.

        Parameters
        ----------
        main_kind : {'Eco', 'CO2', 'Exergy'}
            The kind of objective function to minimize.
        aux_kind : {'Eco', 'CO2', 'Exergy'}
            The kind of objective function that is constrained by epsilon constraints.
        mask_weighted_average : list of int, each int `x` verifies 0 < x < 10 000
            In per 1e4.
            Values defining the epsilon constraints.
            Describes the upper bound of `aux_kind` as a fraction of the difference 'A - B', where:

            * 'A' is the value of `aux_kind` KPI in `main_kind` optimization
            * 'B' is the value of `aux_kind` KPI in `aux_kind` optimization

            See the examples section.

        Returns
        -------
        df: pandas.DataFrame
            A table summing-up the values of KPIs `main_kind` and `aux_kind` (columns)
            for the optimization according to `main_kind` and `aux_kind` objective functions (rows).

        Notes
        -----
        This method proceeds according to these steps:

        1. Optimize the problem according to `main_kind` objective function.
           Save the values of `aux_kind` KPI, say 'A'.
        2. Optimize the problem according to `aux_kind` objective function.
           Save the values of `aux_kind` KPI, say 'B'.
        3. For each value `x` of `mask_weighted_average`, do:

           * Define an epsilon constraint on KPI `aux_kind` (using the `declare_max_KPI_constraint` method of `MILPModel`).
             The value of the constraint is: B + x * (A - B)
           * Optimize the problem according to `main_kind` objective function, given the previously defined epsilon constraint.


        Examples
        --------
        >>> adv_solve.solve_front(main_kind="Eco", aux_kind="CO2", mask_weighted_average=[1000, 2000, 5000, 9000])
            The problem is solved according to "Eco" and "CO2" objective functions.
            Then it is solved according to "Eco" objective function allowing successively:

                * 10% of CO2 (1000)
                * 20% of CO2 (2000)
                * 50% of CO2 (5000)
                * 90% of CO2 (9000)

        """
        assert main_kind in self._all_kinds, f"'main_kind' must be one of {self._all_kinds}"
        assert aux_kind in self._all_kinds, f"'aux_kind' must be one of {self._all_kinds}"
        assert aux_kind != main_kind, f"'aux_kind' must be different from 'main_kind'"
        # outer key is KPI, inner key is objective type
        KPI_values = {
            main_kind: {main_kind: None, aux_kind: None},
            aux_kind: {main_kind: None, aux_kind: None},
        }
        self._clean_metadata(aux_kind)
        for obj_kind in [main_kind, aux_kind]:
            self._MILPModel.description["Effective kind"] = obj_kind
            self._MILPModel.name = f"{self._base_name}_{obj_kind}"
            pp, exists = self._solve(kind=obj_kind)
            KPI_values[aux_kind][obj_kind], KPI_values[main_kind][obj_kind] = pp.solution_summary[aux_kind], \
                                                                              pp.solution_summary[main_kind]

        max_aux_kind_values = []
        quota_values = []
        name_suffixes = []
        best = KPI_values[aux_kind][aux_kind]
        worst = KPI_values[aux_kind][main_kind]
        for value in mask_weighted_average:
            if 0 < value < 10000:
                max_aux_kind_values.append(best + (value / 10000) * (worst - best))
                quota_values.append((value / 10000) * 100)
                name_suffixes.append(f"wa{value}")
            else:
                print(f"{self}: value {value} of '{mask_weighted_average}' does not verifies 0 < value < 10 000\n"
                      f"thus it is ignored.")


        for max_aux_kind, quota, name_suffix in list(
                zip(max_aux_kind_values, quota_values, name_suffixes)):
            self._MILPModel.description["Effective kind"] = "N.D"
            self._MILPModel.name = f"{self._base_name}_{main_kind}_{aux_kind}_{name_suffix}"
            self._MILPModel.declare_max_KPI_constraint(aux_kind,
                                                       max_aux_kind)  # removing old constraint is covered by this call
            self._MILPModel.description[f"{aux_kind} quota (%)"] = quota
            pp, exists = self._solve(kind=main_kind)
        self._clean_metadata(aux_kind)

        df = DataFrame(KPI_values)
        df.columns.name = "KPI"
        df.index.name = "Objective kind"
        return df

    @_clean_last_binaries
    def solve(self,
              kind,
              exact_cost=False):
        """
        Solves `MILPModel` with added features regarding "Exergy" and "CO2" objective functions.

        Parameters
        ----------
        kind : {'Eco', 'CO2', 'Exergy'}
            The kind of objective function to minimize.
            If 'Eco', this method does nothing much than calling `MILPModel.solve` method.
        exact_cost : bool optional, default False
            Only for `kind`='CO2' or `kind`='Exergy'.

            * If True, a second optimization is performed following an 'Eco' objective function but with an
              epsilon constraint so that the minimum amount of KPI `kind` is allowed.
            * If False, no second optimization is performed.

            The main consequence of setting `exact_cost` = True is to get the real system cost with yet a minimization of
            non-economic KPI 'Exergy' or 'CO2'.


        Notes
        -----
        Setting `exact_cost`=True may result in:

        * an infeasible problem due to a lack of numerical tolerance
        * a problem having a different solution from the first optimization, though they are theoretically supposed
          to be identical.

        """
        self._clean_metadata()
        if kind in ["CO2", "Exergy"]:
            self._MILPModel.name = f"{self._base_name}_{kind}"
            self._MILPModel.description["Effective kind"] = kind
            pp1, exists1 = self._solve(kind, write=False)
            self._MILPModel.declare_max_KPI_constraint(kind, pp1.solution_summary[kind])
            self._MILPModel.name = f"{self._base_name}_{kind}_EC"
            pp2, exists2 = self._solve("Eco", write=False)
            if self._MILPModel._last_solve_ok \
                    and (not exists2):
                self._write_results(pp2)
            else:
                print(f"{self}: Exact costs optimization failed.\n"
                      f"Try again manually by setting numerical tolerance on max {kind} constraint.")
                if not exists1:
                    self._write_results(pp1)
            self._clean_metadata(kind)
        else:
            assert kind == "Eco", f"{self}: 'kind' must be one of {self._all_kinds}, got '{kind}'."
            if exact_cost:
                print(f"{self}: 'exact_cost' is not used with Eco optim.")
            self._MILPModel.description["Effective kind"] = kind
            self._MILPModel.name = f"{self._base_name}_{kind}"
            self._solve(kind=kind)
            self._clean_metadata()
