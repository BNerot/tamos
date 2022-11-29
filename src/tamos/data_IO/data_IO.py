# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:25:06 2020
"""
from functools import reduce, wraps
from itertools import product, chain
from os import listdir
from pathlib import Path
from pickle import dump, load

from numpy import ndarray, nan
from pandas import DataFrame, Series, read_csv, concat, to_numeric
from tqdm import tqdm


class NamesFormatter:
    naming = False
    _separator_left = "("
    _separator_right = ")"
    _separator_both = ")("

    @staticmethod
    def use_name_in_MILP(bool):
        """
        Defines whether the name of components is used to give an exhaustive human-friendly name
        to decision variables, constraints and KPIs in the MILP problem.

        This has no effect on the exported variables and KPIs which will always be named according to the component name.
        Consider calling `use_name_in_MILP(True)` only for debugging and exporting LP or MPS files of the model with a
        human-friendly content.

        Parameters
        ----------
        bool : bool

            * If False (default), variables, constraints and KPIs will receive non human-friendly names.
            * If True, components names are used.
              This has two consequences:

                  * It slightlys lengthen the declaration time of the MILP problem.
                  * It makes LP and MPS files more readable but with a bigger size on disk.

        """
        if bool not in [True, False]:
            raise ValueError(f"'use_name_in_MILP': 'bool' must be a boolean, got '{bool}'.")
        NamesFormatter.naming = bool

    @staticmethod
    def _name_assembly(base_name, args):
        return f"{base_name}" \
               f"{NamesFormatter._separator_left}" \
               f"{f'{NamesFormatter._separator_both}'.join(map(str, args))}" \
               f"{NamesFormatter._separator_right}"

    @staticmethod
    def fmt(base_name, *args, KPI=False):
        if KPI or NamesFormatter.naming:
            return (NamesFormatter._name_assembly(base_name, obj) for obj in product(*args))

    @staticmethod
    def fmt_light(base_name, *args, KPI=False):
        # Does the same than names_formatter for the particular case where all args are not lists of str but str.
        if KPI or NamesFormatter.naming:
            return NamesFormatter._name_assembly(base_name, args)

    @staticmethod
    def decompose(name):
        parts = name.split(NamesFormatter._separator_both)
        parts[-1] = parts[-1].split(NamesFormatter._separator_right)[0]
        return parts[0].split("(") + parts[1:]


class FileReader:
    @staticmethod
    def read_properties(path):
        """
        Read a CSV file describing techno-economic properties of a component.

        Parameters
        ----------
        path : str or path-like object
            The path of the file to load properties from.

        Returns
        -------
        properties: dict
            A dict of the content of the file located at `path`.

        Notes
        -----
        1. The file must have two columns:

           * The first column contains properties name
           * The second column contains properties value

        2. The file must not have a header row (i.e. title row).
        3. Lines starting with '#' are treated as comments.

        """
        return read_csv(path, index_col=0, comment="#", header=None).squeeze("columns").to_dict()

    @staticmethod
    def read(path, col_name, start=None, end=None):
        """
        Read a CSV file describing a serie of values

        Parameters
        ----------
        path : str or path-like object
            The path of the file to load values from.
        col_name: str, optional
            The name of the column to be loaded from the CSV file.
        start : int, optional
            First row of values being loaded.
            If not provided, start=0.
        end : int, optional
            Last row of values being loaded.
            If not provided, end is the last row of the file.

        Returns
        -------
        A numpy.ndarray of the content located at `path` under `col_name`.

        Notes
        -----
        1. The file must not have a header row (i.e. title row).
        2. Lines starting with '#' are treated as comments.

        """
        df = read_csv(path, comment="#", usecols=[col_name]).iloc[start:end]
        return df[col_name].values


class DataAccessors:
    @staticmethod
    def type_checker(arg, caller, attr_name, kind):
        if kind == "numeric":
            types = (int, float, ndarray)
        else:
            types = kind                    # explicitly specified
        if not isinstance(arg, types):
            raise TypeError(f"{caller}: Invalid data type for '{attr_name}'.")
        else:
            try:            # if arg is a ThermalVector
                cond = not bool(arg._used_in_TVP) # if arg._used_in_TVP>0, cond = False
                if not cond:
                    raise TypeError(f"{caller}: Using ThermalVector {arg} for '{attr_name}' is impossible because " 
                                    f"{arg} is already used in a ThermalVectorPair.")
            except AttributeError as e:
                pass


    @staticmethod
    def get1(arg, t):
        if isinstance(arg, ndarray):
            return arg[t]
        else:
            return arg

    @staticmethod
    def get2(arg, t1, t2):
        if isinstance(arg, ndarray):
            if t2 - t1 == 1:
                return arg[t1]
            else:
                return arg[t1:t2].mean()
        else:
            return arg

class ResultsIO:
    @staticmethod
    def load_object(path):
        """
        Loads a binary ResultsExport or ResultsBatch object.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        The loaded object.

        """
        with open(path, "rb") as file:
            export = load(file)
        export._is_loaded = True
        return export

    @staticmethod
    def remove_small_values(df, name, threshold=0.1, drop_zeros=None):
        """
        Removes small numerical values from a dataframe containing decision variable or KPI values.

        Parameters
        ----------
        df : pandas.DataFrame
        name : str
            Name of the column of `df` that contains the numerical values.
        threshold : int or float, optional, default 0.1
            All values 'x' for which abs(x) < `threshold` are replaced by 0.
        drop_zeros : {'Series', 'All'}, optional

            * If not provided, all rows where `name` value is zero are kept.
            * 'Series': each temporal data series is kept if at least one value in this serie is not zero.
              `df` must include a 'Date' column for `drop_zeros`='Series' to be effective.
              A temporal data serie of `df` is a subset of df such that only the 'Date' and `name` columns
              have changing values in this subset.
            * 'All' : remove all rows where `name` value is zero.

        Notes
        -----
        This method works inplace, i.e. `df` is modified.
        Near-zero values in column `name` are removed according to `threshold` and `drop_zeros`.

        """
        assert drop_zeros in [None, "Series", "All"]
        df[name] = df[name].where(abs(df[name]) >= threshold, 0)
        if drop_zeros is None:
            pass
        elif (drop_zeros == "Series") and ("Date" in df.columns):
            # keep all values if at least one is not zero
            index = df.columns.difference([name, "Date"]).to_list()
            cond = df.groupby(index)[name].transform(lambda x: any(x!=0))
            df.drop(index=df.index[~cond], inplace=True)
        else:
            cond = (df[name] != 0)
            df.drop(index=df.index[~cond], inplace=True)

    @staticmethod
    def reduce_memory(df):
        """
        Reduces the memory used by a DataFrame by changing the data type of numerical columns.

        Data type of float and integer columns is changed for the type with the smallest memory usage.

        Parameters
        ----------
        df : DataFrame

        Notes
        -----
        This method works inplace, i.e. `df` is modified.

        """
        for col in df.columns:
            if df[col].dtype in ["float32", "float64"]:
                df.loc[:, col] = to_numeric(df.loc[:, col], downcast="float")
            if df[col].dtype in ["int32", "int64"]:
                df.loc[:, col] = to_numeric(df.loc[:, col], downcast="integer")

    @property
    def working_dir(self):
        """
        Directory the results are read from or exported to.

        """
        self._working_dir.mkdir(parents=True, exist_ok=True)  # dir must be created only when content is written on disk
        return self._working_dir

class ResultsExport(ResultsIO):

    def _dump_status_checker(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            if self._is_loaded:
                raise AttributeError("Impossible: data was deleted for disk dump.")
            else:
                res = func(self, *args, **kwargs)
            return res
        return new_func

    def __init__(self, MILPModel, get_LP=False, get_MPS=False,
                 parent_working_dir=None, csv_precision=3,
                 replace_inverted_TVP=False     # only for "Element" columns in vars the use it, i.e. "F_P" and "F_S"
                 ):
        """
        ResultsExport instances extract and write to disk the decision variables and KPIs of a solved MILPModel instance.

        Variable and KPIs values are collected at the instance creation. These are stored as attributes.
        A call to a `write_` method copies this information on disk as CSV and text files.
        A call to `dump_object` method write a binary form of this instance on disk.


        Parameters
        ----------
        MILPModel : MILPModel
            If the solving of the MILPModel instance did not succeed, only pre solve model information is collected by ResultsExport.
            If the solving succeeded, KPIs, decision variables and some non-essential model results are collected.
        get_LP : bool, optional, default, False
            Whether to collect the model following the LP standard.
        get_MPS : bool, optional, default False
            Whether to collect the model following the MPS standard.
        parent_working_dir : str or path-like, optional
            This defines the `working_dir` attribute according to working_dir=parent_working_dir/name,
            where 'name' is the `name` attribute of MILPModel.
            The export of results is done in `working_dir`.
            If not provided, `parent_working_dir` is a new directory called 'results' in the parent working directory.
        csv_precision : int, optional, default 3
            Number of decimals regarding CSV export of decision variables and KPIs.
        replace_inverted_TVP : bool, optional, False

            * If True, ThermalVectorPair elements '~TVP' are replaced by 'TVP' if both of them are used in the model.
              This does not change the values of decision variables and KPIs since
              a flow associated with 'TVP' is the opposite of the same flow associated with '~TVP'.
              Setting `replace_inverted_TVP`=True can make the results reading easier.
            * If False, both '~TVP' and 'TVP' are kept.

        Notes
        -----
        A ResultsExport instance is a representation of a MILPModel instance at the time it is created.
        The further change in the MILPModel instance are not reflected in the already existing ResultsExport instance.

        """
        DataAccessors.type_checker(get_LP, "ResultsExport", "get_LP", bool)
        DataAccessors.type_checker(get_MPS, "ResultsExport", "get_MPS", bool)
        DataAccessors.type_checker(csv_precision, "ResultsExport", "csv_precision", int)
        DataAccessors.type_checker(replace_inverted_TVP, "ResultsExport", "replace_inverted_TVP", bool)
        self._is_loaded = False
        self._MILPModel = MILPModel
        self._name = self._MILPModel._name
        self._var_dataframes = {}
        self._KPI_dataframes = {}
        if parent_working_dir is None:
            parent_working_dir = Path(r"../results")
        self._working_dir = Path(parent_working_dir, self._name).resolve()
        self._replace_inverted_TVP = replace_inverted_TVP
        self._csv_precision = csv_precision
        
        if get_LP:
            self._get_LP()
        if get_MPS:
            self._get_MPS()

        self._get_model_complexity()
        self._get_description()
        if self._MILPModel._last_solve_ok:
            self._get_solution_summary()
            self._get_variables()
            self._get_KPIs()
            self._get_unsatisfied_constraints()
            self._get_exec_time()
        else:
            print("Model is unsolved or model properties cannot be collected.")

    

    
    def _convert_var_to_dataframe(self, var_name):
        optim_var = self._MILPModel._model_data.vars[var_name]
        def _replace_inverted_TVP__(df):
            inverted_TVPs = {}
            for element in df.Element.unique():
                try:
                    other = element._invert_matcher
                    if other != element:
                        inverted_TVPs[element] = other
                except:     # AttributeError
                    pass

            if inverted_TVPs:
                df = df.pivot(values=var_name, columns="Element",
                              index=list(df.columns.difference([var_name, "Element"])))
                for col, other in inverted_TVPs.items():
                    df[col] = - df[col]
                    df = df.rename(columns={col: other})
                df = df.melt(value_vars=df.columns, value_name=var_name, ignore_index=False)
                df = df.dropna().reset_index()
            return df
        def _convert_var_to_dataframe__():
            var_map = self._MILPModel._model_data.var_mapper[var_name]
            columns = []
            for name in var_map:
                if name in columns:
                    name += "_bis"
                columns.append(name)
            columns.append(var_name)
            df = self._MILPModel._model_data.mdl.solution.get_value_df(optim_var)
            df.columns = columns
            if df[var_name].any():
                # if timestep is not fully regular, then dataframes must be resampled for an easy reading of data
                if (len(self._MILPModel._TS._t) - 1 != self._MILPModel._TS._n) and ("Date" in df.columns):
                    df = df.pivot(values=var_name, index=["Date"], columns=list(df.columns.difference([var_name, "Date"]))).sort_index()
                    df = df.reindex(range(df.index.min(), df.index.max()+1)).ffill()
                    df = df.unstack().reset_index()
                    df.columns = list(df.columns[:-2]) + ["Date", var_name]
                if self._replace_inverted_TVP and "Element" in df.columns:
                    df = _replace_inverted_TVP__(df)
            else:
                df = DataFrame(columns=columns)
            # code below is needed to work on dataframes on a name basis rather than an instance basis.
            # prefered because KPIs
            for col in df.columns.difference(["Date", var_name]):
                df[col] = df[col].apply(repr)
            return df
        if var_name not in self._var_dataframes.keys():
            df = _convert_var_to_dataframe__()
            self._var_dataframes[var_name] = df
        return self._var_dataframes[var_name]

    def _convert_KPI_to_dataframe(self, KPI_name):
        if KPI_name not in self._KPI_dataframes.keys():
            data = []
            KPI_map = self._MILPModel._model_data.KPI_mapper[KPI_name]
            columns = []
            for name in KPI_map:
                if name in columns:
                    name += "_bis"
                columns.append(name)
            columns.append(KPI_name)
            for name, value in self._MILPModel._model_data.mdl.kpis_as_dict().items():
                base_name, *keys = NamesFormatter.decompose(name)
                if base_name == KPI_name:
                    data.append(keys + [value])

            self._KPI_dataframes[KPI_name] = DataFrame(data=data, columns=columns)
        return self._KPI_dataframes[KPI_name]

    @_dump_status_checker
    def _get_exec_time(self):
        self._exec_time = self._MILPModel._exec_time.copy()
        self._exec_time["Total time"] = self._MILPModel._exec_time[["MILPModel.declare_variables",
                                                                    "MILPModel.declare_constraints_and_KPIs",
                                                                    "MILPModel.solve"]].sum()


    def write_exec_time(self):
        """
        Writes the time spent on:

        * decision variables declaration ('MILPModel.declare_variables')
        * constraints and KPIs declaration ('MILPModel.declare_constraints_and_KPIs')
        * solving process, in seconds ('MILPModel.solve')
        * solving process, in deterministic Cplex ticks ('Solving deterministic time')
        * total time (variables, constraints, KPIs, solving), in seconds ('Total time')

        """
        file_name = "EXEC TIME __ " + self._name + ".csv"
        path = Path(self.working_dir, file_name)
        self._exec_time.to_csv(path)

    @_dump_status_checker
    def _get_LP(self):
        self.lp_string = self._MILPModel._model_data.mdl.export_as_lp_string()

    @_dump_status_checker
    def _get_MPS(self):
        self.mps_string = self._MILPModel._model_data.mdl.export_as_mps_string()

    @_dump_status_checker
    def _get_KPIs(self):
        KPI_names = self._MILPModel._model_data.KPI_mapper.keys()
        pbar = tqdm(KPI_names, desc="Fetching KPIs")
        for KPI_name in pbar:
            pbar.set_postfix_str(f"{KPI_name}")
            self._convert_KPI_to_dataframe(KPI_name)


    def write_KPIs(self):
        """
        Writes all KPIs defined at component level in a directory named 'KPIs'.

        """
        pbar = tqdm(self._KPI_dataframes.items(), desc="Writing KPIs")
        for KPI_name, df in pbar:
            pbar.set_postfix_str(f"{KPI_name}")
            file_name = "{:s} __ {:s}.csv".format(self._name, KPI_name)
            Path(self.working_dir, "KPIs").mkdir(parents=True, exist_ok=True)
            path = Path(self.working_dir, "KPIs", file_name)
            df.to_csv(path,
                      index=False,
                      float_format="%.{:d}f".format(self._csv_precision),
                      )


    @_dump_status_checker
    def _get_variables(self):
        pbar = tqdm(self._MILPModel._model_data.var_mapper.keys(), desc="Fetching variables")
        for var_name in pbar:
            pbar.set_postfix_str(f"{var_name}")
            self._convert_var_to_dataframe(var_name)


    def write_variables(self):
        """
        Writes all decision variables in a directory named 'Variables'.

        """
        path = Path(self.working_dir, "Variables")
        Path(path).mkdir(exist_ok=True)
        pbar = tqdm(self._var_dataframes.items(), desc="Writing variables")
        for var_name, df in pbar:
            pbar.set_postfix_str(f"{var_name}")
            df.to_csv(Path(path, self._name + " __ " + var_name + ".csv"),
                      float_format="%.{:d}f".format(self._csv_precision),
                      index=False)

    @_dump_status_checker
    def _get_model_complexity(self):
        self.model_complexity = Series(self._MILPModel._model_data.mdl.statistics.__dict__)


    def write_model_complexity(self):
        """
        Writes the number of decision variables and constraints per type (continuous, discrete, etc...).

        """
        file_name = "COMPLEXITY __ " + self._name + ".csv"
        path = Path(self.working_dir, file_name)
        self.model_complexity.to_csv(path)

    @_dump_status_checker
    def _get_solution_summary(self):
        self.solution_summary = Series(self._MILPModel._model_data.mdl.solution.solve_details.as_worker_dict())
        self.solution_summary["Objective value"] = self._MILPModel._model_data.mdl.solution.objective_value
        for kind in ["Eco", "CO2", "Exergy"]:
            self.solution_summary[kind] = self._MILPModel._model_data.mdl.solution.\
                get_value(self._MILPModel._model_data.total_KPI[kind])

    def _get_description(self):
        self._description = self._MILPModel.description.copy()

    def write_description(self):
        """
        Writes the `description` attribute of `MILPModel`.

        """
        file_name = "DESCRIPTION __ " + self._name + ".csv"
        path = Path(self.working_dir, file_name)
        Series(self._description).to_csv(path)

    def write_solution_summary(self):
        """
        Writes important information about the solved model, including the values of the 3 main KPIs, "Eco", "CO2" and "Exergy".

        """
        file_name = "SUMMARY __ " + self._name + ".csv"
        path = Path(self.working_dir, file_name)
        self.solution_summary.to_csv(path)

    @_dump_status_checker
    def _get_unsatisfied_constraints(self):
        self.unsatisfied_constraints = self._MILPModel._model_data.mdl.solution.find_unsatisfied_constraints(self._MILPModel._model_data.mdl)
        self.unsatisfied_constraints = list(map(str, self.unsatisfied_constraints))


    def write_unsatisfied_constraints(self):
        """
        Writes every constraint of the model that is unsatisfied after model solve, given a tolerance of 1e-6.

        """
        file_name = "UNSATISFIED CTS __ " + self._name + ".txt"
        path = Path(self.working_dir, file_name)
        with open(path, "w") as f:
            f.write("\n".join(self.unsatisfied_constraints))


    def write_LP(self):
        """
        Writes the model according to Cplex LP standard.

        The `get_LP` argument must have been set to 'True' at instance creation.

        Notes
        -----
        Cplex LP standard slightly differs from the conventional LP standard.
        Prefer the use `write_MPS` in case a communication with another solver is needed

        """
        if hasattr(self, "lp_string"):
            file_name = "LP __ " + self._name + ".txt"
            path = Path(self.working_dir, file_name)
            with open(path, "w") as f:
                f.write(self.lp_string)
        else:
            print(f"Warning: {self}: please call 'ResultsExport' with 'get_LP=True' for this method call to work.")

    def write_MPS(self):
        """
        Writes the model according to the MPS standard.

        The `get_MPS` argument must have been set to 'True' at instance creation.

        """
        if hasattr(self, "mps_string"):
            file_name = "MPS __ " + self._name + ".mps"
            path = Path(self.working_dir, file_name)
            with open(path, "w") as f:
                f.write(self.mps_string)
        else:
            print(f"Warning: {self}: please call 'ResultsExport' with 'get_MPS=True' for this method call to work.")

    def dump_object(self):
        """
        Writes the binary form of this instance.

        Notes
        -----
        1. Model data (decision variables, constraints, KPIs, etc...) are lost during this operation.
        2. The binary form of this instance is called 'results `name`' with `name` the `name` attribute of `MILPModel`.
        3. The `description` attribute of `MILPModel` is also dumped to speed up the call of `ResultsBatch.create_batch` method.

        """
        path = Path(self.working_dir, "results {:s}".format(self._name))
        with open(path, "wb") as file:
            dump(self, file)

        path = Path(self.working_dir, "description {:s}".format(self._name))
        with open(path, "wb") as file:
            dump(self._description, file)

    def write_all(self):
        """
        Writes all the content of `MILPModel` collected by this instance.

        """
        if hasattr(self, "lp_string"):
            self.write_LP()
        if hasattr(self, "mps_string"):
            self.write_MPS()
        self.write_model_complexity()
        self.write_description()
        if self._MILPModel._last_solve_ok:
            self.write_variables()
            self.write_KPIs()
            self.write_solution_summary()
            self.write_unsatisfied_constraints()
            self.write_exec_time()

    def __repr__(self):
        return self._name + " (results)"


class ResultsBatch(ResultsIO):

    def __init__(self, working_dir, name=None):
        """
        Do not use. See `create_batch` and `create_batch_from_binaries` class methods.

        """
        self._working_dir = Path(working_dir)
        self._is_loaded = False
        self._name = name if name is not None else "Results batch"

    @classmethod
    def create_batch(cls, working_dir, description_matcher,
                   keep_unknown_file_descriptor=True,
                   keep_unknown_matcher_descriptor=True,
                   upfront_processing_func=None,
                   keep_results_exports=False,
                   name=None,
                   **upfront_processing_kwargs):
        """
        A ResultsBatch instance binds ResultsExport instances together to ease the process of results analysis.

        Important attributes of ResultsBatch instances are:
        * `vars_` : concatenation of the decision variables of the ResultsExport instances
        * `KPIs` : concatenation of the KPIs of the ResultsExport instances
        * `results` : other information about ResultsExport instances

        This method creates a ResultsBatch instance from ResultsExports stored on disk.
        It must be used when various optimization results are stored in the same directory and clearly identified by
        the `description` attribute of MILPModel instances.

        The ResultsExport instances to keep are specified using the `keep_unknown_file_descriptor` and
        `keep_unknown_matcher_descriptor` attributes.

        Parameters
        ----------
        working_dir: str or path-like
            The directory containing the ResultsExport results.
            This directory is typically the same than the one specified for the `parent_working_dir` argument of ResultsExport instances.
            This directory is also the one where the aggregated results will be written.
        description_matcher: dict {str: list(int | float | str) | None}
            Any ResultsExport in `working_dir` having a `description` attribute matching `description_matcher` is kept.
            Let `key` and `values` such that description_matcher[key] = values.

            * If `values` is None, every ResultsExports having `key` as a key of its `description` attribute is kept.
            * Else, `values` is a list of int, float and str.
              A ResultsExports having `key` as a key of its `description` attribute is kept only if description[key] is one of `values`.

        keep_unknown_file_descriptor : bool, optional, default True
            If False, ResultsExports instances whose `description` attribute contains keys that are not keys of `description_matcher` are discarded.
            If True, these instances are kept only if the the conditions specified by the `description_matcher` attribute are met.
        keep_unknown_matcher_descriptor: bool, optional, default True
            If False, ResultsExports instances whose `description` attribute lacks keys that are keys of `description_matcher` are discarded.
            If True, these instances are kept only if the the conditions specified by the `description_matcher` attribute are met.
        upfront_processing_func : callable f(x), optional.
            Useful to process ResultsExport instances before they are aggregated into one single ResultsBatch.
            f must accept a ResultsExport instance and performs inplace.
            Attribute `description` of `x` must not be modified.
        keep_results_exports : bool, optional, default False
            If True, all the ResultsExport instances that match the selection criteria are kept in the `results_exports` attribute (dict).
        name: str, optional
            Name of this instance.
            Setting a name may prevent the deletion of existing files during disk dump of this instance.
        upfront_processing_kwargs : optional
            Keyword arguments passed to the `upfront_processing_func` callable, like: f(x, **kwargs).

        Returns
        -------
        RB: ResultsBatch

        Notes
        -----
        1. The `relevant_descriptors` attribute is a dict summing-up the `description` attribute of the MILPModel instances.
        2. The 'batch_analysis' Jupyter Notebook requires a ResultsBatch instance.

        """
        RB = cls(working_dir, name)
        files = listdir(RB._working_dir)
        to_collect_pp = {}
        pbar = tqdm(files, desc="Reading files")
        relevant_descriptors = dict()
        for file in pbar:
            pbar.set_postfix_str(file)
            path = Path(RB._working_dir) / Path(fr"{file}/results {file}")
            path_desc = Path(RB._working_dir) / Path(fr"{file}/description {file}")
            try:
                with open(str(path_desc), "rb") as f:
                    description = load(f)
                conditions = {}
                description_set = set(description)
                description_matcher_set = set(description_matcher)
                
                all_descriptors = description_set | description_matcher_set
                for descriptor in all_descriptors:
                    if descriptor in description_set.difference(description_matcher_set):
                        condition = keep_unknown_file_descriptor

                    elif descriptor in description_matcher_set.difference(description_set):
                        condition = keep_unknown_matcher_descriptor

                    else:
                        assert descriptor in description_matcher_set.intersection(description_set)
                        possible_values = description_matcher[descriptor]
                        condition = (possible_values is None) or (description[descriptor] in possible_values)
                    conditions[descriptor] = condition
                    if not condition:
                        break
                    

                if all(conditions.values()):
                    for pp2_name, pp2 in to_collect_pp.items():
                        if description == pp2._description:
                            raise NotImplementedError(
                                f"{cls}: '{pp2}' and another ResultsExport share the same description, "
                                f"cannot proceed.")
                    pp = cls.load_object(str(path))
                    to_collect_pp[repr(pp)] = pp
                    for descriptor in all_descriptors:
                        if descriptor not in relevant_descriptors:
                            relevant_descriptors[descriptor] = set()
                        if descriptor in description:
                            relevant_descriptors[descriptor] |= set([description[descriptor]])
            except FileNotFoundError as e:
                pass
            except NotADirectoryError as e:
                pass
        print("Kept results:\n", "\n ".join(map(repr, to_collect_pp)))
        RB.relevant_descriptors = relevant_descriptors
        vars_list, KPIs_list, results_list = [], [], []
        for pp_name, pp in to_collect_pp.items():
            vars_, KPIs, results = RB._collect(pp, upfront_processing_func, **upfront_processing_kwargs)
            vars_list.append(vars_)
            KPIs_list.append(KPIs)
            results_list.append(results)
        RB._bind(vars_list, KPIs_list, results_list)
        if keep_results_exports:
            RB.results_exports = to_collect_pp
        else:
            RB.results_exports = {}
        return RB

    @classmethod
    def create_batch_from_binaries(cls,
                                   working_dir,
                                   results_exports: "List of pp",
                                   upfront_processing_func=None,
                                   name=None,
                                   **upfront_processing_kwargs):
        """
        A ResultsBatch instance binds ResultsExport instances together to ease the process of results analysis.

        Important attributes of ResultsBatch instances are:
        * `vars_`: concatenation of the decision variables of the ResultsExport instances
        * `KPIs`: concatenation of the KPIs of the ResultsExport instances
        * `results`: other information about ResultsExport instances

        This method creates a ResultsBatch instance from a list of ResultsExports.

        Parameters
        ----------
        working_dir: str or path-like
            The directory where the aggregated results will be written.
        results_exports: list of ResultsExport
            ResultsExport instances that define the ResultsBatch instance. Duplicates will be ignored.
        upfront_processing_func : callable f(x), optional.
            Useful to process ResultsExport instances before they are aggregated into one single ResultsBatch.
            f must accept a ResultsExport instance and performs inplace.
            Attribute `description` of `x` must not be modified.
        name: str, optional
            Name of this instance.
            Setting a name may prevent the deletion of existing files during disk dump of this instance.
        upfront_processing_kwargs : optional
            Keyword arguments passed to the `upfront_processing_func` callable, like: f(x, **kwargs).

        Returns
        -------
        RB: ResultsBatch

        """
        RB = cls(working_dir, name)
        to_collect_pp = {}
        relevant_descriptors = dict()
        results_exports = set(results_exports)
        for pp in results_exports:
            description = pp._description

            to_collect_pp[repr(pp)] = pp
            for descriptor in set(description):
                if descriptor not in relevant_descriptors:
                    relevant_descriptors[descriptor] = set()
                relevant_descriptors[descriptor] |= {description[descriptor]}

        RB.relevant_descriptors = relevant_descriptors
        vars_list, KPIs_list, results_list = [], [], []
        for pp_name, pp in to_collect_pp.items():
            vars_, KPIs, results = RB._collect(pp, upfront_processing_func, **upfront_processing_kwargs)
            vars_list.append(vars_)
            KPIs_list.append(KPIs)
            results_list.append(results)
        RB._bind(vars_list, KPIs_list, results_list)
        RB.results_exports = {repr(pp): pp for pp in results_exports}
        return RB


    def _collect(self, pp, upfront_processing_func, **upfront_processing_kwargs):
        vars_ = {}
        KPIs = {}

        values_base = {descriptor: pp._description.get(descriptor, nan)
                       for descriptor in self.relevant_descriptors}
        if upfront_processing_func:
            upfront_processing_func(pp, **upfront_processing_kwargs)
        values = values_base.copy()
        values["Solving time"] = pp._MILPModel._exec_time["MILPModel.solve"]
        values["Solving deterministic time"] = pp._MILPModel._exec_time["Solving deterministic time"]
        values["Declaration time"] = pp._MILPModel._exec_time.filter(like="declare").sum()
        values["Progress gap"] = pp.solution_summary["PROGRESS_GAP"]
        values.update(pp.solution_summary[["Eco", "CO2", "Exergy", "Objective value"]].to_dict())

        for KPI_name in pp._KPI_dataframes:
            df = pp._KPI_dataframes[KPI_name]
            for key, value in values_base.items():
                df[key] = value
            KPIs[KPI_name] = df

        for var_name, var in pp._var_dataframes.items():
            for key, value in values_base.items():
                var[key] = value
            vars_[var_name] = var
        results = values

        return vars_, KPIs, results

    def _bind(self, vars_list, KPIs_list, results_list):
        self.vars_ = {}
        all_vars_names = set(chain(*vars_list))
        for var_name in all_vars_names:
            to_concat = []
            for vars_ in vars_list:
                var = vars_.get(var_name, DataFrame())
                if not var.empty: # note: var in vars_ may exist but be empty
                    to_concat.append(var)
            
            self.vars_[var_name] = concat(to_concat) if to_concat else DataFrame()
            
        
        self.KPIs = {}
        all_KPIs_names = set(chain(*KPIs_list))
        for KPI_name in all_KPIs_names:
            to_concat = []
            for KPIs in KPIs_list:
                KPI = KPIs.get(KPI_name, DataFrame())
                if not KPI.empty: # note: KPI in KPIs may exist but be empty
                    to_concat.append(KPI)
            
            self.KPIs[KPI_name] = concat(to_concat) if to_concat else DataFrame()
        


        all_KPIs_names = set(chain(*KPIs_list))
        self.KPIs = {KPI_name: concat([KPIs.get(KPI_name, DataFrame()) for KPIs in KPIs_list])
                for KPI_name in all_KPIs_names}


        self.results = DataFrame.from_records(results_list)

    @classmethod
    def sum(cls, RBs):
        """
        Defines a unique ResultsBatch instance that is the sum of several instances.

        Attributes `vars_`, `KPIs` and `results` are concatenated and merged.
        Attributes `relevant_indicators` are merged.

        Parameters
        ----------
        RBs : list of ResultsBatch

        Returns
        -------
        RB: ResultsBatch
            A ResultsBatch instance

        Notes
        -----
        1. The addition operator '+' may be used instead of a call to `sum(RBs)` for `RBs` of length 2,
           i.e. these two lines perform the same operation:

            >>> ResultsBatch.sum([RB1, RB2])
            >>> RB1 + RB2

        2. The `working_dir` attribute of `RB` is the one of the first element of `RBs`.

        """
        assert RBs and all([isinstance(RB, cls) for RB in RBs]), f"'RBs' must be an iterable of ResultsBatch objects."
        new_RB = cls(RBs[0]._working_dir, name="Results batch sum")
        new_RB._is_loaded = any([RB._is_loaded for RB in RBs])
        new_RB.results = concat([RB.results for RB in RBs], ignore_index=True)       # old: merge(X, Y, how="outer")

        all_vars_names = set(chain(*[RB.vars_ for RB in RBs]))
        new_RB.vars_ = {}
        for var_name in all_vars_names:
            vars_list = [RB.vars_.get(var_name, DataFrame())
                         for RB in RBs]
            new_RB.vars_[var_name] = concat(vars_list, ignore_index=True)


        all_KPIs_names = set(chain(*[RB.KPIs for RB in RBs]))
        new_RB.KPIs = {}
        for KPI_name in all_KPIs_names:
            KPIs_list = [RB.KPIs.get(KPI_name, DataFrame())
                         for RB in RBs]
            new_RB.KPIs[KPI_name] = concat(KPIs_list, ignore_index=True)

        new_RB.relevant_descriptors = reduce(lambda x, y: {k: x.get(k, set()) | y.get(k, set()) for k in set(x)|set(y)},
                                             [RB.relevant_descriptors for RB in RBs])
        return new_RB
    
    def remove_descriptors(self, descriptors, check_unique=False):
        """
        Removes some descriptors, i.e. keys of attribute `relevant_descriptor`.

        Parameters
        ----------
        descriptors : list of str
            Must be a subset of the keys of `relevant_descriptors`.
        check_unique: bool, optional, default False
            Describes the method behavior when any of the descriptor in `descriptors` is associated with multiple values:

                * If True, an AssertionError is raised.
                * If False, descriptor is removed, leading to possible duplicated rows in attributes `vars_`, `KPIs` and `results`

        Returns
        -------

        """
        descriptors = set(descriptors)
        assert set(descriptors).issubset(set(self.relevant_descriptors)), \
            f"'descriptors' must be an iterable of descriptors used by self."
        for descriptor in descriptors:
            for var_name, var in self.vars_.items():
                assert not (len(var[descriptor].unique()) != 1 and check_unique)
                self.vars_[var_name] = var.drop(descriptor, axis=1)
            for KPI_name, KPI in self.KPIs.items():
                assert not (len(KPI[descriptor].unique()) != 1 and check_unique)
                self.KPIs[KPI_name] = KPI.drop(descriptor, axis=1)
            self.results = self.results.drop(descriptor, axis=1)
        self.relevant_descriptors = {k: v for k, v in self.relevant_descriptors.items()
                                     if k not in descriptors}
    
    def dump_object(self):
        """
        Writes the binary form of this instance in the directory `working_dir`.

        """
        path = self.working_dir /  Path(self._name)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / self._name, "wb") as file:
            dump(self, file)

    def write_all(self):
        """
        Writes the `vars_`, `KPIs` and `results` attributes as CSV files, in the directory `working_dir`.

        """
        path = self.working_dir /  Path(self._name)
        path.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(path / "Results.csv", index=False)
        (path / "Variables").mkdir(parents=True, exist_ok=True)
        [df.to_csv(Path(path / f"Variables/{var_name}.csv")) for var_name, df in self.vars_.items()]
        (path / "KPIs").mkdir(parents=True, exist_ok=True)
        [df.to_csv(Path(path / f"KPIs/{KPI_name}.csv")) for KPI_name, df in self.KPIs.items()]


    def __add__(self, other):
        return self.sum([self, other])

    def __repr__(self):
        return self._name

