from glob import glob

import pandas as pd
import polars as pl


class Pipeline:
    @staticmethod
    def set_table_dtypes(df):  # Standardize the dtype.
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))

        return df

    @staticmethod
    def handle_dates(
        df,
    ):  # Change the feature for D to the difference in days from date_decision.
        for col in df.columns:
            if (col[-1] in ("D",)) and ("count" not in col):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())

        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(
        df,
    ):  # Remove those with an average is_null exceeding 0.95 and those that do not fall within the range 1 < nunique < 200.
        drop_cols = []
        for col in df.columns:
            # if col not in ["target", "case_id", "WEEK_NUM"]:
            #    isnull = df[col].is_null().mean()
            #    if isnull > 0.95:
            #        drop_cols.append(col)

            if (
                col
                not in [
                    "target",
                    "case_id",
                    "WEEK_NUM",
                ]
            ) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (
                    freq > 10
                ):  # 50 #len(df) * 0.20): # 95 # fe4 down at fq20
                    drop_cols.append(col)

            # eliminate yaer, month feature
            # 644
            if (col[-1] not in ["P", "A", "L", "M"]) and (
                ("month_" in col) or ("year_" in col)
            ):  # or ('num_group' in col):
                # if (('month_' in col) or ('year_' in col)):# or ('num_group' in col):
                drop_cols.append(col)

        return drop_cols


class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]

        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_{col}") for col in cols]
        # expr_3 = [pl.median(col).alias(f"median_{col}") for col in cols]
        # expr_3 = [pl.var(col).alias(f"var_{col}") for col in cols]+ [pl.sum(col).alias(f"sum_{col}") for col in cols]
        # expr_3 = [pl.last(col).alias(f"last_{col}") for col in cols] #+ \
        #     [pl.first(col).alias(f"first_{col}") for col in cols] + \
        #     [pl.mean(col).alias(f"mean_{col}") for col in cols] + \
        #     [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_3 = [pl.count(col).alias(f"count_{col}") for col in cols]

        cols2 = [col for col in df.columns if col[-1] in ("L", "A")]
        expr_3 = (
            [pl.mean(col).alias(f"mean_{col}") for col in cols2]
            + [pl.std(col).alias(f"std_{col}") for col in cols2]
            + [pl.sum(col).alias(f"sum_{col}") for col in cols2]
            + [pl.median(col).alias(f"median_{col}") for col in cols2]
        )  # + \
        # [pl.first(col).alias(f"first_{col}") for col in cols2] + [pl.last(col).alias(f"last_{col}") for col in cols2]

        # BAD
        # cols3 = [col for col in df.columns if col[-1] in ("A")]
        # expr_4 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_{col}")
        #           for col in cols3]
        return (
            expr_1 + expr_2 + expr_3
        )  # + [pl.col(col).diff().last().alias(f"diff-last_{col}") for col in cols3] # + expr_4

    @staticmethod
    def bureau_a1(df):
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_{col}") for col in cols]

        cols2 = [
            # bad
            "annualeffectiverate_199L",
            "annualeffectiverate_63L",
            "contractsum_5085717L",
            "credlmt_230A",
            "credlmt_935A",
            # 'debtoutstand_525A', 'debtoverdue_47A', 'dpdmax_139P', 'dpdmax_757P',
            #    'instlamount_768A', 'instlamount_852A',
            #    'interestrate_508L', 'monthlyinstlamount_332A',
            #    'monthlyinstlamount_674A',
            # good?
            "nominalrate_281L",
            "nominalrate_498L",
            "numberofcontrsvalue_258L",
            "numberofcontrsvalue_358L",
            "numberofinstls_229L",
            "numberofinstls_320L",
            "numberofoutstandinstls_520L",
            "numberofoutstandinstls_59L",
            "numberofoverdueinstlmax_1039L",
            "numberofoverdueinstlmax_1151L",
            "numberofoverdueinstls_725L",
            "numberofoverdueinstls_834L",
            # bad?
            #    'outstandingamount_354A', 'outstandingamount_362A', 'overdueamount_31A',
            #    'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2_398A',
            #    'overdueamountmax_155A', 'overdueamountmax_35A',
            # bad ?
            #    'periodicityofpmts_1102L', 'periodicityofpmts_837L',
            #    'prolongationcount_1120L', 'prolongationcount_599L',
            # 520?
            #    'residualamount_488A', 'residualamount_856A', 'totalamount_6A',
            #    'totalamount_996A', 'totaldebtoverduevalue_178A',
            #    'totaldebtoverduevalue_718A', 'totaloutstanddebtvalue_39A',
            #    'totaloutstanddebtvalue_668A',
        ]

        # .697
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2]

        # .696
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2]

        # .697
        # expr_3 = [pl.std(col).alias(f"std_{col}") for col in cols2]

        # .6985
        # expr_3 = [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .696
        # expr_3 = [pl.sum(col).alias(f"sum_{col}") for col in cols2]

        # .6981
        # expr_3 = [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .696
        # expr_3 = [pl.first(col).alias(f"first_{col}") for col in cols2] + [pl.last(col).alias(f"last_{col}") for col in cols2] # + \

        # .696
        # expr_3 = [pl.std(col).alias(f"std_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .699
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2] + \
        #     [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        expr_3 = (
            [pl.mean(col).alias(f"mean_{col}") for col in cols2]
            + [pl.std(col).alias(f"std_{col}") for col in cols2]
            + [pl.sum(col).alias(f"sum_{col}") for col in cols2]
            + [pl.median(col).alias(f"median_{col}") for col in cols2]
            + [pl.first(col).alias(f"first_{col}") for col in cols2]
        )  # + [pl.last(col).alias(f"last_{col}") for col in cols2] # not applied

        # expr_3 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_depth2_{col}") for col in cols2]
        return expr_1 + expr_2 + expr_3

    @staticmethod
    def deposit_exprs(df):
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [
            pl.min(col).alias(f"min_{col}") for col in cols
        ]  # + \
        # [pl.last(col).alias(f"last_{col}") for col in cols]
        # [pl.mean(col).alias(f"mean_{col}") for col in cols] # + \
        # [pl.std(col).alias(f"std_{col}") for col in cols]  + \

        # [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_2 = [pl.first('openingdate_857D').alias(f'first_openingdate_857D')] + [pl.last('openingdate_857D').alias(f'last_openingdate_857D')]

        return expr_1  # + expr_2 #+ expr_ngmax

    @staticmethod
    def debitcard_exprs(df):
        # cols = [col for col in df.columns if (col[-1] in ["A"])]
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [
            pl.min(col).alias(f"min_{col}") for col in cols
        ]
        # [pl.mean(col).alias(f"mean_{col}") for col in cols] + \
        # [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_2 = [pl.first('openingdate_857D').alias(f'first_openingdate_857D')] + [pl.last('openingdate_857D').alias(f'last_openingdate_857D')]

        return expr_1  # + expr_2 #+ expr_ngmax
        # return expr_1

    @staticmethod
    def person_expr(df):
        cols1 = [
            "empl_employedtotal_800L",
            "empl_employedfrom_271D",
            "empl_industry_691L",
            "familystate_447L",
            "incometype_1044T",
            "sex_738L",
            "housetype_905L",
            "housingtype_772L",
            "isreference_387L",
            "birth_259D",
        ]
        # cols1 = [col for col in df.columns]
        expr_1 = [pl.first(col).alias(f"first_{col}") for col in cols1]

        expr_2 = [
            pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
            pl.col("mainoccupationinc_384A")
            .filter(pl.col("incometype_1044T") == "SELFEMPLOYED")
            .max()
            .alias("mainoccupationinc_384A_any_selfemployed"),
        ]

        # No Effect ...
        # cols = ['personindex_1023L', 'persontype_1072L', 'persontype_792L']
        # expr_3 = [pl.col(col).last().alias(f"last_{col}") for col in cols] + [pl.col(col).drop_nulls().mean().alias(f"mean_{col}") for col in cols]

        # cols2 = [col for col in df.columns if col not in cols1]
        # expr_4 = [pl.max(col).alias(f"max_{col}") for col in cols2] + [pl.min(col).alias(f"min_{col}") for col in cols2] #  good at cv, bad at lb ?
        # [pl.col(col).drop_nulls().last().alias(f"last_{col}") for col in cols2] + [pl.col(col).drop_nulls().first().alias(f"first_{col}") for col in cols2] # no effect

        return expr_1 + expr_2  # + expr_4 # + expr_3

    @staticmethod
    def person_2_expr(df):
        # cols = [col for col in df.columns]
        cols = [
            "empls_economicalst_849M",
            "empls_employedfrom_796D",
            "empls_employer_name_740M",
        ]  # + \
        # ['relatedpersons_role_762T', 'conts_role_79M']
        # ['addres_district_368M', 'addres_role_871L', 'addres_zip_823M']

        expr_1 = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_2 = [pl.last(col).alias(f"last_{col}") for col in cols]

        # BAD
        # expr_ngc = [pl.count("num_group2").alias(f"count_num_group2")]
        # cols2 = [col for col in df.columns if (col in ("num_group1", "num_group2"))]
        # expr_ngmax = [pl.min(col).alias(f"min_{col}") for col in cols2] + [pl.max(col).alias(f"max_{col}") for col in cols2]

        # cols2 = [col for col in df.columns if col not in cols]
        # # expr_3 = [pl.max(col).alias(f"max_{col}") for col in cols2] + [pl.min(col).alias(f"min_{col}") for col in cols2] # no effect
        # expr_3 = [pl.col(col).drop_nulls().last().alias(f"last_{col}") for col in cols2] # no effect

        return expr_1 + expr_2  # + expr_3# + expr_ngc

    @staticmethod
    def other_expr(df):
        expr_1 = [
            pl.first(col).alias(f"__other_{col}")
            for col in df.columns
            if ("num_group" not in col) and (col != "case_id")
        ]
        # cols1 = ['amtdepositbalance_4809441A', 'amtdepositincoming_4809444A', 'amtdepositoutgoing_4809442A']
        # expr_1 = [pl.last(col).alias(f"last_{col}") for col in cols1]
        # cols2 = ['amtdebitincoming_4809443A', 'amtdebitoutgoing_4809440A']
        # expr_3 = [(pl.col('amtdebitincoming_4809443A') - pl.col('amtdebitoutgoing_4809440A')).alias('amtdebit_incoming-outgoing')]
        return expr_1  # + expr_2 + expr_3

    @staticmethod
    def tax_a_exprs(df):
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]
        expr_1 = (
            [pl.max(col).alias(f"max_{col}") for col in cols]
            + [pl.min(col).alias(f"min_{col}") for col in cols]
            + [pl.last(col).alias(f"last_{col}") for col in cols]
            + [pl.first(col).alias(f"first_{col}") for col in cols]
            + [pl.mean(col).alias(f"mean_{col}") for col in cols]
            + [pl.std(col).alias(f"std_{col}") for col in cols]
        )
        # expr_1 = [pl.max(col).alias(f"max_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'num_group1']] + \
        #     [pl.min(col).alias(f"min_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', ]] + \
        #     [pl.mean(col).alias(f"mean_{col}") for col in ['amount_4527230A']] + \
        #     [pl.std(col).alias(f"std_{col}") for col in ['amount_4527230A']] + \
        #     [pl.last(col).alias(f"last_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'name_4527232M']] + \
        #     [pl.first(col).alias(f"first_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'name_4527232M']] # BAD?

        expr_4 = [
            pl.col(col)
            .fill_null(strategy="zero")
            .map_elements(lambda x: x.max() - x.min(), return_dtype=pl.Float32)
            .alias(f"max-min_gap_depth2_{col}")
            for col in ["amount_4527230A"]
        ]

        return expr_1 + expr_4

    @staticmethod
    def bureau_a2(df):  # 122만
        # cols = ['collater_valueofguarantee_1124L', 'pmts_dpd_1073P', 'pmts_overdue_1140A',]
        cols = [
            col
            for col in df.columns
            if (col[-1] in ("T", "L", "M", "D", "P", "A")) or ("num_group" in col)
        ]

        expr_1 = [pl.max(col).alias(f"max_depth2_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_depth2_{col}") for col in cols]
        expr_3 = [pl.mean(col).alias(f"mean_depth2_{col}") for col in cols] + [
            pl.std(col).alias(f"std_{col}") for col in cols
        ]
        # expr_ngs = [pl.max(col).alias(f"max_{col}") for col in ['num_group1', 'num_group2', ]]

        expr_4 = [
            pl.col(col)
            .fill_null(strategy="zero")
            .map_elements(lambda x: x.max() - x.min(), return_dtype=pl.Float32)
            .alias(f"max-min_gap_depth2_{col}")
            for col in [
                "collater_valueofguarantee_1124L",
                "pmts_dpd_1073P",
                "pmts_overdue_1140A",
            ]
        ]

        expr_ngc = [pl.count("num_group2").alias(f"count_depth2_a2_num_group2")]

        # expr_5 = [pl.last(col).alias(f"last_{col}") for col in cols] + \
        #     [pl.first(col).alias(f"first_{col}") for col in cols] + \
        #     [pl.std(col).alias(f"std_{col}") for col in cols]

        return expr_1 + expr_2 + expr_3 + expr_4 + expr_ngc  # + expr_5

    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df)

        return exprs

    # no use from here
    @staticmethod
    def applprev2_exprs(df):
        cols = [col for col in df.columns if "num_group" not in col]
        # expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_2 = [
            pl.first(col).alias(f"first_{col}") for col in cols
        ]  #  + [pl.last(col).alias(f"last_{col}") for col in cols]
        return []  # expr_2

    @staticmethod
    def bureau_b1(df):  # 0.95 filterにかかるため未使用
        # cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        # expr_1 = [pl.max(col).alias(f"bureau_b1_max_{col}") for col in cols]
        # expr_2 = [pl.min(col).alias(f"bureau_b1_min_{col}") for col in cols]

        # return expr_1 + expr_2 #  + expr_3
        return []

    @staticmethod
    def bureau_b2(df):  # 0.95filterにかかるため未使用
        # cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        # expr_1 = [pl.max(col).alias(f"bureau_b2_max_{col}") for col in cols]
        # expr_2 = [pl.min(col).alias(f"bureau_b2_min_{col}") for col in cols]

        # return expr_1 + expr_2 #  + expr_3
        return []


def agg_by_case(path, df):
    path = str(path)
    if "_applprev_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))

    #     elif '_applprev_2' in path:
    #         df = df.group_by("case_id").agg(Aggregator.applprev2_exprs(df))

    elif "_credit_bureau_a_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.bureau_a1(df))

    elif "_credit_bureau_b_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.bureau_b1(df))

    elif "_deposit_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.deposit_exprs(df))
    elif "_debitcard_1" in path:
        df = (
            df.sort("num_group1")
            .group_by("case_id")
            .agg(Aggregator.debitcard_exprs(df))
        )

    elif "_tax_registry_a" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.tax_a_exprs(df))
    elif "_tax_registry_b" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))
    elif "_tax_registry_c" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))

    elif "_other_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.other_expr(df))
    elif "_person_1" in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.person_expr(df))
    elif "_person_2" in path:
        df = df.group_by("case_id").agg(Aggregator.person_2_expr(df))

    elif "_credit_bureau_a_2" in path:
        df = df.group_by("case_id").agg(Aggregator.bureau_a2(df))
    elif "_credit_bureau_b_2" in path:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))

    return df


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)

    if depth in [1, 2]:
        df = agg_by_case(path, df)

    return df


def read_files(regex_path, depth=None):
    print(regex_path)
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = agg_by_case(path, df)
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])

    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        decision_month=pl.col("date_decision").dt.month(),
        decision_weekday=pl.col("date_decision").dt.weekday(),
    )

    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    print(df_data.info())
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
        # cat_cols = [c for c in cat_cols if 'diff_' not in c]

    df_data[cat_cols] = df_data[cat_cols].fillna("Missing").astype("category")

    return df_data, cat_cols
