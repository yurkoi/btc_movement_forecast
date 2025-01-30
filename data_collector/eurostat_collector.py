import eurostat

# LONG WAIT TIME ~15 min, unless you filter data

# https://ec.europa.eu/eurostat/cache/metadata/en/irt_euryld_esms.htm
  # Euro yield curves (irt_euryld)
  # Reference Metadata in Euro SDMX Metadata Structure (ESMS)
  # Compiling agency: Eurostat, the statistical office of the European Uni

# the service is slow, you need to specify addtional params : https://pypi.org/project/eurostat/0.2.3/

filter_pars = {'startPeriod':'2024-01-01', 'endPeriod':'2024-03-01'}

code = 'ei_isir_m'
eurostat_euro_yield_df = eurostat.get_data_df(code, flags=True, verbose=True)

res_df = eurostat_euro_yield_df[(eurostat_euro_yield_df['geo\TIME_PERIOD'] == 'EU27_2020') & (eurostat_euro_yield_df['unit'] == 'RT1') & (eurostat_euro_yield_df['nace_r2'] == 'MIG_NRG')]

res_df.drop(['freq', 'indic', 'nace_r2', 'unit', 'geo\TIME_PERIOD'], axis=1, inplace=True)
df = res_df.T
df.columns = ['values']

df_cleaned = df[~df.index.str.contains('_flag')]
df_cleaned.index = df_cleaned.index.str.replace('_value', '', regex=False)
df_cleaned = df_cleaned.reset_index().rename(columns={'index': 'date'})