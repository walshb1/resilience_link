myCountry = 'PH'
poverty_type = 'cons'

import numpy as np
import pandas as pd
from covid_shift import *
from plotting_libraries import plot_avg_time, plot_relative_losses, plot_absolute_losses, middle_class_stuff, plot_avg_annual_displacement
from resilience_libraries import setup_resilience,plot_resilience,load_covid_income_shock,plot_recovery_time,plot_recovery_time_shift
from maps_libraries import choropleth_reel

set_directories(myCountry)

master_df,n_hazards = load_hh_df(myCountry)

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

# slow
# avg_c = get_average_income(myCountry)

#loop over hazards:
for _haz in master_df.index.get_level_values(1).unique():

	# slice master df on hazard
	_ = slice(None)
	df = master_df.loc[(_,[_haz],_,_,_,_,_),[kc for kc in keep_cols if kc in master_df.columns]].sort_index()

	# covid impact
	df_ev = get_expectation_values(df,myCountry)
	df_ev = setup_resilience(myCountry,df_ev)
	df_ev = load_covid_income_shock(df_ev,'base')

	# recovery times
	plot_recovery_time_shift(df_ev,myCountry)
	plot_recovery_time(master_df,df_ev,myCountry)
	assert(False)


	# make choropleths of resilience
	choropleth_reel(df_ev)
	plot_resilience(myCountry,df_ev)	



	continue
	# slice on pop for recovery times
	df,fom = slice_on_pop(df,'poor','cons')
	df_ev = get_expectation_values(df,myCountry)

	try: plot_avg_time(df_ev,_haz,'poor',fom,myCountry)
	except: pass
    
    #df_aal = average_over_rp(df,return_probs=False)

assert(False)
# OTHER OUTPUT

try: df_out = pd.read_csv('csv/df_out.csv')
except: df_out = generate_df_out(myCountry,master_df,poverty_type)
plot_avg_annual_displacement(myCountry,poverty_type)
plot_relative_losses(myCountry,df_out,n_hazards)
plot_absolute_losses(myCountry,df_out,n_hazards)
middle_class_stuff(myCountry,master_df)


