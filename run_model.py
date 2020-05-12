myCountry = 'PH'
poverty_type = 'cons'

import pandas as pd
from covid_shift import *
from plotting_libraries import plot_avg_time, plot_relative_losses, plot_absolute_losses, middle_class_stuff, plot_avg_annual_displacement
set_directories(myCountry)

master_df,n_hazards = load_hh_df(myCountry)

# slow
# avg_c = get_average_income(myCountry)


#loop over hazards:
for _haz in master_df.index.get_level_values(1).unique():
        
    _ = slice(None)
    df = master_df.loc[(_,[_haz],_,_,_,_,_),[kc for kc in keep_cols if kc in master_df.columns]].sort_index()
    df,fom = slice_on_pop(df,'poor','cons')
    
    df_ev = get_expectation_values(df,myCountry)
    
    try: plot_avg_time(df_ev,_haz,'poor',fom,myCountry)
    except: pass
    
    #df_aal = average_over_rp(df,return_probs=False)


# OUTPUT
try: df_out = pd.read_csv('csv/df_out.csv')
except: df_out = generate_df_out(myCountry,master_df,poverty_type)

plot_avg_annual_displacement(myCountry,poverty_type)
plot_relative_losses(myCountry,df_out,n_hazards)
plot_absolute_losses(myCountry,df_out,n_hazards)
middle_class_stuff(myCountry,master_df)


