import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid_shift import *
from predictive_libraries import *
from lib_get_hhid_lfs import *

def setup_resilience(myCountry,df_ev,_rp=25):

	# collapse index, leaving only hhid & rp
	df_ev = df_ev.sum(level=['rp','hhid']).reset_index()
	# choose an RP
	df_ev = df_ev.loc[df_ev.rp==_rp]
	# subset will do

	# to ppp per day
	df_ev['c'] *= ppp_factor/365
	
	df_ev['resilience'] = 1E2*df_ev.eval('dk0/dw')

	return df_ev

def plot_resilience(myCountry,df_ev):

	tmp_df = df_ev.sample(5000).sort_values('c',ascending=True)

	plt.scatter(tmp_df['c'],tmp_df['resilience'],alpha=0.2)
	x_new, pfit = df_to_polynomial_fit(tmp_df,'c','resilience',power=4,x_new=None)

	plt.plot(x_new,pfit,color=greys_pal[5])

	plt.xlabel('Income [PPP\$/cap/day]')
	plt.ylabel('Socioeconomic resilience [%]')

	plt.xlim(0,40)
	plt.ylim(0,1000)

	sns.despine(left=True)
	plt.savefig('figures/{}/resilience_vs_income.pdf'.format(myCountry),format='pdf',bbox_inches='tight')
	plt.close('all')

def load_covid_income_shock(df_ev,scode):

	# get FIES <--> LFS ids
	codex = pd.read_csv('~/Desktop/BANK/hh_resilience_model/inputs/PH/fies2015.csv')[['w_regn','w_prov','w_mun','w_bgy','w_ea','w_shsn','w_hcn']]
	get_hhid_FIES(codex)
	get_hhid_lfs(codex)
	codex = codex[['hhid','hhid_lfs']].astype('int')

	df_ev = df_ev.reset_index()
	df_ev['hhid'] = df_ev['hhid'].astype('int')

	df_ev = pd.merge(df_ev.reset_index(drop=True),codex.reset_index(drop=True),on='hhid')

	# load covid info
	df_covid = pd.read_csv('csv/FIES2015_COVID.csv')[['region','hhid_lfs','pcinc_final']].rename(columns={'pcinc_final':'pcinc_ppp_covid'})
	df_covid['pcinc_ppp_covid']*=(12/365)
	df_ev = pd.merge(df_ev.reset_index(drop=True),df_covid.reset_index(drop=True),on='hhid_lfs')
	
	#get new resilience value
	x_new, pfit = df_to_polynomial_fit(df_ev,'c','resilience',power=4,x_new=df_ev['pcinc_ppp_covid'].squeeze().T)
	df_ev['covid_resilience'] = pfit
	df_ev['covid_resilience'] = df_ev['covid_resilience'].clip(lower=0)

	# get new recovery time value
	df_ev['c_inverse'] = 1/(df_ev['c'].clip(lower=1E-5))
	df_ev['pcinc_ppp_covid_inv'] = 1/(df_ev['pcinc_ppp_covid'].clip(lower=1E-5))
	df_ev['recovery_time'] = (np.log(1/0.05)/df_ev['optimal_hh_reco_rate'].clip(lower=1E-5))
	df_ev['log_recovery_time']	= np.log(np.array(df_ev['recovery_time']))

	pfit,_ = df_to_linear_fit(df_ev,'c_inverse','log_recovery_time',x2=df_ev['pcinc_ppp_covid_inv'].values.reshape(-1, 1))
	df_ev['covid_recovery_time'] = np.exp(pfit)/5#hack

	df_ev['covid_recovery_time'].to_csv('~/Desktop/tmp/out.csv')
	return df_ev

def plot_recovery_time(mdf,df,myCountry):

	# print(df.head())
	# df_aff = df.reset_index()
	# df_aff = df_aff.loc[(df_aff.affected_cat=='a')&(df_aff.optimal_hh_reco_rate>0)]
	# df_aff['recovery_time'] = (np.log(1/0.05)/df_aff['optimal_hh_reco_rate'])

	tmp_df = df.sample(20000).sort_values('pcinc_ppp_covid',ascending=False)
	plt.scatter(tmp_df['c'],tmp_df['recovery_time'],alpha=0.2)
	# x_pred,y_pred = df_to_decay_fit(tmp_df,'c','recovery_time')


	plt.plot(tmp_df['pcinc_ppp_covid'],tmp_df['covid_recovery_time'],color=greys_pal[5])

	plt.xlabel('Income [PPP\$/cap/day]')
	plt.ylabel('Recovery time [years]')

	plt.xlim(0,40)
	plt.ylim(0,10)

	sns.despine(left=True)
	plt.savefig('figures/{}/recoverytime_vs_income.pdf'.format(myCountry),format='pdf',bbox_inches='tight')
	plt.close('all')

def plot_recovery_time_shift(df_ev,myCountry):

	df_class = pd.DataFrame({'pre':None,'post':None},index=['sub','pov','vul','sec','mc'])
	print(df_class)

	df_class.loc['sub','pre'] = df_ev.loc[df_ev['c']<=1.90,['recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[df_ev['c']<1.90,'pcwgt'].sum()
	df_class.loc['pov','pre'] = df_ev.loc[(df_ev['c']>1.90)&(df_ev['c']<=3.20),['recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>1.90)&(df_ev['c']<=3.20),'pcwgt'].sum()
	df_class.loc['vul','pre'] = df_ev.loc[(df_ev['c']>3.20)&(df_ev['c']<=5.50),['recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>3.20)&(df_ev['c']<=5.50),'pcwgt'].sum()
	df_class.loc['sec','pre'] = df_ev.loc[(df_ev['c']>5.5)&(df_ev['c']<=15),['recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>5.5)&(df_ev['c']<=15),'pcwgt'].sum()
	df_class.loc['mc','pre'] = df_ev.loc[(df_ev['c']>15),['recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>15),'pcwgt'].sum()

	df_class.loc['sub','post'] = df_ev.loc[df_ev['c']<=1.90,['covid_recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[df_ev['c']<1.90,'pcwgt'].sum()
	df_class.loc['pov','post'] = df_ev.loc[(df_ev['c']>1.90)&(df_ev['c']<=3.20),['covid_recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>1.90)&(df_ev['c']<=3.20),'pcwgt'].sum()
	df_class.loc['vul','post'] = df_ev.loc[(df_ev['c']>3.20)&(df_ev['c']<=5.50),['covid_recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>3.20)&(df_ev['c']<=5.50),'pcwgt'].sum()
	df_class.loc['sec','post'] = df_ev.loc[(df_ev['c']>5.5)&(df_ev['c']<=15),['covid_recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>5.5)&(df_ev['c']<=15),'pcwgt'].sum()
	df_class.loc['mc','post'] = df_ev.loc[(df_ev['c']>15),['covid_recovery_time','pcwgt']].prod(axis=1).sum()/df_ev.loc[(df_ev['c']>15),'pcwgt'].sum()

	print(df_class)

