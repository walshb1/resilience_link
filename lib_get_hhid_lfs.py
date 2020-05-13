import pandas as pd

def get_hhid_lfs(df):

	df['hhid_lfs'] = (df['w_prov'].map(str)
    				  + df['w_mun'].map(str)
                      + df['w_bgy'].map(str)
                      + df['w_ea'].map(str)
                      + df['w_shsn'].map(str)
                      + df['w_hcn'].map(str)).astype(str)

	# if not df.index.is_unique:
    	# df.loc[df.index.duplicated()].to_csv('csv/get_hhid_lfs_nonunique.csv')
    	# assert(False)

def get_hhid_FIES(df):
    df['hhid'] =  df['w_regn'].astype('str')
    df['hhid'] += df['w_prov'].astype('str')
    df['hhid'] += df['w_mun'].astype('str')
    df['hhid'] += df['w_bgy'].astype('str')
    df['hhid'] += df['w_ea'].astype('str')
    df['hhid'] += df['w_shsn'].astype('str')
    df['hhid'] += df['w_hcn'].astype('str')