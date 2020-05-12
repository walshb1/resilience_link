import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob,os
import scipy

from covid_shift import haz_dict, get_currency,keep_cols,slice_on_pop,get_expectation_values,get_population,get_economy,average_over_rp

# formatting & aesthetics
font = {'family':'sans serif', 'size':10}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.facecolor'] = 'white'
sns.set_style("white")

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)




def plot_avg_time(df_ev,_haz,segment,fom,myCountry,yr2mth=True):
    
    sf = 1.
    if yr2mth: sf = 12.  
        
    _ = slice(None)
    
    pop_df = get_population(myCountry,segment).sum(level='quintile').squeeze().to_frame(name='total_population')
    # indexed on quintile
    
    # 
    if segment != 'poor': df_ev[fom] = (10.019268 - df_ev[fom]).clip(lower=0)
    
    # plot weighted means
    # cycle over rps
    for _rp in df_ev.index.get_level_values(2).unique():
            
        _df = df_ev.loc[(_,_,[_rp],_,_),:]

        df_quint = _df[['pcwgt',fom]].prod(axis=1).sum(level='quintile').to_frame(name=fom)
        df_quint[fom] = df_quint[fom]/pop_df['total_population']
        plt.plot(df_quint.index,sf*df_quint[fom],color=greys_pal[6],lw=1.0,zorder=99)
                
        for _nreg, _reg in enumerate(df_ev.index.get_level_values(0).unique()):
            
            _regpop = get_population(myCountry,segment).reset_index(get_economy(myCountry))
            reg_pop = pd.DataFrame(index=get_population(myCountry,None).sum(level='quintile').index)
            reg_pop['tot_pop'] = _regpop.loc[_regpop[get_economy(myCountry)]==_reg].sum(level='quintile')
            reg_pop = reg_pop.fillna(-1)
            
            _dfreg = df_ev.loc[([_reg],_,[_rp],_,_),:]
            df_quint['{}_{}'.format(fom,_nreg)] = _dfreg[['pcwgt',fom]].prod(axis=1).sum(level='quintile')
            df_quint['{}_{}'.format(fom,_nreg)] = df_quint['{}_{}'.format(fom,_nreg)]/reg_pop['tot_pop']
            plt.plot(df_quint.index,sf*df_quint['{}_{}'.format(fom,_nreg)],color=greys_pal[4],alpha=0.25,lw=0.5,zorder=11)
            
        plt.fill_between(df_quint.index,sf*df_quint.min(axis=1),sf*df_quint.max(axis=1),alpha=0.1,color=greys_pal[3],zorder=10)
            
        plt.xlabel('Income quintile',labelpad=8,fontsize=10,linespacing=1.75)
        plt.ylabel('Time in poverty [{}]'.format('months' if yr2mth else 'years'),labelpad=10,fontsize=10,linespacing=1.75)

        plt.gca().tick_params(axis='both', which='major', pad=10,size=4)
        plt.xticks([1,2,3,4,5])
        plt.gca().set_xticklabels(['Poorest\nquintile','second','third','fourth','Wealthiest\nquintile'],ha='center',rotation=0)
        
        #plt.legend()
        sns.despine()
        plt.savefig('figures/{}/duration/income_poverty_{}_{}.pdf'.format(myCountry,_haz,_rp),format='pdf',bbox_inches='tight')
        plt.close('all')

def plot_relative_losses(myCountry,df_out,n_hazards):

	# PLOT relative losses
	barwidth = 0.8

	for nseg, segment in enumerate(['poor','vulnerable','secure','middleclass']):
	    if myCountry == 'HR' and segment == 'poor': continue

	    df_seg = df_out.loc[df_out.eval("segment=='{}'".format(segment))]
	    
	    # loop over hazards
	    _dktot = 0
	    _dwtot = 0
	    #
	    dk_bot = [0,0,0,0]
	    dw_bot = [0,0,0,0]
	    for nhaz, _haz in enumerate(['HU']):

	        _dkabs = df_seg['dk_avg_'+_haz].sum()
	        _dkrel = 1E2*_dkabs/df_seg[['c','total_pop']].prod(axis=1).sum()
	        _dktot += _dkabs
	        
	        plt.bar(2*nseg,_dkrel,bottom=dk_bot[nseg],
	                color=sns_pal[1+nhaz],zorder=99,linewidth=0,edgecolor=None,width=barwidth,alpha=0.4)
	        
	        # Here, should we be plotting dw_pc as fraction of *national average* hh consumption?
	        _dwabs = df_seg['dw_avg_'+_haz].sum()
	        #_dwrel = 1E2*_dwabs/(avg_c*df_seg['total_pop'].sum())
	        _dwrel = 1E2*_dwabs/(df_seg[['c','total_pop']].prod(axis=1).sum())
	        #_dwrel = -1E2*_dwabs/(df_seg['affpop_avg_'+_haz].sum()*(avg_c/(get_rho(myCountry)*(1-1.5))))
	        _dwtot += _dwabs
	        
	        _lbl = haz_dict[_haz] if nseg == 0 else ''
	        plt.bar(2*nseg+barwidth*1.05,_dwrel,bottom=dw_bot[nseg],
	                color=sns_pal[1+nhaz],zorder=99,linewidth=0,edgecolor=None,width=barwidth,alpha=0.7,label=_lbl)        
	        
	        dk_bot[nseg]+=_dkrel
	        dw_bot[nseg]+=_dwrel
	                
	        # Annotate hazard type
	        #if nseg == 0: 
	        #    plt.annotate(haz_dict[_haz],xy=(0.05,dk_bot[nseg]-0.025),color=greys_pal[6],fontsize=7,va='top',zorder=99,style='italic')
	                    
	        # Annotate total losses per cap
	        if nhaz == n_hazards-1:
	            
	            #plt.annotate('Assets',xy=(2*nseg-0*barwidth/2,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='center',fontsize=7,zorder=99,color=greys_pal[6],style='italic')
	            #plt.annotate('Well-being',xy=(2*nseg+2*barwidth/2,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='center',fontsize=7,zorder=99,color=greys_pal[6],style='italic')

	            y_cap = max(dk_bot[nseg],dw_bot[nseg])
	            
	            ccy = get_currency(myCountry)[2]
	            _ = int(round(_dktot/df_seg['total_pop'].sum(),0))
	            _w = int(round(_dwtot/df_seg['total_pop'].sum(),0))
	            _out = 'asset loss = {} {}\nwell-being loss = {} {}\nresilience = {}%\n'.format(ccy.replace('PPP',''),_,ccy.replace('PPP',''),_w,int(round(1E2*_dktot/_dwtot,0)))

	            if nseg == 0: 
	                plt.annotate('Avg. impact ({} per cap):\n\n'.format('PPP ' if 'PPP' in ccy else '') +_out,xy=(2*nseg-0.5*barwidth/2,y_cap),
	                             ha='left',va='bottom',color=greys_pal[8],fontsize=6.5,zorder=99)
	            else: plt.annotate(_out,xy=(2*nseg-0.25*barwidth,y_cap),
	                                   ha='left',va='bottom',color=greys_pal[8],fontsize=6.5,zorder=99)
	        


	for nseg, segment in enumerate(['poor','vulnerable','secure','middleclass']):
	    plt.annotate('Assets',xy=(2*nseg-barwidth/2,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='left',fontsize=7,zorder=99,color=greys_pal[6],style='italic')
	    plt.annotate('Well-being',xy=(2*nseg+2*barwidth/4,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='left',fontsize=7,zorder=99,color=greys_pal[6],style='italic')
	        
	plt.legend()

	#plt.xlim(-0.2,3.9)
	plt.xticks([2.05*barwidth/4,2+(2.05*barwidth/4),4+(2.05*barwidth/4),6+(2.05*barwidth/4)])
	plt.gca().set_xticklabels(['Poor\n$i < \$5.50/$day','Vulnerable\n$\$5.50 < i < \$10$','Secure\n$\$10 < i < \$15$','Middle class\n$i > \$15$'],ha='center',rotation=0)
	plt.gca().tick_params(axis='x', which='major', pad=18)

	plt.ylabel('Average annual losses\n[% of total consumption]',labelpad=10,linespacing=2.)

	sns.despine(left=True)
	plt.savefig('figures/{}/consumption_loss_rel.pdf'.format(myCountry),format='pdf',bbox_inches='tight')
	plt.close('all')        


def plot_absolute_losses(myCountry,df_out,n_hazards):
	# PLOT absolute losses
	barwidth = 0.8

	for nseg, segment in enumerate(['poor','vulnerable','secure','middleclass']):

	    df_seg = df_out.loc[df_out.eval("segment=='{}'".format(segment))]
	    
	    # loop over hazards
	    dk_bot = [0,0,0,0]
	    dw_bot = [0,0,0,0]
	    for nhaz, _haz in enumerate(['HU']):
	        
	        _lbl = haz_dict[_haz] if nseg == 0 else ''
	        
	        _dk = df_seg['dk_avg_'+_haz].sum()*get_currency(myCountry)[0]
	        plt.bar(2*nseg,_dk,bottom=dk_bot[nseg],
	                color=sns_pal[1+nhaz],zorder=99,linewidth=0,edgecolor=None,width=barwidth,alpha=0.4)
	        
	        _dw = df_seg['dw_avg_'+_haz].sum()*get_currency(myCountry)[0]
	        plt.bar(2*nseg+barwidth*1.05,_dw,bottom=dw_bot[nseg],
	                color=sns_pal[1+nhaz],zorder=99,linewidth=0,edgecolor=None,width=barwidth,alpha=0.7,label=_lbl)        
	        
	        dk_bot[nseg]+=_dk
	        dw_bot[nseg]+=_dw
	                
	        # Annotate hazard type
	        #if nseg == 0: 
	        #    plt.annotate(haz_dict[_haz],xy=(2*nseg+barwidth/2+0.1,dw_bot[nseg]-0.2),color=greys_pal[7],fontsize=6.5,va='top',ha='left',zorder=99,style='italic')
	        #    
	        #    plt.draw()
	                
	        # Annotate total population
	        if nhaz == n_hazards-1:
	  
	            y_cap = dw_bot[nseg]#max(dk_bot[nseg],dw_bot[nseg])
	            
	            ccy = get_currency(myCountry)[2]
	            _ = round(1E-6*df_seg['total_pop'].sum(),1)
	            _out = '{} mil.'.format(_)

	            if nseg == 0: 
	                plt.annotate('Population:\n'+_out,xy=(2*nseg+3/2*barwidth,y_cap),
	                             ha='right',va='bottom',color=greys_pal[8],fontsize=7,zorder=99)
	            else: plt.annotate(_out,xy=(2*(nseg+1.5*barwidth/2),y_cap),
	                                   ha='right',va='bottom',color=greys_pal[8],fontsize=7,zorder=99)

	for nseg, segment in enumerate(['poor','vulnerable','secure','middleclass']):
	    plt.annotate('Assets',xy=(2*nseg-barwidth/2,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='left',fontsize=7,zorder=99,color=greys_pal[6],style='italic')
	    plt.annotate('Well-being',xy=(2*nseg+2*barwidth/4,-plt.gca().get_ylim()[1]/25),annotation_clip=False,ha='left',fontsize=7,zorder=99,color=greys_pal[6],style='italic')
	        
	plt.legend(loc='upper right')        

	plt.xlim(-0.5,8)
	plt.xticks([2.05*barwidth/4,2+(2.05*barwidth/4),4+(2.05*barwidth/4),6+(2.05*barwidth/4)])
	plt.gca().set_xticklabels(['Poor\n$i < \$5.50/$day','Vulnerable\n$\$5.50 < i < \$10$','Secure\n$\$10 < i < \$15$','Middle class\n$i > \$15$'],ha='center',rotation=0)
	plt.gca().tick_params(axis='x', which='major', pad=18)

	plt.ylabel('Average annual losses\n[{} per year]'.format(get_currency(myCountry)[1]+get_currency(myCountry)[2]),labelpad=10,linespacing=2)


	sns.despine(left=True)
	plt.savefig('figures/{}/consumption_loss_abs.pdf'.format(myCountry),format='pdf',bbox_inches='tight')
	plt.close('all')       


def middle_class_stuff(myCountry,master_df):
	# plot middle class stuff
	df_natl = pd.DataFrame({'HU':None},index=('vulnerable','secure','middleclass'))

	# plot middle class stuff
	for segment in ['vulnerable','secure','middleclass']:
	    for pt in ['cons']:
	        
	        sf = 1.#12.

	        # loop over hazards
	        for _haz in np.array(master_df.index.get_level_values(1).unique()):
	        
	            # slice to select hazard
	            _ = slice(None)
	            df = master_df.loc[(_,[_haz],_,_,_,_,_), keep_cols].sort_index()
	                
	    
	            # slice to select class (poor, vulnerable, secure, or middle class)
	            df,fom = slice_on_pop(df,segment,pt)
	    
	            # if we're looking at poverty, it's time *IN* poverty
	            # otherwise, it's time outside of initial class
	            if segment != 'poor': df[fom] = (10.019268-df[fom]).clip(lower=0)
	    
	            # get expectation values (sum over a/na/helped/not-helped)
	            df_ev = get_expectation_values(df,myCountry)
	    
	            # load entire population on class = segment
	            try: pop_df = get_population(myCountry,segment).sum(level=get_economy(myCountry)).squeeze().to_frame(name='pop')
	            except: 
	                pop_df = get_population(myCountry,segment).sum(level=get_economy(myCountry))
	                pop_df = pop_df.rename(columns={'pcwgt':'pop'})
	    
	            # get entire population of affected, for class we're considering
	            pop_mc_aff = df[['pcwgt']].sum(level=[get_economy(myCountry),'rp'])
	            assert(pop_mc_aff.shape[0]>0)
	        
	            # total displacement [time, years]
	            df_region = df[['pcwgt',fom]].prod(axis=1).sum(level=[get_economy(myCountry),'rp']).to_frame(name=fom)
	            #df_region[fom] /= pop_mc_aff['pcwgt']
	            
	            # average over rp
	            df_region_aal = average_over_rp(df_region,return_probs=False)
	            # ^ this is displacement*time * RESULT*
	            df_region_aal['frac_{}'.format(fom)] = df_region_aal[fom]/pop_df['pop']
	            # this is fraction of personXyears spent displaced

	            df_region_aal['pop_aff_{}'.format(fom)] = average_over_rp(pop_mc_aff,return_probs=False)
	            # ^ this is average annual affected population
	    
	            # get national values
	            df_natl.loc[segment,_haz] = df_region_aal[fom].sum()/pop_df['pop'].sum()        
	    
	            # Sort ascending
	            df_region_aal = df_region_aal.sort_values(fom,ascending=True).reset_index()
	    
	            # plot 
	            plt.bar(df_region_aal.index,1E-3*sf*df_region_aal[fom],color=sns_pal[1],lw=1.0,zorder=99,width=0.8,alpha=0.3)
	    
	            # label barplot    
	            rects = plt.gca().patches
	            for nrect, rect in enumerate(rects):
	                
	                try:
	                    _val = df_region_aal.iloc[nrect]['frac_{}'.format(fom)]
	                    _max = df_region_aal['frac_{}'.format(fom)].max()
	                    _col = sns_pal[0] if (_val == _max) else greys_pal[8]
	                    _wgt = 'bold'  if (_val == _max) else 'normal'
	                
	                    #print(nrect)
	                    # total population of middle class +
	                    #plt.annotate('{}'.format(int(1E-3*pop_mc_aff.iloc[nrect].squeeze())),
	                    #             xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),va='bottom',ha='center')
	            
	                    # fraction of person-years
	                    plt.annotate('{}%'.format(round(float(1E2*df_region_aal.iloc[nrect]['frac_{}'.format(fom)].squeeze()),1)),
	                                 xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),va='bottom',ha='center',fontsize=6.5,color=_col,weight=_wgt)
	                except: pass
	    
	            plt.ylabel('Average annual displacement\namong {} [,000 person-years]'.format(segment.replace('middle','middle ')),labelpad=8,fontsize=10,linespacing=1.75)
	    
	            plt.xticks(df_region_aal.index+0.4)
	        
	            plt.gca().set_xticklabels(df_region_aal[get_economy(myCountry)],ha='center',rotation=90)
	            plt.grid(True,axis='y')
	    
	            sns.despine(left=True)
	            plt.savefig('figures/{}/{}_{}_{}.pdf'.format(myCountry,pt,segment,_haz),format='pdf',bbox_inches='tight')
	            plt.close('all')

def plot_avg_annual_displacement(myCountry,ptype):
	df_natl = pd.DataFrame({'HU':None},index=('vulnerable','secure','middleclass'))
	try: df_natl = df_natl.reset_index()
	except: pass

	wid = 0.9

	labels_dict = {'vulnerable':'vulnerable',
	              'secure':'secure',
	              'middleclass':'middle class'}


	for nh,h in enumerate(['HU']):
	    plt.bar(df_natl.index*5+nh,1E2*df_natl[h],color=sns_pal[nh+1],width=wid,label=haz_dict[h]+' ')

	plt.legend()

	plt.xticks(df_natl.index*5+1.5)
	plt.yticks([0.5,1.0,1.5,2.0])
	plt.gca().tick_params(axis='both', which='major', pad=7,size=0)
	    
	plt.gca().set_xticklabels([labels_dict[i] for i in df_natl['index']],ha='center')
	plt.grid(True,axis='y')  
	    
	plt.ylabel('Average annual displacement\n[% of consumption class]',labelpad=8,fontsize=10,linespacing=1.75)
	    
	sns.despine(left=True)
	plt.savefig('figures/{}/{}_losses.pdf'.format(myCountry,ptype),format='pdf',bbox_inches='tight')
	plt.close('all')
