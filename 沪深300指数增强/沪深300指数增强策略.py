import jqdata
import pandas as pd

#from jqdata import get_fundamentals

from udf import to_date
from udf import *


## 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 设定沪深300作为基准
    g.benchmark='000300.XSHG'
    set_benchmark(g.benchmark)
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True) 
    # 设定成交量比例
    set_option('order_volume_ratio', 1)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, \
                             open_commission=0.0003, close_commission=0.0003,\
                             close_today_commission=0, min_commission=5), type='stock')
    # 基金类交易手续费是：0
    set_order_cost(OrderCost(open_tax=0, close_tax=0, \
                             open_commission=0, close_commission=0,\
                             close_today_commission=0, min_commission=0), type='fund')

    # 运行函数
    run_monthly(trade,monthday=1, time='09:30', reference_security='000300.XSHG')

## 计算股票列表和权重
def calc_stock_weight(context):

    tdate=context.current_dt
    
    components_df=get_index_weights(g.benchmark,date=tdate)
    hs300_components=list(components_df.index)
    
    
    #history(security_list=hs300_components,count=22,field='money')
    # 成分股成交量及按成交量计算的权重,最近22天成交量占比的平均
    money=get_price(security=hs300_components,count=22,end_date=tdate,fields='money',panel=False)
    money2=pd.pivot_table(money,index='code',columns=['time'],values=['money'])
    money2.columns=money2.columns.droplevel()
    weight_money=(money2.apply(lambda col:col/money2.apply(sum,axis=0)*100.0,axis=1)).apply(mean,axis=1)
    
    weight_money.sort_values(ascending=False)
    components_df['weight_money']=weight_money
    
    # 成分股申万一、二级行业
    sw_1=pd.Series({code:v['sw_l1']['industry_name']  for code,v in get_industry(security=hs300_components,date=tdate).items() if v.get('sw_l1','null')!='null'})
    sw_2=pd.Series({code:v['sw_l2']['industry_name']      for code,v in get_industry(security=hs300_components,date=tdate).items() if v.get('sw_l2','null')!='null'})
        
    components_df['sw_1']=sw_1
    components_df['sw_2']=sw_2
    
    # 成分股市值
    df_cap = get_fundamentals(query(
              valuation.code,valuation.day,valuation.circulating_market_cap,valuation.market_cap
          ).filter(
              # 这里不能使用 in 操作, 要使用in_()函数
              valuation.code.in_(hs300_components)
          ), date=tdate)
    df_cap=df_cap.set_index('code')
    components_df['market_cap']=df_cap['market_cap']
    components_df['c_market_cap']=df_cap['circulating_market_cap']
    
    # 按行业分组按股票的指数权重排名
    components_df['sw1_weight_rank']=components_df.groupby(['sw_1'])['weight'].rank(method='min',ascending=False)
    
    # 计算成分股最近3个TTM的营收及归母利润
    ttm,ttm_detail=get_fundamentals_ttm(tdate,3,hs300_components,income,['operating_revenue','np_parent_company_owners'])
    
    ttm_report_df=pd.pivot_table(ttm,index=['code'],columns='ttm_lag',values=['operating_revenue','np_parent_company_owners'],aggfunc=np.sum)
    # 列名重命名 revenue_1 表示 最近一期TTM收入，_2表示第2期TTM收入
    
    ttm_report_df.columns=['profit_1','profit_2','profit_3','revenue_1','revenue_2','revenue_3']
    
    ttm_report_df[(ttm_report_df.index=='601318.XSHG')]
    
    # 合并成宽表
    components_all=pd.concat([components_df,ttm_report_df],axis=1,sort=True)
    
    # 按申万一级行业汇总
    t=components_all.groupby('sw_1').agg({
        'weight':['sum'],'weight_money':['sum'],'market_cap':['sum'],'date':['count']
        ,'revenue_1':['sum'],'profit_1':['sum']
        ,'revenue_2':['sum'],'profit_2':['sum']
        ,'revenue_3':['sum'],'profit_3':['sum']},sort=True)
    
    # 
    t.columns=['weight','weight_money', 'market_cap', 'stock_cnt','revenue_1','profit_1','revenue_2','profit_2','revenue_3','profit_3']
    
    t_percent=t.div(t.sum())*100
    # 计算各行业占比 指数权重、成交量占比、市值占比、股市数、收入占比，利润占比
        
    t['market_cap_percent']=t_percent['market_cap']
    
    t['revenue_1_p']=t_percent['revenue_1']
    t['profit_1_p']=t_percent['profit_1']
    t['revenue_2_p']=t_percent['revenue_2']
    t['profit_2_p']=t_percent['profit_2']
    
    t['revenue_3_p']=t_percent['revenue_3']
    t['profit_3_p']=t_percent['profit_3']
    
    t.sort_values(['weight'],ascending=False)

    growth_ind=t[(t['profit_1_p']>t['profit_2_p']) & (t['profit_2_p']>t['profit_3_p']) &
     (t['revenue_1_p']>t['revenue_2_p']) & (t['revenue_2_p']>t['revenue_3_p'])]
    
    decline_ind=t[(t['profit_1_p']<t['profit_2_p']) & (t['profit_2_p']<t['profit_3_p']) &
     (t['revenue_1_p']<t['revenue_2_p']) & (t['revenue_2_p']<t['revenue_3_p'])]
   
   
    components_df_growth=components_df[[growth_ind.index.contains(i) for i in components_df['sw_1']]].sort_values(['market_cap'],ascending=False)
    components_df_notgrowth=components_df[[not growth_ind.index.contains(i) for i in components_df['sw_1']]].sort_values(['market_cap'],ascending=False)

    #components_df_notgrowth=components_df_notgrowth[[not decline_ind.index.contains(i) for i in components_df_notgrowth['industry']]]

    components_df_growth2=components_df_growth[(components_df_growth['sw1_weight_rank']<=10) & (components_df_growth['weight']>=0.1)]
    components_df_notgrowth2=components_df_notgrowth[(components_df_notgrowth['sw1_weight_rank']<=4) & (components_df_notgrowth['weight']>=0.5)]
    
    print('%.3f,%d' % (components_df_growth2['weight'].sum(),components_df_growth2['weight'].count()))

    #print(components_df_growth2.describe())
    print('%.3f,%d' % (components_df_notgrowth2['weight'].sum(),components_df_notgrowth2['weight'].count()))

    #print(components_df_notgrowth2.describe())
    components_df_alpha=components_df_growth2['weight'].append(components_df_notgrowth2['weight'])
    components_df_alpha=components_df_alpha/components_df_alpha.sum()*100
    components_df_alpha=components_df_alpha.sort_values(ascending=False)

    log.info(components_df_alpha.head())
    

    record(growth_w=components_df_growth2['weight'].sum())
    record(growth_c=components_df_growth2['weight'].count())
    record(other_w=components_df_notgrowth2['weight'].sum())
    record(other_c=components_df_notgrowth2['weight'].count())
    
    return components_df_alpha

from jqdata import jy
import pandas as pd



## 交易函数
def trade(context):
    month = context.current_dt.month
    if month not in (1,3,4,5,6,8,9,10,11,12):
        return

    #pos_ratio = calc_pos(context)
    #real_pos = context.portfolio.positions_value
    # if '511880.XSHG' in context.portfolio.positions:
    #     p = context.portfolio.positions['511880.XSHG']
    #     real_pos = real_pos - p.total_amount * p.price 
    # real_pos_ratio = real_pos/context.portfolio.total_value

    stock_weights = calc_stock_weight(context)
    #log.info(stock_weights)
    old_stocks = set(context.portfolio.positions.keys())
    old_stocks = old_stocks - set(stock_weights.index)

    if len(old_stocks) != 0:
        log.info('clear stocks :{}'.format(old_stocks))
        # 清仓不在新的沪市300权重里的股票
        for code in old_stocks:
            order_target_value(code, 0)
    # 投资组合当前总市值
    total_value = context.portfolio.total_value 

    # 股票调仓，调仓至新的权重

    def get_current_val(context,code):
        current_val=context.portfolio.positions[code].value
        return current_val
        
    stock_target_value= [ (code,weight,total_value * weight/100.0,get_current_val(context,code)) for code,weight in  stock_weights.iteritems() ]
    
    # 先卖
    for code,weight,target_value,current_val  in stock_target_value:
        if target_value<current_val: 
            log.info('%s,%d,%f,%d' % (code, total_value,weight,target_value) )
            val = int(target_value)
            order_target_value(code, target_value)
    # 再买
    for code,weight,target_value,current_val  in stock_target_value:
        if target_value>current_val: 
            log.info('%s,%d,%f,%d' % (code, total_value,weight,target_value) )
            val = int(target_value)
            order_target_value(code, target_value)     
