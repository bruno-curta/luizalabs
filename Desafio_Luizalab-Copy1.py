
# coding: utf-8

# # Desafio Luizalabs
# ## Predição de Demanda
# ### Importando dados e fazendo a análise exploratória

# In[302]:

import pandas as pd
import numpy as np


# In[303]:

#Leitura da base de dados
data = pd.read_csv('/home/notbru/Documents/Luizalabs/desafio.csv')


# In[304]:

# Checando informação das variáveis disponíveis
data.info()


# In[305]:

data.head()


# In[306]:

# convertendo a data de captura para o formato data
data['capture_date'] =  pd.to_datetime(data['capture_date'], format='%Y-%m-%d', errors='coerce')

# substituindo data zerada por missing e convertendo para formato data
data['process_date'] = data['process_date'].replace('0000-00-00', np.nan)
data['process_date'] =  pd.to_datetime(data['process_date'], format='%Y-%m-%d')


# In[307]:

data['month'] = pd.DatetimeIndex(data['capture_date']).month
data['year'] = pd.DatetimeIndex(data['capture_date']).year
data['month_year'] = data['capture_date'].apply(lambda x: x.strftime('%m-%Y'))
data['month_year'] = pd.to_datetime(data['month_year'], format= '%m-%Y')


# In[308]:

data.describe()


# In[309]:

print('Número de categorias distintas: ', len(set(data['category'])))
print('Número de itens distintos: ', len(set(data['code'])))
print('Canais distintos: ', len(set(data['source_channel'])))


# In[310]:

# criando a variável de vendas totais e preço unitário
data['total_sales'] = data['price']
data['unit_price'] = data['price']/data['quantity']


# # Visualização dos dados

# In[311]:

import seaborn as sns
import matplotlib
import squarify
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().magic('matplotlib inline')


# In[312]:

#Avaliando se a base contém dados faltantes
#A variável process_date apresentará missing dado o ajuste realizado anteriormente para converter o formato em data

msno.matrix(df=data, figsize=(20, 5), color=(0, 0, 0))


# ## Total de vendas por Status do pedido
# ### Há mais de 10% de pedidos cancelados por terem os boletos não pagos

# In[313]:

# Base agregada pelo status do pedido - ordenada pela quantidade total
dataorder = data.groupby(['order_status']).sum().sort_values('quantity', ascending=False)
dataorder['order_status'] = dataorder.index
dataorder['Imp_order_status']=dataorder['quantity']/dataorder['quantity'].sum()


# In[314]:

f, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Imp_order_status', y='order_status', data=dataorder)
ax.set(ylabel="", xlabel="Percentual de pedidos por status")


# ## Total de vendas por Canal
# ### Principal canal concentra mais de 45% das vendas

# In[315]:

# Base agregada pelo status do pedido - ordenada pela quantidade total
datachannel = data.groupby(['source_channel']).sum().sort_values('quantity', ascending=False)
datachannel['source_channel'] = datachannel.index
datachannel['Imp_source_channel']=datachannel['quantity']/dataorder['quantity'].sum()
print('O principal canal responde por ', "{:.1%}".format(datachannel['Imp_source_channel'].head(1).sum()), 'das vendas em unidades')


# In[316]:

f, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Imp_source_channel', y='source_channel', data=datachannel)
ax.set(ylabel="", xlabel="Percentual de pedidos por canal")


# ## Total de venda por Categoria
# ### Mais de 85% das vendas em unidades são de uma categoria

# In[402]:

# Base agregada pelo item - ordenada pela quantidade total
datacat = data.groupby(['category']).sum().sort_values('quantity', ascending=False)
datacat['category'] = datacat.index
datacat['Imp_cat']=datacat['quantity']/datacat['quantity'].sum()
print('Top 3 categorias respondem por ', "{:.1%}".format(datacat['Imp_cat'].head(3).sum()), 'das vendas em unidades')


# In[318]:

f, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Imp_cat', y='category', data=datacat)
ax.set(ylabel="", xlabel="Importancia das Categorias")


# ## Quantidade de itens por categoria
# ### A categoria 388128822cef4b4f102ae881e040a64b é a que apresenta maior número de itens, entretando a categoria 9a97178a18aa6333aabdfb21de182b99 tem a maior venda média por item.

# In[319]:

datacode = data[['category','code','quantity']].groupby(['category','code']).sum()
datacode.groupby('category').agg(['sum','count','mean'])


# In[320]:

datacode.sort_values('quantity', ascending=False).head(10)


# ## Total de venda por item
# ### Os top 10 itens respondem por mais de 50% das vendas

# In[321]:

# Base agregada pelo item - ordenada pela quantidade total
datacode = data.groupby(['code']).sum().sort_values('quantity', ascending=False)
datacode['code'] = datacode.index
datacode['Imp_code']=datacode['quantity']/datacode['quantity'].sum()
datacode10 = datacode.head(10)
print('Os top 10 itens respondem por ', "{:.1%}".format(datacode10['Imp_code'].sum()))


# In[322]:

f, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Imp_code', y='code', data=datacode10)
ax.set(ylabel="", xlabel="Importancia dos top 10 itens")


# ## Gráficos de séries de tempo

# In[323]:

datats = pd.crosstab(data.capture_date, data.category, data.quantity, aggfunc=sum)


# In[324]:

for i in datats.columns:
    plt.figure()
    datats[i].plot(figsize=(12, 6))
    plt.ylabel(i)


# # Respondendo às perguntas do desafio
# ## a) Separação dos produtos em grupos, usando algoritimo não supervisionado de classificação

# In[325]:

# Importando os pacotes a serem utilizados
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# ## Estudando a segmentação por item (como observações) e datas como variáveis (Features)

# In[326]:

# Transformando a base de dados para rodar a análise de cluster
# Neste caso usaremos a data como um feature e os itens como os exemplos
# Foi feita a substituição de missing por '-1'
dataclus = pd.crosstab(data.code, data.capture_date, data.quantity, aggfunc = sum).replace(np.NaN,0)
dataclus.head()


# In[344]:

# Definindo o melhor número de clusters
import random
random.seed(7)
distancia = []
K = range(1,130)
for k in K:
    km = KMeans(n_clusters=k).fit(dataclus)
    km.fit(dataclus)
    distancia.append(sum(np.min(cdist(dataclus, km.cluster_centers_, 'euclidean'), axis=1)) / dataclus.shape[0])


# In[345]:

# Plotando o gráfico de cotovelo
plt.figure(figsize=(12,5))
plt.plot(K, distancia)
plt.xlabel('Número de grupos')
plt.ylabel('Distância')
plt.title('Gŕafico de cotovelo para diferentes segmentações de grupos')
plt.xticks(np.arange(0,140, 10))
plt.show()


# In[346]:

plt.figure(figsize=(12,5))
plt.plot([j-i for i, j in zip(distancia[:-1], distancia[1:])])
plt.xticks(np.arange(0, 131, 5))
plt.show()


# ### À partir de 11 clusters a distância dentro dos grupos passa a diminuir menos, indicando que esta segmentação poderia ser uma boa opção.

# In[347]:

# Considerando que à partir de 11 clusters a distância média dentro dos clusters aumenta
# e depois volta a decrescer mais lentamente

kmeans = KMeans(n_clusters=11)


# In[348]:

clus_data = pd.DataFrame.as_matrix(dataclus)


# In[349]:

kmeans.fit(clus_data)


# In[438]:

dataclus.reset_index(inplace=True)
dataclus['cluster'] = kmeans.labels_


# In[352]:

dataclus_f = pd.merge(data, dataclus[['code','cluster']], how='left', on=['code'])
dataclus_ts = dataclus_f.groupby(['category', 'code', 'capture_date', 'cluster']).sum()
dataclus_ts.reset_index(inplace=True)


# ### Avaliando o gráfico  de dispersão dos produtos por cluster se identificam grupos bem definidos

# In[353]:

f, ax = plt.subplots(figsize=(12, 6))
ax = sns.pointplot(x='capture_date', y='quantity', hue='cluster', data=dataclus_ts, ci=None)


# ### Quantidade de produtos por cluster segue a seguinte distribuição:

# In[454]:

datagroup=dataclus_f[['code','quantity','cluster']].groupby(['cluster','code']).sum()
datagroup.groupby('cluster').agg(['sum','count','mean'])


# ### Os menores itens ficaram agrupados no cluster 0, enquanto os demais itens, maiores, ficaram praticamente sozinhos ou em pequenos grupos

# ## Alocação dos volumes (Quantidade) por categoria e cluster

# In[355]:

pd.crosstab(dataclus_f.category, dataclus_f.cluster, dataclus_f.quantity, aggfunc = sum)


# ### Gráfico das vendas por produto por cluster

# In[356]:

for i in np.unique(dataclus_ts['cluster']):
    f, ax = plt.subplots(figsize=(12, 6))
    ax = sns.pointplot(x='capture_date', y='quantity', hue='category', data=dataclus_ts[dataclus_ts['cluster']==i], ci=None)
    plt.title('Cluster '+str(i))
    fig.autofmt_xdate()
    myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=1))


# ### O cluster 0 concentra itens de todas categorias e em geral itens com vendas menores. O cluster 1 contém o maior item da categoria *9a97178a18aa6333aabdfb21de182b99* e os demais clusters são diferentes itens da maior categoria, a *388128822cef4b4f102ae881e040a64b*

# ## b) Previsão de vendas para os meses de junho, julho e agosto

# ### Limpando os itens cancelados da base

# In[357]:

data = data[(data['order_status'] == 'entrega total') | (data['order_status'] =='em rota de entrega') | (data['order_status'] =='entrega parcial')]


# In[545]:

data = data[(data['capture_date'] != '2017-06-01')]


# In[546]:

dataclus_f = pd.merge(data, dataclus[['code','cluster']], how='left', on=['code'])
dataclus_ts = dataclus_f.groupby(['category', 'code', 'capture_date', 'cluster']).sum()
dataclus_ts.reset_index(inplace=True)


# ### Ajuste da base para a modelagem

# In[547]:

dataprophet = pd.crosstab(dataclus_ts.capture_date, dataclus_ts.cluster, dataclus_ts.quantity, aggfunc=sum)


# In[548]:

dataprophet.tail()


# ## Gráfico das vendas por cluster

# In[360]:

fig, ax = plt.subplots(figsize=(20,7))
ax.plot(dataprophet)
fig.autofmt_xdate()
myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=7))
plt.show()


# In[362]:

# Checando as datas "Outliers" de cada cluster
dataprophet[10].sort_values(ascending=False).head(10)


# In[363]:

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


# In[364]:

bf = pd.DataFrame({
  'holiday': 'blackfriday',
  'ds': pd.to_datetime(['2016-11-25','2016-11-24']),
  'lower_window': 0,
  'upper_window': 1,
})
ny = pd.DataFrame({
  'holiday': 'newyear',
  'ds': pd.to_datetime(['2017-01-06','2017-01-07']),
  'lower_window': 0,
  'upper_window': 1,
})
events = pd.concat((bf, ny))


# In[549]:

result={}
dataprophet_cv={}
error={}
for i in dataprophet.columns:
    x=pd.DataFrame(dataprophet[i])
    x.reset_index(inplace=True)
    x.columns = ['ds','y']
    x['y']=np.log(x['y'])
    m=Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, holidays=events, interval_width=0.95)
    m.fit(x)
    future=m.make_future_dataframe(periods=91)
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)
    result[i]=pd.merge(forecast[['ds', 'yhat_lower', 'yhat_upper','yhat']],x,how='left', on=['ds'])
    
    #Cross validation
    dataprophet_cv[i]=cross_validation(m, horizon = '31 days')
    


# In[550]:

for i in range(0,len(result)):
    
    #convertendo os dados de volta para a unidade original
    result[i]['y']=np.exp(result[i]['y'])
    result[i]['yhat']=np.exp(result[i]['yhat'])
    result[i]['yhat_lower']=np.exp(result[i]['yhat_lower'])
    result[i]['yhat_upper']=np.exp(result[i]['yhat_upper'])


# In[552]:

result_f={}
for i in range(0,len(result)):
    
    #Agrupando por mês
    result[i]['month_year']=result[i]['ds'].apply(lambda x: x.strftime('%m-%Y'))
    result[i]['month_year']=pd.to_datetime(result[i]['month_year'], format= '%m-%Y')
    result_f[i]=result[i].groupby('month_year').sum()
    result_f[i].replace(np.nan,0)


# ### Plotando os gráficos com valores originais (y), valor estimado (yhat) e limites de confiança superior (yhat_upper) e inferior (yhat_lower) - com 95% de confiança

# In[558]:

for i in range(0,len(result)):
        
    #plot
    plt.figure(figsize=(20,7))
    result_f[i].plot()
    plt.title('cluster '+str(i))
    plt.show()


# In[554]:

from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[555]:

MAE={}
MAPE={}
for i in range(0,len(dataprophet_cv)):
    dataprophet_cv[i]['y']=np.exp(dataprophet_cv[i]['y'])
    dataprophet_cv[i]['yhat']=np.exp(dataprophet_cv[i]['yhat'])
    MAE[i]=mean_absolute_error(dataprophet_cv[i]['y'],dataprophet_cv[i]['yhat'])
    MAPE[i]=mean_absolute_percentage_error(dataprophet_cv[i]['y'],dataprophet_cv[i]['yhat'])


# In[524]:

MAPE


# ### Apesar dos gráficos apresentarem tendências que visualmente façam sentido, os erros das estimativas sairam bastante altos

# ## c) Análise dos resultados encontrados

# ### Os dados fornecidos para esta análise estavam completos, não sendo necessária a aplicação de um método de imputação ou descarte de observações. Ainda assim, após as análises realizadas foram excluídas as compras canceladas, fraudes, entre outras opções que não caracterizavam a entrega total do produto (cerca de 10% de toda a base de dados), de forma a evitar a previsão de venda de um produto que não seria de fato realizado.

# ### Esta base de dados estava caracterizada pela presença de 11 categorias de produtos, 131 itens, vendidos em 16 diferentes canais de comercialização.

# ### Os itens e categorias apresentam uma grande dispersão, sendo que a categoria mais importante representa mais de 85% de todo volume (quantidade) vendido, concentrando também a maior parte dos itens do estudo. Os top 10 itens concentram mais de 50% do volume (Quantidade). Alguns itens aparentam ser mais sazonais, com períodos intermitentes de vendas. Muitos itens apresentam picos de vendas em momentos específicos do ano, como na Black Friday e no primeiro final de semana após a virada do ano (na liquidação anual do Magazine Luiza).

# ### Por conta desta dispersão e variabilidade entre os itens (produtos), foi considerado fazer um agrupamento por análise de cluster para usar usá-los na predição de demanda dos meses de Junho à Agosto 2017. Para esta análise foi utilizado o método Kmeans, modelando apenas as vendas em quantidade, a mesma variável que posteriormente seria utilizada para uma projeção de demanda através da aplicação de uma modelagem por séries temporais.

# ### Os dados foram arranjados de forma que tivessemos os itens como observações (Examples) e as datas de compra como variáveis (Features). Ao montar a tabela cruzada com as observações nas linhas e variáveis em colunas, os dados faltantes foram completados com 0 (venda igual a zero - ausência de vendas).

# ### A análise de cluster pelo Kmeans refere-se a um método de classificação no qual as observações são agrupadas de acordo com sua similaridade (neste caso, menor distância euclidiana das vendas). Espera-se que cada grupo seja homogêneo entre si e heterogêneo se comparado aos demais.

# ### Para definir o número ideal de clusters, utilizou-se o método do cotovelo (Elbow Method). Quando as distâncias entre as observações e seu respectivo cluster não apresentam uma redução significativa ao adicionar uma segmentação adicional, então este é um indicativo de que aumentar o número de clusters não deve ajudar a explicar melhor os grupos. Para este caso chegamos ao número ótimo de 11 grupos.

# ### Analisando os resultados, percebe-se que as categorias menores e mais esparsas ficaram agrupadas no cluster 0, juntamente com itens menores das categorias de maior relevância. Os maiores itens ficaram separados sozinhos ou em grupos menores.

# ### Após classificar os grupos passamos para a preparação da base para modelagem por séries temporais. Para esta tarefa foi utilizado o pacote fbprophet, desenvolvido pelo Facebook para projeções de forma mais rápidas e automatizadas (https://facebook.github.io/prophet/).

# ### Este pacote decompõe a série histórica em sazonalidades anuais, semanais, diárias e eventos especiais. Para nosso caso, dado que não havia diferenciação por hora do dia, não foi utilizada a decomposição diária. Entretanto, dado os claros eventos sazonais, como Black Friday e Liquidação anual Magazine Luiza, incluiram-se estas datas como eventos sazonais.

# ### Os modelos foram rodados considerando-se um intervalo de confiança de 95% e a previsão para três meses (91 dias), cubrindo a solicitação para fazer a previsão de demanda para os meses de Junho, Julho e Agosto de 2017.

# ### O pacote fbprophet traz a opção de gerar uma base de validação cruzada (Cross Validation), entretanto os resultados obtidos com o cálculo do MAPE mostram que os erros das estimativas são bastante altos (acima de 96%) e ainda que visualmente as estimativas acompanhem as tendências históricas, existe um risco grande em tomar uma decisão com estes resultados.

# ### Como próximos passos, seria aconselhável buscar metodologias alternativas como modelos VAR multivariados, modelos econométricos ou um maior estudo dos modelos apresentados para chegar em erros mais satisfatórios.

# In[ ]:



