import streamlit as st
import plotly.express as px
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
from scipy import stats

st.title("Data Analysis Dashboard")

def to_snake(name):
    name = name.strip().replace(' ($B)','')
    name = name.lower()
    name = name.replace(' ','_')
    return name

def get_metadata():
  df = pd.read_csv('companies_metadata.csv')[['Company','Funding','Year Founded']]
  df.columns = ['company','funding','year_founded']
  df['funding'] = df['funding'].apply(lambda x: x if x!='Unknown' else np.nan)
  df['funding'] = df['funding'].str.replace('$','').str.replace('B','').str.replace('M','').astype(float)
  return df

@st.cache_resource
def load_data():
    df = pd.read_csv('./unicorns_companies.csv')
    df.columns = df.columns.map(to_snake)
    df['valuation'] = df['valuation'].str.replace('$', '').astype(float)
    df['date_joined'] = pd.to_datetime(df['date_joined'])
    df['date_joined'] = df['date_joined'].dt.year
    df.loc[df['company']=='LinkSure Network', 'investors'] = (
        'Bank of China Group Investment, China Merchants Innovation '
        'Investment Management, and Hopu Fund'
    )
    df['investors'] = df['investors'].fillna(df['industry'])
    df.loc[df['investors'] == df['industry'], 'industry'] = df['city']
    df.loc[df['industry'] == df['investors'], 'industry'] = df['city']
    df.loc[df['city'] == df['industry'], 'city'] = df['country']
    df['industry'] = df['industry'].str.capitalize()
    df['city'] = df['city'].str.title()
    df['country'] = df['country'].str.title()
    df['company'] = df['company'].str.title()
    metadata = get_metadata()
    metadata['company'] = metadata['company'].str.title()
    df = pd.merge(
        df,
        metadata,
        how='left',
        on='company'
    )
    return df

try:
    unicorns_merged = load_data()
    # st.write("Data loaded OK")  # debug

    top_valuation = unicorns_merged.nlargest(5, 'valuation')[['company', 'valuation']]
    st.table(top_valuation)
    fig = px.bar(
        top_valuation,
        x='valuation',
        y='company',
        orientation='h',
        title='Top Unicorns by Valuation',
        labels = {'valuation':'Valuation ($B)','company':'Company'}
    )
    st.plotly_chart(fig, key="top_valuation_chart")


    bins = [0,1,2,5,10,20,50,100,500]
    labels = ['1-2B','2-5B','5-10B','10-20B','20-50B','50-100B','100-500B','500+']
    unicorns_merged['valuation_bands'] = pd.cut(unicorns_merged['valuation'], bins = bins, labels = labels)
    unicorn_merged = unicorns_merged['valuation_bands'].value_counts().sort_index()
    band_counts = unicorns_merged['valuation_bands'].value_counts().sort_index()
    band_df = band_counts.reset_index()
    band_df.columns = ['valuation_band','count']

    fig_bar = px.bar(
        band_df,
        x='count',
        y='valuation_band',
        orientation='h',
        title='Unicorn Count by Valuation Band',
        labels = {'count':'Number of Unicorns','valuation_band':'Valuation Band'}
    )
    st.plotly_chart(fig_bar, key="bar_chart")

    fig_barv = px.bar(
        band_df,
        x="valuation_band",
        y="count",
        title="Unicorn Count by Valuation Band",
        labels={"count": "Number of Unicorns", "valuation_band": "Valuation Band"},
    )
    fig_barv.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_barv)

    st.write(stats.skew(unicorns_merged['valuation'].values))

    fig_pie = px.pie(
        band_df,
        names="valuation_band",
        values="count",
        title="Valuation Band Proportions",
    )
    st.plotly_chart(fig_pie, key="pie_chart")

    year_counts = unicorns_merged['year_founded'].value_counts().sort_index()
    year_df = year_counts.reset_index()
    year_df.columns = ['year_founded','count']
    fig_year = px.line(
        year_df,
        x='year_founded',
        y='count',
        title='Unicorn Foundings over time',
        labels = {'year_founded':'Founded Year','count':'Number of Unicorns'},
        markers=True
    )
    st.plotly_chart(fig_year, key="line_chart")

    val_by_ind_year = unicorns_merged.groupby(['date_joined','industry'])['valuation'].sum().reset_index()
    fig_val_by_ind_year_heatnmap = px.density_heatmap(
        val_by_ind_year,
        x='date_joined',
        y='industry',
        z='valuation',
        color_continuous_scale='Blues',
        title='Total unicorn valuation by industry over the years'
    )

    st.plotly_chart(fig_val_by_ind_year_heatnmap)

    fig_val_by_ind_year_line = px.line(
        val_by_ind_year,
        x='date_joined',
        y='valuation',
        color='industry',
        title='Total unicorn valuation by industry over the years',
        markers=True
    )
    st.plotly_chart(fig_val_by_ind_year_line, key="line_chart_industry")

    industry1 = unicorns_merged[unicorns_merged['industry']=='Artificial intelligence']['valuation']
    industry2 = unicorns_merged[unicorns_merged['industry']=='Internet']['valuation']
    t_stat, p_value = ttest_ind(industry1, industry2, equal_var=False)

    unicorns_merged['roi'] = unicorns_merged['valuation']/unicorns_merged['funding']
    top_roi = unicorns_merged.nlargest(5, 'roi')[['company', 'roi']]

    fig_roi = px.bar(
        top_roi,
        x='roi',
        y='company',
        orientation='h',
        title='Top Unicorns by ROI',
        labels = {'roi':'ROI ($B)','company':'Company'}
    )
    st.plotly_chart(fig_roi, key="roi_chart")

    city_industry_counts = (
        unicorns_merged
        .groupby(['city', 'industry'])
        .size()
        .reset_index(name='count')
    )

    city_industry_counts_fig = px.treemap(
        city_industry_counts,
        path=['city', 'industry'],
        values='count',
        title='City-Industry Unicorn Hubs'
    )
    st.plotly_chart(city_industry_counts_fig,   key="treemap_chart")

    city_industry_counts_tree = px.treemap(
        city_industry_counts,
        path=['city', 'industry'],
        values='count',
        title='City-Industry Unicorn Hubs'
    )
    st.plotly_chart(city_industry_counts_tree)

    all_investors = unicorns_merged['investors'].str.split(', ').explode().str.strip().value_counts()
    top_investors = all_investors.head(12).reset_index()
    top_investors.columns = ['investor','count']
    top_investors_fig_barv = px.bar(
        top_investors,
        x="investor",
        y="count",
        title="Top 12 Investors",
        labels={"investor": "Investor", "count": "Count"},
    )
    top_investors_fig_barv.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(top_investors_fig_barv)
except Exception as e:
    st.error(f"Error in load_data: {e}")
